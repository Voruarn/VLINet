import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
import math
from typing import Tuple, List
import clip
import numpy as np
from network.convnextVLI import convnext_tiny, convnext_small, convnext_base
import clip

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvNextModel(nn.Module):
    embed_dims = {
        "convnext_tiny": [96, 192, 384, 768],    # c1, c2, c3, c4
        "convnext_small": [96, 192, 384, 768],
        "convnext_base": [128, 256, 512, 1024]
    }
    def __init__(self, model_name='convnext_base', pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.cur_embed_dims = self.embed_dims[model_name]  
        
        self.convnext = eval(model_name)(pretrained=pretrained)
        
        self.depth_adapter = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)

        nn.init.kaiming_normal_(self.depth_adapter.weight, mode='fan_out', nonlinearity='relu')
        if self.depth_adapter.bias is not None:
            nn.init.constant_(self.depth_adapter.bias, 0)

    def forward(self, rgb, depth, text):
        """
        Args:
            rgb: (B, 3, H, W)
            depth: (B, 1, H, W) 
        """
        depth_3ch = self.depth_adapter(depth)  # (B, 3, H, W)
        
        out_V, out_VL = self.convnext(rgb, text)
        V1, V2, V3, V4 = out_V
        out_D, out_DL = self.convnext(depth_3ch, text)
        D1, D2, D3, D4 = out_D
        
        return {
            'visual':[V1, V2, V3, V4],
            'depth': [D1, D2, D3, D4],
            'out_VL':out_VL,
            'out_DL':out_DL,
        }


class TextEncoder(nn.Module):
    def __init__(self, pretrained_model="ViT-B/16"):
        super().__init__()
        self.clip_model = clip.load(pretrained_model, device=device)[0]  
        self.clip_model.eval()
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        self.output_dim = 512 

    def forward(self, texts):
        text_tokens = clip.tokenize(texts, truncate=True).to(device)
        
        # 编码（无梯度计算）
        with torch.no_grad():
            text_feats = self.clip_model.encode_text(text_tokens)
        
        # 1. 补全文本序列维度：(B, 512) → (B, 1, 512)
        text_feats = text_feats.unsqueeze(1)  # 关键：添加序列长度维度L=1
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        return text_feats
    
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.ReLU()  # default activation
    # default_act=nn.ReLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
    

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class MutliScaleFusion(nn.Module):
    def __init__(self, in_dims, out_dim): 
        super().__init__()
        self.conv1 = Bottleneck(in_dims[0] + in_dims[1], out_dim)
        self.conv2 = Bottleneck(in_dims[1] + in_dims[2], in_dims[1])
        self.conv3 = Bottleneck(in_dims[2] + in_dims[3], in_dims[2])

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, feats):
        c1,c2,c3,c4=feats

        up3 = torch.cat([c3, self.upsample2(c4)], dim=1)
        up3 = self.conv3(up3)

        up2 = torch.cat([c2, self.upsample2(up3)], dim=1)
        up2 = self.conv2(up2)

        up1 = torch.cat([c1, self.upsample2(up2)], dim=1)
        up1 = self.conv1(up1)

        return up1
    






