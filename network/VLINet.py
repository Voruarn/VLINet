import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np
from timm.models.layers import DropPath, trunc_normal_
from network.Modules import ConvNextModel, MutliScaleFusion, TextEncoder
from network.SODMaskDecoder import SODMaskDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VLINet(nn.Module):
    def __init__(self, visual_encoder_name='convnext_base', text_dim=512, dec_dim=256):
        super().__init__()
        # 1. 视觉编码器
        self.visual_encoder = ConvNextModel(model_name=visual_encoder_name)
        self.embed_dims = self.visual_encoder.cur_embed_dims  # [128,256,512,1024]
        
        # 2. 多尺度融合模块
        self.rgb_msf = MutliScaleFusion(self.embed_dims, dec_dim)
        self.depth_msf = MutliScaleFusion(self.embed_dims, dec_dim)
        
        # 3. Mask Decoder
        self.sam_decoder = SODMaskDecoder(
            text_embed_dim=self.embed_dims[-1],
            transformer_dim=dec_dim
        )
        
        # 模型参数自动移到指定设备
        self.to(device)

    def forward(self, rgb, depth, texts,):
        B, _, H_orig, W_orig = rgb.shape
        
        # Step 1: 提取原始多尺度特征
        multi_modal_feats = self.visual_encoder(rgb, depth, texts)
        visual_feats = multi_modal_feats['visual']  # [V1,C1; V2,C2; V3,C3; V4,C4]
        depth_feats = multi_modal_feats['depth']    # [D1,C1; D2,C2; D3,C3; D4,C4]
        out_VL = multi_modal_feats['out_VL']
        out_DL = multi_modal_feats['out_DL']

        text_fuse = out_VL + out_DL
        # Step 2: U型多尺度融合（到C1层）
        fused_rgb_c1 = self.rgb_msf(visual_feats)    # (B,256,H/4,W/4)
        fused_depth_c1 = self.depth_msf(depth_feats) # (B,256,H/4,W/4)
        
        # Step 3:  Decoder前向
        pred = self.sam_decoder(
            visual_feat=fused_rgb_c1,
            dense_prompt_feat=fused_depth_c1,
            text_feat=text_fuse,
            orig_size=(H_orig, W_orig)
        )
        
        return pred
