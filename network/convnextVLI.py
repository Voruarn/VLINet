import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        # 默认参数补全
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # 全连接层定义
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # MLP前向传播：fc1 → act → drop → fc2 → drop
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class VisionLanguageInteraction(nn.Module):

    def __init__(self, visual_dim, text_dim, dropout=0.1, num_heads=4):
        super().__init__()

        self.text_proj = nn.Linear(text_dim, visual_dim)
        
        self.attn_vis = nn.MultiheadAttention(visual_dim, num_heads=num_heads, batch_first=True)
        self.attn_lang = nn.MultiheadAttention(visual_dim, num_heads=num_heads, batch_first=True)

        self.ffn_vis = Mlp(visual_dim, hidden_features=visual_dim * 4, drop=dropout)
        self.ffn_lang = Mlp(visual_dim, hidden_features=visual_dim * 4, drop=dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """ Custom weight initialization for linear layers. """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming正态初始化（针对ReLU/GELU等非线性激活）
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # Bias参数初始化为0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, visual_feat, text_feat):
        B, C, H, W = visual_feat.shape
        # 视觉特征：(B, C, H, W) → (B, H, W, C) → (B, H*W, C)（展平空间维度为序列长度）
        visual_feat_perm = visual_feat.permute(0, 2, 3, 1)  # 交换通道维度与空间维度
        visual_feat_flat = visual_feat_perm.reshape(B, H * W, C)  # 展平为序列格式

        # 文本特征：(B, text_dim) → (B, visual_dim) → (B, 1, visual_dim)（扩展为序列长度1）
        text_feat_expand = self.text_proj(text_feat)  # 维度对齐

        out_lang, _ = self.attn_lang(text_feat_expand, visual_feat_flat, visual_feat_flat)
        out_lang = self.ffn_lang(out_lang)  # 文本特征经过FFN增强

        out_vis, _ = self.attn_vis(visual_feat_flat, out_lang, out_lang)
        out_vis = self.ffn_vis(out_vis) 

        out_vis = out_vis.reshape(B, H, W, C)
        out_vis = out_vis.permute(0, 3, 1, 2)
        out_lang = out_lang + text_feat_expand

        return out_vis, out_lang


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1., text_dim=512,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        self.vlis = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
            if i>0:
                text_dim=dims[i-1]
            self.vlis.append(VisionLanguageInteraction(visual_dim=dims[i], text_dim=text_dim))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x, x_lang):
        outputs=[]
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            xi, x_lang = self.vlis[i](x, x_lang)
            x = x + xi
            outputs.append(x)
        return outputs, x_lang

    def forward(self, x, x_lang):
        outputs, x_lang = self.forward_features(x, x_lang)
        return outputs, x_lang


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

# @register_model
def convnext_tiny(pretrained=False,in_22k=True, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)

        pretrained_dict = checkpoint["model"]
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 
                           k in model_dict and model_dict[k].shape == v.shape}
        
        # 更新当前模型的状态字典
        model_dict.update(pretrained_dict)
        # 加载更新后的状态字典
        model.load_state_dict(model_dict)

    return model

# @register_model
def convnext_small(pretrained=False,in_22k=True, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        pretrained_dict = checkpoint["model"]
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 
                           k in model_dict and model_dict[k].shape == v.shape}
        
        # 更新当前模型的状态字典
        model_dict.update(pretrained_dict)
        # 加载更新后的状态字典
        model.load_state_dict(model_dict)
    return model

# @register_model
def convnext_base(pretrained=False, in_22k=True, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        pretrained_dict = checkpoint["model"]
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 
                           k in model_dict and model_dict[k].shape == v.shape}
        
        # 更新当前模型的状态字典
        model_dict.update(pretrained_dict)
        # 加载更新后的状态字典
        model.load_state_dict(model_dict)
    return model

# @register_model
def convnext_large(pretrained=False, in_22k=True, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        pretrained_dict = checkpoint["model"]
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 
                           k in model_dict and model_dict[k].shape == v.shape}
        
        # 更新当前模型的状态字典
        model_dict.update(pretrained_dict)
        # 加载更新后的状态字典
        model.load_state_dict(model_dict)
    return model

# @register_model
def convnext_xlarge(pretrained=False, in_22k=True, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        pretrained_dict = checkpoint["model"]
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 
                           k in model_dict and model_dict[k].shape == v.shape}
        
        # 更新当前模型的状态字典
        model_dict.update(pretrained_dict)
        # 加载更新后的状态字典
        model.load_state_dict(model_dict)
    return model
