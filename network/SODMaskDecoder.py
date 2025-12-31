import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        skip_first_layer_pe: bool = False
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        # 禁用batch_first，显式处理维度顺序
        self.self_attn = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=False)
        self.norm1 = nn.LayerNorm(embedding_dim)
        
        self.cross_attn_token_to_image = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=False)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        self.mlp = MLP(embedding_dim, mlp_dim, embedding_dim, num_layers=2)
        self.norm3 = nn.LayerNorm(embedding_dim)
        
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=False)
        
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self,
        queries: torch.Tensor,  # Token序列: (B, N_token, C)
        keys: torch.Tensor,     # 图像特征序列: (B, N_img, C)
        query_pe: torch.Tensor, # Token位置编码: (B, N_token, C)
        key_pe: torch.Tensor    # 图像位置编码: (B, N_img, C)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 维度转换：(B, seq_len, C) → (seq_len, B, C)
        q = queries.permute(1, 0, 2)          
        k_img = keys.permute(1, 0, 2)         
        q_pe = query_pe.permute(1, 0, 2)      
        k_img_pe = key_pe.permute(1, 0, 2)    

        # 1. Token自注意力
        if self.skip_first_layer_pe:
            attn_out, _ = self.self_attn(q, q, q)  
        else:
            attn_out, _ = self.self_attn(q + q_pe, q + q_pe, q)  
        attn_out = attn_out.permute(1, 0, 2)  
        queries = self.norm1(attn_out + queries)

        # 2. Token→Image 交叉注意力
        q = queries.permute(1, 0, 2)
        attn_out, _ = self.cross_attn_token_to_image(q + q_pe, k_img + k_img_pe, k_img)
        attn_out = attn_out.permute(1, 0, 2)
        queries = self.norm2(attn_out + queries)

        # 3. Token MLP层
        mlp_out = self.mlp(queries)
        queries = self.norm3(mlp_out + queries)

        # 4. Image→Token 交叉注意力
        q = queries.permute(1, 0, 2)
        k_img = keys.permute(1, 0, 2)
        attn_out, _ = self.cross_attn_image_to_token(k_img + k_img_pe, q + q_pe, q)
        attn_out = attn_out.permute(1, 0, 2)  
        keys = self.norm4(attn_out + keys)

        return queries, keys


class SODMaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        text_embed_dim: int = 512,       
        transformer_dim: int = 256,      
        num_heads: int = 8,
        num_multimask_outputs: int = 0,  # 显著目标检测仅需单掩码
        activation: nn.Module = nn.GELU,
    ):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.num_multimask_outputs = num_multimask_outputs
        self.orig_size = None  # 原始图像尺寸

        self.text_proj = nn.Linear(text_embed_dim, transformer_dim)
        
        self.mask_tokens = nn.Embedding(1 + num_multimask_outputs, transformer_dim)
        
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
            for _ in range(1 + num_multimask_outputs)
        ])

        self.transformer = nn.ModuleList([
            TwoWayAttentionBlock(
                embedding_dim=transformer_dim,
                num_heads=num_heads,
                mlp_dim=transformer_dim * 8,
                skip_first_layer_pe=(i == 0)
            ) for i in range(2)
        ])

        # 8. Token位置编码
        self.pe_layer = nn.Embedding(512, transformer_dim)
        
        # 9. 2D图像位置编码生成（内置，无需外部传入）
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            torch.randn((2, transformer_dim // 2)),
            persistent=False,
        )

    def _get_2d_pe(self, feat_h: int, feat_w: int) -> torch.Tensor:
        """生成2D图像位置编码（适配VDLNet的特征图尺寸）"""
        # 生成坐标网格
        y_coord = torch.linspace(-1, 1, feat_h, device=self.positional_encoding_gaussian_matrix.device)
        x_coord = torch.linspace(-1, 1, feat_w, device=self.positional_encoding_gaussian_matrix.device)
        y_grid, x_grid = torch.meshgrid(y_coord, x_coord, indexing="ij")
        coords = torch.stack([x_grid, y_grid], dim=-1)  # (H, W, 2)
        
        # 高斯位置编码
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * torch.pi * coords  # (H, W, C//2)
        pe = torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)  # (H, W, C)
        
        # 适配batch维度并调整顺序: (H,W,C) → (B,C,H,W)
        pe = pe.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        return pe

    def forward(
        self,
        visual_feat: torch.Tensor,          # fused_rgb_c1: (B, 256, H/4, W/4)
        dense_prompt_feat: torch.Tensor,    # fused_depth_c1: (B, 256, H/4, W/4)
        text_feat: torch.Tensor,            # texts: (B, text_dim=512) 或 (B, T, 512)
        orig_size: Tuple[int, int]          # 原始图像尺寸: (H_orig, W_orig)
    ) -> torch.Tensor:
        B, C, H_feat, W_feat = visual_feat.shape
        self.orig_size = orig_size

        # 1.1 RGB视觉特征展平: (B, C, H, W) → (B, H*W, C)
        visual_flat = visual_feat.permute(0, 2, 3, 1).reshape(B, H_feat*W_feat, C)
        dense_flat = dense_prompt_feat.permute(0, 2, 3, 1).reshape(B, H_feat*W_feat, C)
        
        # 1.3 文本特征适配（兼容一维/二维输入）
        if len(text_feat.shape) == 2:  # (B, 512) → 扩展为(B, 1, 512)
            text_feat = text_feat.unsqueeze(1)
        text_proj = self.text_proj(text_feat)  # (B, T, 256)
  
        # 1.4 生成图像位置编码（内置，无需外部传入）
        image_pe = self._get_2d_pe(H_feat, W_feat)  # (1, C, H, W)
        image_pe = image_pe.expand(B, -1, -1, -1)   # (B, C, H, W)
        image_pe_flat = image_pe.permute(0, 2, 3, 1).reshape(B, H_feat*W_feat, C)  # (B, H*W, C)

        # --------------------------
        # 2. 构建Token序列
        # --------------------------
        # 掩码Token: (1, 1, C) → (B, 1, C)
        N_mask = 1 + self.num_multimask_outputs
        mask_tokens = self.mask_tokens.weight.reshape(1, N_mask, C).expand(B, -1, C)
        
        # 合并Token: 掩码Token + 文本Token
        tokens = torch.cat([mask_tokens, text_proj], dim=1)  # (B, 1+T, C)
        N_token = tokens.shape[1]
        
        # Token位置编码
        token_pe = self.pe_layer.weight[:N_token, :].reshape(1, N_token, C).expand(B, -1, C)

        # --------------------------
        # 3. Transformer双向注意力
        # --------------------------
        image_embedding = visual_flat + dense_flat
        for block in self.transformer:
            tokens, image_embedding = block(
                queries=tokens,
                keys=image_embedding,
                query_pe=token_pe,
                key_pe=image_pe_flat
            )

        image_embedding = image_embedding.reshape(B, H_feat, W_feat, C).permute(0, 3, 1, 2)
        # --------------------------
        # 5. 掩码生成
        # --------------------------
        mask_tokens_out = tokens[:, :N_mask, :]
        upscaled_embedding = self.output_upscaling(image_embedding)  # (B, 32, 4*H_feat, 4*W_feat)
        
        # 生成单掩码（显著目标检测）
        hypernet = self.output_hypernetworks_mlps[0]
        mask_features = hypernet(mask_tokens_out[:, 0, :])  # (B, 32)
        mask_features = mask_features[:, :, None, None]     # (B, 32, 1, 1)
        mask = (mask_features * upscaled_embedding).sum(dim=1, keepdim=True)  # (B, 1, 4H_feat, 4W_feat)

        pred = F.interpolate(
            mask, 
            size=orig_size, 
            mode='bilinear', 
            align_corners=False
        )  # (B, 1, H_orig, W_orig)

        return pred

