from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from common import DEVICE, IMG_SIZE, NUM_CLASSES

class PatchEmbed(nn.Module):
    """Split image into patches and project to embedding dim."""
    def __init__(self, img_size: Tuple[int, int], patch_size: int, emb_dim: int) -> None:
        super().__init__()
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.proj = nn.Conv2d(
            in_channels=3,
            out_channels=emb_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) → (B, emb_dim, H/ps, W/ps)
        x = self.proj(x)
        # flatten patches: (B, emb_dim, N_h, N_w) → (B, N, emb_dim)
        return x.flatten(2).transpose(1, 2)

class TransformerEncoderBlock(nn.Module):
    """Single Transformer encoder block: MHSA → FFN."""
    def __init__(self, emb_dim: int, num_heads: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, B, emb_dim) for nn.MultiheadAttention
        y = self.norm1(x)
        # Self-attention; query=key=value=y
        y, _ = self.attn(y, y, y)
        x = x + y
        # Feed-forward
        y = self.norm2(x)
        y = self.mlp(y)
        return x + y

class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_size: int,
        emb_dim: int,
        depth: int,
        num_heads: int,
        mlp_dim: int,
        num_classes: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, emb_dim)
        # Class token + positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.num_patches, emb_dim)
        )
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(emb_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_dim)

        # Classification head
        self.head = nn.Linear(emb_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.patch_embed(x)                             # (B, N, emb_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)       # (B, 1, emb_dim)
        x = torch.cat((cls_tokens, x), dim=1)               # (B, 1+N, emb_dim)
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer expects sequence first: (seq_len, B, emb_dim)
        x = x.transpose(0, 1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Extract [CLS] token embedding
        cls_final = x[0]                                     # (B, emb_dim)
        return self.head(cls_final)

def build_vit_model(
    num_classes: int,
    img_size: Tuple[int, int],
    device: torch.device,
    patch_size: int = 16,
    emb_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    mlp_dim: int = 3072,
    dropout: float = 0.1
) -> nn.Module:
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        emb_dim=emb_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        num_classes=num_classes,
        dropout=dropout
    )
    return model.to(device)

def build_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.05
) -> optim.Optimizer:
    # AdamW is commonly used for ViTs
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

if __name__ == "__main__":
    model = build_vit_model(NUM_CLASSES, IMG_SIZE, DEVICE)
    optimizer = build_optimizer(model)
