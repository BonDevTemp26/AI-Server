"""VideoMAEv2-compatible Vision Transformer for video classification.

Self-contained (torch-only) re-implementation of the fine-tuning architecture
from the official repo (OpenGVLab/VideoMAEv2, ``models/modeling_finetune.py``).
Parameter names and shapes match the released checkpoints, so the official
K710-distilled weights (``vit_s/b_k710_dl_from_giant.pth``) and any checkpoint
produced by ``training/train_videomae.py`` load directly.

Input:  float tensor ``(B, 3, T, H, W)`` normalized with ImageNet mean/std.
Output: logits ``(B, num_classes)``.
"""

from __future__ import annotations

import logging
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as torch_ckpt

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Stochastic depth per sample (when applied to the main path of residuals)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Attention(nn.Module):
    """Multi-head self-attention with VideoMAE's split q/v bias convention.

    The qkv projection is a bias-free Linear plus separate ``q_bias``/``v_bias``
    parameters (k bias fixed to zero) — this matches the official checkpoints.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop_p = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((
                self.q_bias,
                torch.zeros_like(self.v_bias, requires_grad=False),
                self.v_bias,
            ))
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, heads, N, hd)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            scale=self.scale,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path_rate=0.0, init_values=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Video to tubelet-patch embedding via 3D convolution."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 num_frames=16, tubelet_size=2):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.num_patches = (num_frames // tubelet_size) * (img_size // patch_size) ** 2
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x):
        # (B, C, T, H, W) -> (B, N, embed_dim)
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


def get_sinusoid_encoding_table(n_position: int, d_hid: int) -> torch.Tensor:
    """Fixed sine/cosine position table, shape (1, n_position, d_hid)."""
    position = torch.arange(n_position, dtype=torch.float32).unsqueeze(1)
    dim_idx = torch.arange(d_hid, dtype=torch.float32)
    angle = position / torch.pow(10000.0, 2.0 * torch.div(dim_idx, 2, rounding_mode="floor") / d_hid)
    table = angle.clone()
    table[:, 0::2] = torch.sin(angle[:, 0::2])
    table[:, 1::2] = torch.cos(angle[:, 1::2])
    return table.unsqueeze(0)


class VisionTransformer(nn.Module):
    """VideoMAE / VideoMAEv2 fine-tuning backbone + classification head."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 710,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values: float | None = None,
        use_learnable_pos_emb: bool = False,
        all_frames: int = 16,
        tubelet_size: int = 2,
        use_mean_pooling: bool = True,
        grad_checkpointing: bool = False,
        head_init_scale: float = 0.001,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.grad_checkpointing = grad_checkpointing

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, num_frames=all_frames, tubelet_size=tubelet_size,
        )
        num_patches = self.patch_embed.num_patches

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            # Fixed sinusoid table; excluded from the state dict on purpose —
            # the official checkpoints do not carry it either.
            self.register_buffer(
                "pos_embed",
                get_sinusoid_encoding_table(num_patches, embed_dim),
                persistent=False,
            )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path_rate=dpr[i],
                init_values=init_values, norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        if isinstance(self.head, nn.Linear):
            nn.init.trunc_normal_(self.head.weight, std=0.02)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed.to(x.dtype).detach()
        x = self.pos_drop(x)

        for blk in self.blocks:
            if self.grad_checkpointing and self.training and not torch.jit.is_scripting():
                x = torch_ckpt.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        if self.fc_norm is not None:
            return self.fc_norm(x.mean(1))
        return self.norm(x)[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


# ── Factories ────────────────────────────────────────────────────────────────

def vit_small_patch16_224(**kwargs) -> VisionTransformer:
    return VisionTransformer(patch_size=16, embed_dim=384, depth=12, num_heads=6,
                             mlp_ratio=4.0, qkv_bias=True, **kwargs)


def vit_base_patch16_224(**kwargs) -> VisionTransformer:
    return VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12,
                             mlp_ratio=4.0, qkv_bias=True, **kwargs)


def vit_large_patch16_224(**kwargs) -> VisionTransformer:
    return VisionTransformer(patch_size=16, embed_dim=1024, depth=24, num_heads=16,
                             mlp_ratio=4.0, qkv_bias=True, **kwargs)


MODEL_REGISTRY = {
    "vit_small_patch16_224": vit_small_patch16_224,
    "vit_base_patch16_224": vit_base_patch16_224,
    "vit_large_patch16_224": vit_large_patch16_224,
}


# ── Checkpoint loading ───────────────────────────────────────────────────────

_STRIP_PREFIXES = ("module.", "_orig_mod.", "backbone.", "encoder.")
_DROP_PREFIXES = ("decoder.", "encoder_to_decoder.", "mask_token")


def _extract_state_dict(ckpt: dict) -> dict:
    for key in ("model", "module", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    return ckpt


def _normalize_key(key: str) -> str | None:
    for prefix in _DROP_PREFIXES:
        if key.startswith(prefix):
            return None
    changed = True
    while changed:
        changed = False
        for prefix in _STRIP_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    # MMAction2-style classification head
    if key.startswith("cls_head.fc_cls."):
        key = key.replace("cls_head.fc_cls.", "head.")
    return key


def load_videomae_checkpoint(model: VisionTransformer, ckpt_path: str,
                             drop_head_on_mismatch: bool = True) -> dict:
    """Load an official VideoMAEv2 (or our fine-tuned) checkpoint into ``model``.

    Handles ``model``/``module``/``state_dict`` wrappers, common key prefixes,
    MAE-pretrain checkpoints (encoder-only weights), and a classifier head of a
    different size (dropped so fine-tuning starts from a fresh head).

    Returns a report dict with ``missing`` / ``unexpected`` / ``dropped`` keys.
    """
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:  # older checkpoints with pickled non-tensor metadata
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    raw_state = _extract_state_dict(ckpt)
    state, dropped = {}, []
    for k, v in raw_state.items():
        nk = _normalize_key(k)
        if nk is None:
            dropped.append(k)
            continue
        state[nk] = v

    own = model.state_dict()
    for k in ("head.weight", "head.bias"):
        if k in state and k in own and state[k].shape != own[k].shape:
            if not drop_head_on_mismatch:
                raise ValueError(
                    f"{k}: checkpoint has {tuple(state[k].shape)}, model expects "
                    f"{tuple(own[k].shape)}. Set drop_head_on_mismatch=True to fine-tune."
                )
            dropped.append(f"{k} (shape {tuple(state[k].shape)} -> new head)")
            state.pop(k)

    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [m for m in missing if m != "pos_embed"]

    report = {"missing": missing, "unexpected": list(unexpected), "dropped": dropped}
    backbone_missing = [m for m in missing if not m.startswith("head.")]
    if backbone_missing:
        logger.warning("Checkpoint %s: missing backbone keys: %s", ckpt_path, backbone_missing[:8])
    logger.info(
        "Loaded %s: %d tensors (%d dropped, %d missing, %d unexpected)",
        ckpt_path, len(state), len(dropped), len(missing), len(unexpected),
    )
    return report
