"""
WanTransformer3DModel wrapper and WanEmbeddingAdapter for DreamPose.
Replaces models/unet_dual_encoder.py for the Wan2.1 backend.
"""

import torch
import torch.nn as nn
from einops import rearrange
from diffusers.models import WanTransformer3DModel


def get_wan_transformer(model_id, n_poses=5):
    """
    Load WanTransformer3DModel and extend its patch embedding to accept
    2*n_poses additional DensePose UV-map channels concatenated to the
    noisy video latents.

    Args:
        model_id : HuggingFace model ID, e.g. "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        n_poses  : Number of DensePose frames to condition on (each contributes 2 channels).

    Returns:
        WanTransformer3DModel with modified patch_embedding.
    """
    transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer")

    original_in_channels = transformer.config.in_channels
    new_in_channels = original_in_channels + 2 * n_poses

    pe = transformer.patch_embedding

    if isinstance(pe, nn.Conv3d):
        new_pe = nn.Conv3d(
            new_in_channels,
            pe.out_channels,
            kernel_size=pe.kernel_size,
            stride=pe.stride,
            padding=pe.padding,
            bias=pe.bias is not None,
        )
        with torch.no_grad():
            new_pe.weight[:, :original_in_channels] = pe.weight.clone()
            new_pe.weight[:, original_in_channels:] = 0.0
            if pe.bias is not None:
                new_pe.bias = nn.Parameter(pe.bias.clone())

    elif isinstance(pe, nn.Linear):
        patch_size = transformer.config.patch_size
        if isinstance(patch_size, (list, tuple)):
            patch_dim = patch_size[0] * patch_size[1] * patch_size[2]
        else:
            patch_dim = int(patch_size) ** 3
        extra = patch_dim * 2 * n_poses
        new_pe = nn.Linear(pe.in_features + extra, pe.out_features, bias=pe.bias is not None)
        with torch.no_grad():
            new_pe.weight[:, :pe.in_features] = pe.weight.clone()
            new_pe.weight[:, pe.in_features:] = 0.0
            if pe.bias is not None:
                new_pe.bias = nn.Parameter(pe.bias.clone())

    else:
        raise ValueError(f"Unrecognised patch_embedding type: {type(pe)}")

    transformer.patch_embedding = new_pe
    return transformer


class WanEmbeddingAdapter(nn.Module):
    """
    Fuses CLIP image embeddings and Wan-VAE spatial features into
    cross-attention-compatible tokens for WanTransformer3DModel.

    WanTransformer3DModel expects encoder_hidden_states shaped
    [B, seq_len, cross_attn_dim], matching the UMT5-XXL text-encoder
    output dimension (cross_attn_dim = 4096 for Wan2.1).

    Inputs
    ------
    clip : [B, n_clip_tokens, clip_dim]   — e.g. [B, 50, 768] from CLIP ViT-B/32
    vae  : [B, vae_channels, H', W']      — e.g. [B, 16, 80, 64] from Wan VAE encoder

    Output
    ------
    [B, n_clip_tokens + n_vae_spatial², cross_attn_dim]
    """

    def __init__(
        self,
        clip_dim: int = 768,
        vae_channels: int = 16,
        cross_attn_dim: int = 4096,
        n_clip_tokens: int = 50,
        n_vae_spatial: int = 16,      # pool VAE to n_vae_spatial × n_vae_spatial grid
    ):
        super().__init__()
        self.n_vae_tokens = n_vae_spatial * n_vae_spatial

        self.vae_pool  = nn.AdaptiveAvgPool2d((n_vae_spatial, n_vae_spatial))
        self.clip_proj = nn.Linear(clip_dim, cross_attn_dim)
        self.vae_proj  = nn.Linear(vae_channels, cross_attn_dim)
        self.norm      = nn.LayerNorm(cross_attn_dim)

    def forward(self, clip: torch.Tensor, vae: torch.Tensor) -> torch.Tensor:
        """
        clip : [B, n_clip_tokens, clip_dim]
        vae  : [B, vae_channels, H', W']

        Returns [B, n_clip_tokens + n_vae_spatial², cross_attn_dim]
        """
        clip_tokens = self.clip_proj(clip)                               # [B, 50, D]

        vae_pooled  = self.vae_pool(vae)                                 # [B, C, s, s]
        vae_tokens  = rearrange(vae_pooled, 'b c h w -> b (h w) c')     # [B, s², C]
        vae_tokens  = self.vae_proj(vae_tokens)                          # [B, s², D]

        combined = torch.cat([clip_tokens, vae_tokens], dim=1)           # [B, 50+s², D]
        return self.norm(combined)
