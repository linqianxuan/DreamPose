"""
WanDreamPosePipeline: pose-guided image-to-video pipeline built on Wan2.1.
Replaces pipelines/dual_encoder_pipeline.py for the Wan2.1 backend.
"""

from typing import Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
from diffusers.models import WanTransformer3DModel
from transformers import CLIPProcessor, CLIPVisionModel

from models.wan_pose_transformer import WanEmbeddingAdapter


_TO_TENSOR = transforms.Compose([
    transforms.ToTensor(),            # [0, 1]
    transforms.Normalize([0.5], [0.5]),  # [−1, 1]
])


def _pil_to_tensor(images):
    """Convert PIL Image(s) to [B, 3, H, W] float tensor in [−1, 1]."""
    if isinstance(images, Image.Image):
        images = [images]
    return torch.stack([_TO_TENSOR(im.convert("RGB")) for im in images])


class WanDreamPosePipeline:
    """
    Pose-guided image-to-video synthesis pipeline based on Wan2.1.

    Given a key-frame image and a batch of DensePose UV maps it produces
    the corresponding synthesised frame using the Wan DiT backbone with
    dual-guidance classifier-free guidance (image guidance s1 + pose
    guidance s2), matching the original DreamPose guidance scheme.

    Components
    ----------
    vae           : AutoencoderKLWan              – 3-D causal spatio-temporal VAE
    transformer   : WanTransformer3DModel         – DiT backbone (pose-extended)
    adapter       : WanEmbeddingAdapter           – CLIP+VAE → cross-attn tokens
    scheduler     : FlowMatchEulerDiscreteScheduler
    image_encoder : CLIPVisionModel               – appearance encoder
    clip_processor: CLIPProcessor
    """

    def __init__(
        self,
        vae: AutoencoderKLWan,
        transformer: WanTransformer3DModel,
        adapter: WanEmbeddingAdapter,
        scheduler: FlowMatchEulerDiscreteScheduler,
        image_encoder: CLIPVisionModel,
        clip_processor: CLIPProcessor,
    ):
        self.vae           = vae
        self.transformer   = transformer
        self.adapter       = adapter
        self.scheduler     = scheduler
        self.image_encoder = image_encoder
        self.clip_processor = clip_processor

        self._latents_mean = None
        self._latents_std  = None

    # ── helpers ────────────────────────────────────────────────────────────

    @property
    def device(self):
        return next(self.transformer.parameters()).device

    @property
    def dtype(self):
        return next(self.transformer.parameters()).dtype

    def _vae_norm(self, device, dtype):
        if self._latents_mean is None:
            self._latents_mean = torch.tensor(
                self.vae.config.latents_mean, device=device, dtype=dtype
            ).view(1, -1, 1, 1, 1)
            self._latents_std = torch.tensor(
                self.vae.config.latents_std, device=device, dtype=dtype
            ).view(1, -1, 1, 1, 1)
        return self._latents_mean, self._latents_std

    def _encode_latents(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        image_tensor : [B, 3, H, W] in [−1, 1]
        Returns normalised Wan-VAE latents [B, 16, 1, H/8, W/8].
        """
        video = image_tensor.to(device=self.vae.device, dtype=torch.float32).unsqueeze(2)
        raw   = self.vae.encode(video).latent_dist.sample()   # [B, 16, 1, H', W']
        mean, std = self._vae_norm(raw.device, raw.dtype)
        return (raw - mean) / std

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        latents : [B, 16, T, H', W'] normalised
        Returns video tensor [B, 3, T, H, W] in [−1, 1].
        """
        mean, std = self._vae_norm(latents.device, latents.dtype)
        raw = latents * std + mean
        return self.vae.decode(raw).sample

    def _clip_embed(self, images, uncond: bool = False) -> torch.Tensor:
        """
        images : PIL Image or list of PIL Images
        Returns CLIP last_hidden_state [B, seq, 768].
        """
        if uncond:
            if isinstance(images, list):
                images = [Image.new("RGB", im.size, 0) for im in images]
            else:
                images = Image.new("RGB", images.size, 0)
        inputs = self.clip_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.image_encoder(**inputs).last_hidden_state   # [B, 50, 768]

    # ── inference ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def __call__(
        self,
        image: Union[Image.Image, List[Image.Image]],
        pose: torch.Tensor,               # [B, 2*n_poses, H, W]
        num_inference_steps: int = 50,
        s1: float = 7.5,                  # image guidance scale
        s2: float = 3.0,                  # pose  guidance scale
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
        callback: Optional[Callable] = None,
        callback_steps: int = 1,
    ):
        device = self.device
        dtype  = self.dtype

        # ── 1. Encode reference image ───────────────────────────────────
        image_list   = [image] if isinstance(image, Image.Image) else image
        image_tensor = _pil_to_tensor(image_list).to(device)          # [B, 3, H, W]
        ref_latents  = self._encode_latents(image_tensor).to(dtype)   # [B, 16, 1, H', W']
        B, C, T, H, W = ref_latents.shape

        # ── 2. Build CLIP+VAE conditioning tokens ───────────────────────
        clip_cond   = self._clip_embed(image_list).to(dtype)           # [B, 50, 768]
        clip_uncond = self._clip_embed(image_list, uncond=True).to(dtype)
        vae_cond    = ref_latents[:, :, 0]                             # [B, 16, H', W']
        vae_uncond  = torch.zeros_like(vae_cond)

        do_cfg = s1 > 1.0 or s2 > 0.0
        if do_cfg:
            # stack [uncond | img_only | full(img+pose)] for one transformer pass
            hs_uncond  = self.adapter(clip_uncond, vae_uncond)
            hs_img     = self.adapter(clip_cond,   vae_cond)
            encoder_hs = torch.cat([hs_uncond, hs_img, hs_img])       # [3B, seq, D]
        else:
            encoder_hs = self.adapter(clip_cond, vae_cond)             # [B, seq, D]

        # ── 3. Prepare pose channels ────────────────────────────────────
        pose       = pose.to(device, dtype=dtype)
        pose_lat   = F.interpolate(pose, (H, W), mode='bilinear', align_corners=False)
        pose_lat   = pose_lat.unsqueeze(2)                             # [B, 2n, 1, H', W']
        zero_pose  = torch.zeros_like(pose_lat)

        # ── 4. Initialise noise ─────────────────────────────────────────
        noise   = torch.randn(B, C, T, H, W, generator=generator, device=device, dtype=dtype)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        latents = noise.clone()

        # ── 5. Denoising loop ───────────────────────────────────────────
        for i, t in enumerate(self.scheduler.timesteps):
            if do_cfg:
                li_uncond = torch.cat([latents, zero_pose], dim=1)
                li_img    = torch.cat([latents, zero_pose], dim=1)
                li_full   = torch.cat([latents, pose_lat],  dim=1)
                model_in  = torch.cat([li_uncond, li_img, li_full])    # [3B, C+2n, T, H', W']
                t_in = t.unsqueeze(0).expand(3 * B)
            else:
                model_in = torch.cat([latents, pose_lat], dim=1)       # [B, C+2n, T, H', W']
                t_in = t.unsqueeze(0).expand(B)

            v_pred = self.transformer(
                model_in,
                timestep=t_in,
                encoder_hidden_states=encoder_hs,
            ).sample

            if do_cfg:
                v_uncond, v_img, v_full = v_pred.chunk(3)
                v_pred = v_uncond + s1 * (v_img - v_uncond) + s2 * (v_full - v_img)

            latents = self.scheduler.step(v_pred, t, latents, return_dict=False)[0]

            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        # ── 6. Decode ───────────────────────────────────────────────────
        frames = self._decode_latents(latents)        # [B, 3, 1, H, W]
        frames = frames[:, :, 0]                      # [B, 3, H, W]
        frames = (frames / 2 + 0.5).clamp(0, 1)
        frames = (frames * 255).byte().permute(0, 2, 3, 1).cpu().numpy()

        if output_type == "pil":
            return [Image.fromarray(f) for f in frames]
        return frames
