"""
Base model training on the UBC Fashion Dataset using Wan2.1.
Replaces Stable Diffusion v1.4 with:
  - WanTransformer3DModel  (DiT backbone, extended with pose channels)
  - AutoencoderKLWan       (3-D causal spatio-temporal VAE)
  - WanEmbeddingAdapter    (CLIP + VAE → cross-attention tokens)
  - FlowMatchEulerDiscreteScheduler
"""

import itertools
import math
import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from transformers import CLIPProcessor, CLIPVisionModel

logger = get_logger(__name__)

from utils.parse_args import parse_args
from datasets.ubc_dataset import DreamPoseDataset
from models.wan_pose_transformer import get_wan_transformer, WanEmbeddingAdapter


# ── helpers ────────────────────────────────────────────────────────────────

def vae_norm_params(vae, device, dtype):
    mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    std  = torch.tensor(vae.config.latents_std,  device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    return mean, std


def encode_latents(vae, images, weight_dtype, vae_mean, vae_std):
    """
    images : [B, 3, H, W] in [−1, 1]
    Returns normalised Wan-VAE latents [B, 16, 1, H/8, W/8].
    """
    video = images.to(dtype=torch.float32).unsqueeze(2)          # [B, 3, 1, H, W]
    raw   = vae.encode(video).latent_dist.sample()               # [B, 16, 1, H', W']
    return (raw - vae_mean) / vae_std


def latents2img(latents, vae, vae_mean, vae_std):
    raw    = latents * vae_std + vae_mean
    images = vae.decode(raw).sample                              # [B, 3, 1, H, W]
    images = images[:, :, 0]                                     # [B, 3, H, W]
    images = (images / 2 + 0.5).clamp(0, 1)
    images = (images * 255).round().byte().cpu().numpy()
    return images


def inputs2img(tensor):
    images = (tensor / 2 + 0.5).clamp(0, 1)
    images = (images * 255).round().byte().detach().cpu().numpy()
    return images


def visualize_dp(im, dp):
    im  = im.transpose((1, 2, 0))
    hsv = np.zeros(im.shape, dtype=np.uint8)
    hsv[..., 1] = 255
    dp  = dp.cpu().detach().numpy()
    mag, ang = cv2.cartToPolar(dp[0], dp[1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr.transpose((2, 0, 1))


# ── main ───────────────────────────────────────────────────────────────────

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    writer      = SummaryWriter(f'results/logs/{args.run_name}')

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # ── Load CLIP image encoder (frozen) ───────────────────────────────
    clip_encoder  = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
    clip_encoder.requires_grad_(False)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # ── Load Wan VAE (frozen) ──────────────────────────────────────────
    vae = AutoencoderKLWan.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    vae.requires_grad_(False)

    # ── Load Wan transformer (trainable) ──────────────────────────────
    transformer = get_wan_transformer(args.pretrained_model_name_or_path, n_poses=5)

    if args.custom_chkpt is not None:
        print("Loading transformer checkpoint:", args.custom_chkpt)
        state = torch.load(args.custom_chkpt, map_location="cpu")
        new_state = OrderedDict(
            (k.replace("module.", ""), v) for k, v in state.items()
        )
        transformer.load_state_dict(new_state)
    transformer = transformer.cuda()

    # ── Load embedding adapter (trainable) ────────────────────────────
    cross_attn_dim = getattr(
        transformer.config, "cross_attn_dim",
        getattr(transformer.config, "cross_attention_dim", 4096),
    )
    adapter = WanEmbeddingAdapter(
        clip_dim=768, vae_channels=16, cross_attn_dim=cross_attn_dim
    )

    if args.custom_chkpt is not None:
        adapter_path = args.custom_chkpt.replace("unet_epoch", "adapter")
        if os.path.exists(adapter_path):
            print("Loading adapter checkpoint:", adapter_path)
            state = torch.load(adapter_path, map_location="cpu")
            new_state = OrderedDict(
                (k.replace("module.", ""), v) for k, v in state.items()
            )
            adapter.load_state_dict(new_state)
    adapter = adapter.cuda()

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        itertools.chain(transformer.parameters(), adapter.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # FlowMatchEulerDiscreteScheduler replaces DDPMScheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    train_dataset = DreamPoseDataset(
        instance_data_root=args.instance_data_dir,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        size=args.resolution,
        center_crop=args.center_crop,
    )

    def collate_fn(examples):
        frame_i = [e["frame_i"] for e in examples]
        frame_j = [e["frame_j"] for e in examples]
        poses   = [e["pose_j"]  for e in examples]

        if args.with_prior_preservation:
            frame_i += [e["class_frame_i"] for e in examples]
            frame_j += [e["class_frame_j"] for e in examples]
            poses   += [e["class_pose_j"]  for e in examples]

        frame_i = torch.cat(frame_i, 0)
        frame_j = torch.cat(frame_j, 0)
        poses   = torch.cat(poses,   0)

        # Classifier-free guidance dropout
        p = random.random()
        if   p <= args.dropout_rate / 3:
            poses   = torch.zeros_like(poses)
        elif p <= 2 * args.dropout_rate / 3:
            frame_i = torch.zeros_like(frame_i)
        elif p <= args.dropout_rate:
            poses   = torch.zeros_like(poses)
            frame_i = torch.zeros_like(frame_i)

        return {
            "frame_i": frame_i.to(memory_format=torch.contiguous_format).float(),
            "frame_j": frame_j.to(memory_format=torch.contiguous_format).float(),
            "poses":   poses.to(memory_format=torch.contiguous_format).float(),
        }

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=1,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    transformer, adapter, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, adapter, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=torch.float32)

    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("dreampose_wan", config=vars(args))

    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    logger.info("***** Running training (Wan2.1 backend) *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs   = {args.num_train_epochs}")
    logger.info(f"  Total batch  = {total_batch_size}")
    logger.info(f"  Total steps  = {args.max_train_steps}")

    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")
    global_step = 0

    # Pre-compute VAE normalisation params once
    vae_mean, vae_std = vae_norm_params(vae, accelerator.device, torch.float32)

    for epoch in range(args.epoch, args.num_train_epochs):
        transformer.train()
        adapter.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # ── Encode target frame to Wan latents ──────────────────
                # Normalise images from [0,1] → [−1,1] for VAE
                frame_j = (batch["frame_j"] * 2 - 1).to(accelerator.device)
                latents = encode_latents(vae, frame_j, weight_dtype, vae_mean, vae_std)
                latents = latents.to(dtype=weight_dtype)

                # ── Flow-matching noise interpolation ───────────────────
                noise = torch.randn_like(latents)
                bsz   = latents.shape[0]

                # Sample random timesteps (integers 0 … num_train_timesteps)
                t_idx = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                # sigma ∈ [0, 1]: fraction of noise
                sigma = (
                    t_idx.float() / noise_scheduler.config.num_train_timesteps
                ).view(bsz, 1, 1, 1, 1)

                # Noisy latents: linear interpolation between clean and noise
                noisy_latents = (1 - sigma) * latents + sigma * noise

                # ── Concatenate pose channels to noisy latents ──────────
                _, _, lt, lh, lw = noisy_latents.shape
                pose_input = F.interpolate(
                    batch["poses"].to(accelerator.device, dtype=weight_dtype),
                    (lh, lw),
                    mode="bilinear",
                    align_corners=False,
                ).unsqueeze(2)                                      # [B, 2n, 1, H', W']
                noisy_latents = torch.cat([noisy_latents, pose_input], dim=1)

                # ── Build CLIP + VAE conditioning ────────────────────────
                frame_i = (batch["frame_i"] * 2 - 1).to(accelerator.device)
                clip_inputs = clip_processor(
                    images=list(frame_i.cpu()),
                    return_tensors="pt",
                )
                clip_inputs = {k: v.to(accelerator.device) for k, v in clip_inputs.items()}
                clip_hidden = clip_encoder(**clip_inputs).last_hidden_state.to(weight_dtype)

                vae_hidden = encode_latents(vae, frame_i, weight_dtype, vae_mean, vae_std)
                vae_hidden = vae_hidden[:, :, 0]                    # [B, 16, H', W']

                encoder_hidden_states = adapter(clip_hidden, vae_hidden.to(weight_dtype))

                # ── Transformer forward ──────────────────────────────────
                v_pred = transformer(
                    noisy_latents, timestep=t_idx, encoder_hidden_states=encoder_hidden_states
                ).sample

                # Flow-matching target: velocity = noise − clean_latents
                # (only over the latent channels, not the appended pose channels)
                target = noise - latents

                if args.with_prior_preservation:
                    v_pred_main, v_pred_prior = torch.chunk(v_pred, 2, dim=0)
                    target_main, target_prior = torch.chunk(target, 2, dim=0)
                    loss = (
                        F.mse_loss(v_pred_main.float(), target_main.float(), reduction="none")
                        .mean([1, 2, 3, 4])
                        .mean()
                        + args.prior_loss_weight
                        * F.mse_loss(v_pred_prior.float(), target_prior.float(), reduction="mean")
                    )
                else:
                    loss = F.mse_loss(v_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        itertools.chain(transformer.parameters(), adapter.parameters()),
                        args.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            # ── Logging ──────────────────────────────────────────────────
            writer.add_scalar("loss/train", loss.detach().item(), global_step)

            if global_step % 50 == 0:
                with torch.no_grad():
                    clean_latents = noisy_latents[:, :16, :, :, :]   # first 16 channels
                    pred_images   = latents2img(clean_latents - v_pred, vae, vae_mean, vae_std)
                    target_images = inputs2img(batch["frame_j"])
                    input_img     = inputs2img(batch["frame_i"])
                    middle_pose   = visualize_dp(target_images[0], batch["poses"][0][4:6])
                    frame_viz = np.concatenate(
                        [input_img[0], middle_pose, pred_images[0], target_images[0]], axis=2
                    )
                    writer.add_image("train/pred_img", frame_viz, global_step=global_step)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            # ── Save checkpoint ──────────────────────────────────────────
            if accelerator.is_main_process and global_step % args.save_steps == 0:
                transformer_path = os.path.join(args.output_dir, f"transformer_epoch_{epoch}.pth")
                adapter_path     = os.path.join(args.output_dir, f"adapter_{epoch}.pth")
                torch.save(accelerator.unwrap_model(transformer).state_dict(), transformer_path)
                torch.save(accelerator.unwrap_model(adapter).state_dict(),     adapter_path)

        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
