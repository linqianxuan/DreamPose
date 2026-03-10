"""
Subject-specific Wan-transformer fine-tuning for DreamPose (Wan2.1 backend).
Replaces the SD UNet fine-tuning with WanTransformer3DModel fine-tuning.

Usage:
    accelerate launch finetune-unet.py \
        --pretrained_model_name_or_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
        --instance_data_dir=demo/sample/train \
        --output_dir=demo/custom-chkpts \
        --resolution=512 \
        --train_batch_size=1 \
        --gradient_accumulation_steps=1 \
        --learning_rate=1e-5 \
        --num_train_epochs=500 \
        --dropout_rate=0.0 \
        --custom_chkpt=checkpoints/transformer_epoch_20.pth
"""

import itertools
import math
import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPProcessor, CLIPVisionModel

logger = get_logger(__name__)

from utils.parse_args import parse_args
from datasets.dreampose_dataset import DreamPoseDataset
from models.wan_pose_transformer import get_wan_transformer, WanEmbeddingAdapter


# ── helpers ────────────────────────────────────────────────────────────────

def vae_norm_params(vae, device, dtype):
    mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    std  = torch.tensor(vae.config.latents_std,  device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    return mean, std


def encode_latents(vae, images, vae_mean, vae_std):
    """images: [B, 3, H, W] in [−1,1] → [B, 16, 1, H/8, W/8] normalised"""
    video = images.to(dtype=torch.float32).unsqueeze(2)
    raw   = vae.encode(video).latent_dist.sample()
    return (raw - vae_mean) / vae_std


def latents2img(latents, vae, vae_mean, vae_std):
    raw    = latents * vae_std + vae_mean
    images = vae.decode(raw).sample[:, :, 0]   # [B, 3, H, W]
    images = (images / 2 + 0.5).clamp(0, 1)
    return (images * 255).round().byte().cpu().numpy()


def inputs2img(tensor):
    images = (tensor / 2 + 0.5).clamp(0, 1)
    return (images * 255).round().byte().detach().cpu().numpy()


# ── main ───────────────────────────────────────────────────────────────────

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

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

    # ── CLIP encoder (frozen) ──────────────────────────────────────────
    clip_encoder  = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
    clip_encoder.requires_grad_(False)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # ── Wan VAE (frozen) ───────────────────────────────────────────────
    vae = AutoencoderKLWan.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    vae.requires_grad_(False)

    # ── Wan transformer (trainable) ────────────────────────────────────
    transformer = get_wan_transformer(args.pretrained_model_name_or_path, n_poses=5)

    if args.custom_chkpt is not None:
        print("Loading transformer checkpoint:", args.custom_chkpt)
        state = torch.load(args.custom_chkpt, map_location="cpu")
        new_state = OrderedDict(
            (k.replace("module.", ""), v) for k, v in state.items()
        )
        transformer.load_state_dict(new_state)
    transformer = transformer.cuda()

    # ── Embedding adapter (trainable) ─────────────────────────────────
    cross_attn_dim = getattr(
        transformer.config, "cross_attn_dim",
        getattr(transformer.config, "cross_attention_dim", 4096),
    )
    adapter = WanEmbeddingAdapter(
        clip_dim=768, vae_channels=16, cross_attn_dim=cross_attn_dim
    )

    if args.custom_chkpt is not None:
        adapter_path = args.custom_chkpt.replace("unet_epoch", "adapter").replace(
            "transformer_epoch", "adapter"
        )
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

    optimizer_class = torch.optim.AdamW
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_class = bnb.optim.AdamW8bit

    optimizer = optimizer_class(
        itertools.chain(transformer.parameters(), adapter.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

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
        frame_j = [e["frame_i"] for e in examples]   # fine-tune: reconstruct same frame
        poses   = [e["pose_j"]  for e in examples]

        frame_i = torch.stack(frame_i, 0)
        frame_j = torch.stack(frame_j, 0)
        poses   = torch.stack(poses,   0)

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
        accelerator.init_trackers("dreampose_finetune_wan", config=vars(args))

    logger.info("***** Fine-tuning transformer (Wan2.1) *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs   = {args.num_train_epochs}")
    logger.info(f"  Total steps  = {args.max_train_steps}")

    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")
    global_step = 0

    vae_mean, vae_std = vae_norm_params(vae, accelerator.device, torch.float32)

    for epoch in range(args.epoch, args.num_train_epochs):
        transformer.train()
        adapter.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # Encode target frame
                frame_j = (batch["frame_j"] * 2 - 1).to(accelerator.device)
                latents = encode_latents(vae, frame_j, vae_mean, vae_std).to(weight_dtype)

                # Flow-matching interpolation
                noise = torch.randn_like(latents)
                bsz   = latents.shape[0]
                t_idx = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                sigma = (
                    t_idx.float() / noise_scheduler.config.num_train_timesteps
                ).view(bsz, 1, 1, 1, 1)
                noisy_latents = (1 - sigma) * latents + sigma * noise

                # Append pose channels
                _, _, lt, lh, lw = noisy_latents.shape
                pose_input = F.interpolate(
                    batch["poses"].to(accelerator.device, dtype=weight_dtype),
                    (lh, lw),
                    mode="bilinear",
                    align_corners=False,
                ).unsqueeze(2)
                noisy_latents = torch.cat([noisy_latents, pose_input], dim=1)

                # CLIP + VAE conditioning
                frame_i = (batch["frame_i"] * 2 - 1).to(accelerator.device)
                clip_inputs = clip_processor(images=list(frame_i.cpu()), return_tensors="pt")
                clip_inputs = {k: v.to(accelerator.device) for k, v in clip_inputs.items()}
                clip_hidden = clip_encoder(**clip_inputs).last_hidden_state.to(weight_dtype)

                vae_hidden = encode_latents(vae, frame_i, vae_mean, vae_std)[:, :, 0]
                encoder_hidden_states = adapter(clip_hidden, vae_hidden.to(weight_dtype))

                # Forward
                v_pred = transformer(
                    noisy_latents, timestep=t_idx, encoder_hidden_states=encoder_hidden_states
                ).sample

                target = noise - latents
                loss   = F.mse_loss(v_pred.float(), target.float(), reduction="mean")

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

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            # Save
            if accelerator.is_main_process and global_step % args.save_steps == 0:
                transformer_path = os.path.join(
                    args.output_dir, f"transformer_epoch_{epoch}.pth"
                )
                adapter_path = os.path.join(args.output_dir, f"adapter_{epoch}.pth")
                torch.save(
                    accelerator.unwrap_model(transformer).state_dict(), transformer_path
                )
                torch.save(
                    accelerator.unwrap_model(adapter).state_dict(), adapter_path
                )

        accelerator.wait_for_everyone()

    # Final save
    if accelerator.is_main_process:
        transformer_path = os.path.join(args.output_dir, f"transformer_epoch_{epoch}.pth")
        adapter_path     = os.path.join(args.output_dir, f"adapter_{epoch}.pth")
        torch.save(accelerator.unwrap_model(transformer).state_dict(), transformer_path)
        torch.save(accelerator.unwrap_model(adapter).state_dict(),     adapter_path)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
