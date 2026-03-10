"""
Subject-specific Wan VAE decoder fine-tuning for DreamPose (Wan2.1 backend).
Only the VAE decoder is trained; the encoder and transformer remain frozen.

Usage:
    accelerate launch --num_processes=1 finetune-vae.py \
        --pretrained_model_name_or_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
        --instance_data_dir=demo/sample/train \
        --output_dir=demo/custom-chkpts \
        --resolution=512 \
        --train_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --learning_rate=5e-5 \
        --num_train_epochs=1500 \
        --run_name finetuning/wan-vae
"""

import itertools
import math
import os
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKLWan
from diffusers.optimization import get_scheduler
from PIL import Image

logger = get_logger(__name__)

from utils.parse_args import parse_args
from datasets.train_vae_dataset import DreamPoseDataset


# ── helpers ────────────────────────────────────────────────────────────────

def vae_norm_params(vae, device, dtype):
    mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    std  = torch.tensor(vae.config.latents_std,  device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    return mean, std


def inputs2img(tensor):
    images = tensor.clamp(0, 1)
    return (images * 255).round().byte().detach().cpu().numpy()


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

    # ── Load Wan VAE ───────────────────────────────────────────────────
    vae = AutoencoderKLWan.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )

    if args.custom_chkpt is not None:
        print("Loading VAE checkpoint:", args.custom_chkpt)
        state = torch.load(args.custom_chkpt, map_location="cpu")
        new_state = OrderedDict(
            (k.replace("module.", ""), v) for k, v in state.items()
        )
        vae.load_state_dict(new_state)
        vae = vae.cuda()

    # Freeze everything; only train the decoder
    vae.requires_grad_(False)
    vae_trainable = []
    for name, param in vae.named_parameters():
        if "decoder" in name:
            param.requires_grad_(True)
            vae_trainable.append(param)

    print(
        f"VAE params total={len(list(vae.parameters()))}, "
        f"trainable decoder params={len(vae_trainable)}"
    )

    if args.gradient_checkpointing:
        vae.gradient_checkpointing_enable()

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
        itertools.chain(vae_trainable),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = DreamPoseDataset(
        instance_data_root=args.instance_data_dir,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        size=args.resolution,
        center_crop=args.center_crop,
    )

    def collate_fn(examples):
        frame_j = torch.stack([e["frame_j"] for e in examples])
        poses   = torch.stack([e["pose_j"]  for e in examples])
        return {
            "target_frame": frame_j.to(memory_format=torch.contiguous_format).float(),
            "poses":        poses.to(memory_format=torch.contiguous_format).float(),
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

    vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        vae, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)

    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("dreampose_vae_wan", config=vars(args))

    logger.info("***** Fine-tuning Wan VAE decoder *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs   = {args.num_train_epochs}")
    logger.info(f"  Total steps  = {args.max_train_steps}")

    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(args.epoch, args.num_train_epochs):
        vae.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(vae):
                # Dataset returns [0,1]; convert to [−1,1] for the Wan VAE
                target = (batch["target_frame"] * 2 - 1).to(
                    device=accelerator.device, dtype=weight_dtype
                )
                video = target.unsqueeze(2)                          # [B, 3, 1, H, W]

                # Encode then decode (autoencoder reconstruction loss on the decoder)
                raw    = vae.encode(video.to(dtype=torch.float32)).latent_dist.sample()
                recon  = vae.decode(raw).sample                      # [B, 3, 1, H, W]
                recon  = recon[:, :, 0]                              # [B, 3, H, W]

                loss = F.mse_loss(recon.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        itertools.chain(vae_trainable), args.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            writer.add_scalar("loss/train", loss.detach().item(), global_step)

            if global_step % 10 == 0:
                weights = accelerator.unwrap_model(vae).decoder.conv_out.weight.cpu().detach().numpy()
                weights = np.sum(weights, axis=0).flatten()
                plt.figure()
                plt.plot(range(len(weights)), weights)
                plt.title(f"VAE Decoder Weights mean={np.mean(weights):.4f}")
                writer.add_figure("decoder_weights", plt.gcf(), global_step=global_step)
                plt.close()

            if global_step == 1 or global_step % 50 == 0:
                with torch.no_grad():
                    recon_vis   = inputs2img((recon / 2 + 0.5).clamp(0, 1))
                    target_vis  = inputs2img((target / 2 + 0.5).clamp(0, 1))
                    viz = np.concatenate([recon_vis[0], target_vis[0]], axis=2)
                    writer.add_image("train/recon_vs_target", viz, global_step=global_step)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            if accelerator.is_main_process and global_step % args.save_steps == 0:
                vae_path = os.path.join(args.output_dir, f"vae_{epoch}.pth")
                torch.save(accelerator.unwrap_model(vae).state_dict(), vae_path)

        accelerator.wait_for_everyone()

    # Final save
    if accelerator.is_main_process:
        print("Saving final VAE to", args.output_dir)
        vae_path = os.path.join(args.output_dir, f"vae_{epoch}.pth")
        torch.save(accelerator.unwrap_model(vae).state_dict(), vae_path)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
