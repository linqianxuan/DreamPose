"""
DreamPose inference using Wan2.1 backend.

Usage:
    python test.py \
        --epoch 499 \
        --folder demo/custom-chkpts \
        --pose_folder demo/sample/poses \
        --key_frame_path demo/sample/key_frame.png \
        --s1 8 --s2 3 \
        --n_steps 50 \
        --output_dir demo/sample/results \
        --custom_vae demo/custom-chkpts/vae_1499.pth
"""

import argparse
import glob
import os
from collections import OrderedDict

import cv2
import numpy as np
import PIL
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
from transformers import CLIPProcessor, CLIPVisionModel

from models.wan_pose_transformer import get_wan_transformer, WanEmbeddingAdapter
from pipelines.wan_pose_pipeline import WanDreamPosePipeline

# ── argument parsing ───────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument(
    "--pretrained_model_name_or_path",
    default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    help="Base Wan2.1 model ID or local path.",
)
parser.add_argument(
    "--folder",
    default="demo/custom-chkpts",
    help="Path to subject-specific checkpoint folder.",
)
parser.add_argument(
    "--pose_folder",
    default="demo/sample/poses",
    help="Folder containing *_densepose.npy files for the target pose sequence.",
)
parser.add_argument("--epoch", type=int, required=True, help="Checkpoint epoch to load.")
parser.add_argument(
    "--key_frame_path",
    default="demo/sample/key_frame.png",
    help="Path to the reference (key) frame image.",
)
parser.add_argument("--s1", type=float, default=7.5, help="Image guidance scale.")
parser.add_argument("--s2", type=float, default=3.0, help="Pose guidance scale.")
parser.add_argument("--n_steps", type=int, default=50, help="Number of denoising steps.")
parser.add_argument("--output_dir", default=None, help="Where to save results.")
parser.add_argument("--j", type=int, default=-1, help="Specific frame index (-1 = all).")
parser.add_argument("--min_j", type=int, default=0, help="First frame index.")
parser.add_argument("--max_j", type=int, default=-1, help="Last frame index (-1 = all).")
parser.add_argument("--custom_vae", default=None, help="Path to a fine-tuned VAE checkpoint.")
parser.add_argument("--batch_size", type=int, default=1, help="Frames per inference call.")
args = parser.parse_args()

save_folder = args.output_dir if args.output_dir is not None else args.folder
os.makedirs(save_folder, exist_ok=True)

device = "cuda"
imSize = (512, 640)   # (width, height)

# ── load models ────────────────────────────────────────────────────────────

model_id = args.pretrained_model_name_or_path

# Wan VAE
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae")
vae = vae.to(device, dtype=torch.float32)

if args.custom_vae is not None:
    print("Loading custom VAE from:", args.custom_vae)
    state = torch.load(args.custom_vae, map_location="cpu")
    new_state = OrderedDict((k.replace("module.", ""), v) for k, v in state.items())
    vae.load_state_dict(new_state)

# Wan transformer
transformer = get_wan_transformer(model_id, n_poses=5)
transformer_path = os.path.join(args.folder, f"transformer_epoch_{args.epoch}.pth")
print("Loading transformer from:", transformer_path)
state = torch.load(transformer_path, map_location="cpu")
new_state = OrderedDict((k.replace("module.", ""), v) for k, v in state.items())
transformer.load_state_dict(new_state)
transformer = transformer.to(device, dtype=torch.bfloat16)

# Embedding adapter
cross_attn_dim = getattr(
    transformer.config, "cross_attn_dim",
    getattr(transformer.config, "cross_attention_dim", 4096),
)
adapter = WanEmbeddingAdapter(clip_dim=768, vae_channels=16, cross_attn_dim=cross_attn_dim)
adapter_path = os.path.join(args.folder, f"adapter_{args.epoch}.pth")
print("Loading adapter from:", adapter_path)
state = torch.load(adapter_path, map_location="cpu")
new_state = OrderedDict((k.replace("module.", ""), v) for k, v in state.items())
adapter.load_state_dict(new_state)
adapter = adapter.to(device, dtype=torch.bfloat16)

# Scheduler
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

# CLIP encoder
image_encoder  = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Build pipeline
pipe = WanDreamPosePipeline(
    vae=vae,
    transformer=transformer,
    adapter=adapter,
    scheduler=scheduler,
    image_encoder=image_encoder,
    clip_processor=clip_processor,
)

# ── pose normalisation transform (keeps DensePose in [−1, 1]) ─────────────
tensor_transforms = transforms.Compose([
    transforms.Normalize([0.5], [0.5]),
])

# ── load reference frame ───────────────────────────────────────────────────
input_image = PIL.Image.open(args.key_frame_path).resize(imSize)

# ── collect pose files ─────────────────────────────────────────────────────
frame_numbers = sorted(
    list(set(
        int(p.split("frame_")[-1].replace("_densepose.npy", ""))
        for p in glob.glob(f"{args.pose_folder}/frame_*_densepose.npy")
    ))
)
pose_paths = [f"{args.pose_folder}/frame_{n}_densepose.npy" for n in frame_numbers]

if args.j >= 0:
    pose_paths    = pose_paths[args.j: args.j + 1]
    frame_numbers = frame_numbers[args.j: args.j + 1]
elif args.max_j > -1:
    pose_paths    = pose_paths[args.min_j: args.max_j]
    frame_numbers = frame_numbers[args.min_j: args.max_j]
else:
    pose_paths    = pose_paths[args.min_j:]
    frame_numbers = frame_numbers[args.min_j:]

# ── inference loop ─────────────────────────────────────────────────────────
h, w = imSize[1], imSize[0]   # 640 height, 512 width

for i, (pose_path, frame_number) in enumerate(zip(pose_paths, frame_numbers)):
    # Construct 5 consecutive DensePose frames centred on the target frame
    poses = []
    for pose_number in range(frame_number - 2, frame_number + 3):
        dp_path = pose_path.replace(str(frame_number), str(pose_number))
        if not os.path.exists(dp_path):
            dp_path = pose_path
        dp = F.interpolate(
            torch.from_numpy(np.load(dp_path).astype("float32")).unsqueeze(0),
            (h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        poses.append(tensor_transforms(dp))

    input_pose = torch.cat(poses, 0).unsqueeze(0)   # [1, 10, H, W]

    result = pipe(
        image=input_image,
        pose=input_pose,
        num_inference_steps=args.n_steps,
        s1=args.s1,
        s2=args.s2,
        output_type="pil",
    )[0]

    # Save
    save_path = os.path.join(save_folder, f"pred_#{frame_number}.png")
    img_arr = np.array(result.convert("RGB"))
    img_arr = img_arr - img_arr.min()
    if img_arr.max() > 0:
        img_arr = (255 * img_arr / img_arr.max()).astype(np.uint8)
    cv2.imwrite(save_path, cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR))
    print(f"Saved {save_path}")
