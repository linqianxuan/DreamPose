# DreamPose
Official implementation of "DreamPose: Fashion Image-to-Video Synthesis via Stable Diffusion" by Johanna Karras, Aleksander Holynski, Ting-Chun Wang, and Ira Kemelmacher-Shlizerman.

> **Note:** This fork replaces the original Stable Diffusion v1.4 backbone with **[Wan2.1](https://github.com/Wan-Video/Wan2.1)** — a state-of-the-art open-source video diffusion transformer. Key changes: SD UNet → `WanTransformer3DModel`, SD VAE → `AutoencoderKLWan` (3-D causal), DDPM → Flow Matching (`FlowMatchEulerDiscreteScheduler`), CLIP text encoder → CLIP vision + `WanEmbeddingAdapter`.

 * [Project Page](https://grail.cs.washington.edu/projects/dreampose)
 * [Paper](https://arxiv.org/abs/2304.06025)

![Teaser Image](media/Teaser.png "Teaser")

## Demo

You can generate a video using DreamPose using our pretrained models.

1. [Download](https://drive.google.com/drive/folders/15SaT3kZFRIjxuHT6UrGr6j0183clTK_D?usp=share_link) and unzip the pretrained models inside demo/custom-chkpts.zip
2. [Download](https://drive.google.com/drive/folders/1CjzcOp_ZUt-dyrzNAFE0T8bS3cbKTsVG?usp=share_link) and unzip the input poses inside demo/sample/poses.zip
3. Run inference using the command below:
    ```
    python test.py --epoch 499 --folder demo/custom-chkpts --pose_folder demo/sample/poses  --key_frame_path demo/sample/key_frame.png --s1 8 --s2 3 --n_steps 50 --output_dir demo/sample/results --custom_vae demo/custom-chkpts/vae_1499.pth
    ```

## Data Preparation

To prepare a sample for finetuning, create a directory containing train and test subdirectories containing the train frames (desired subject) and test frames (desired pose sequence), respectively. Note that the test frames are not expected to be of the same subject. See demo/sample for an example.

Then, run [DensePose](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose) using the "densepose_rcnn_R_50_FPN_s1x" checkpoint on all images in the sample directory. Finally, reformat the pickled DensePose output using utils/densepose.py. You need to change the "outpath" filepath to point to the pickled DensePose output.

## Download or Finetune Base Model

DreamPose is now built on Wan2.1. You can use any Wan2.1 T2V model as the starting point. We recommend `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` for research fine-tuning. Models are downloaded automatically from HuggingFace Hub.

```
accelerate launch --num_processes=4 train.py \
    --pretrained_model_name_or_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --instance_data_dir=../path/to/dataset \
    --output_dir=checkpoints \
    --resolution=512 \
    --train_batch_size=2 \
    --gradient_accumulation_steps=4 \
    --learning_rate=5e-6 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_train_epochs=300 \
    --run_name dreampose \
    --dropout_rate=0.15
```

## Finetune on Sample

In this next step, we finetune DreamPose on one or more input frames to create a subject-specific model.

1. Finetune the Wan transformer

    ```
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
    ```

2. Finetune the Wan VAE decoder

    ```
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
    ```

## Testing

Once you have finetuned your custom, subject-specific DreamPose model, you can generate frames using the following command:

```
python test.py \
    --epoch 499 \
    --folder demo/custom-chkpts \
    --pose_folder demo/sample/poses \
    --key_frame_path demo/sample/key_frame.png \
    --s1 8 --s2 3 \
    --n_steps 50 \
    --output_dir results \
    --custom_vae demo/custom-chkpts/vae_1499.pth
```

## Architecture Changes (SD v1.4 → Wan2.1)

| Component | Original (SD v1.4) | This fork (Wan2.1) |
|-----------|-------------------|-------------------|
| Backbone | `UNet2DConditionModel` | `WanTransformer3DModel` (DiT) |
| VAE | `AutoencoderKL` (4-ch, 2-D) | `AutoencoderKLWan` (16-ch, 3-D causal) |
| Latent space | `[B, 4, H/8, W/8]` | `[B, 16, T, H/8, W/8]` |
| Scheduler | `DDPMScheduler` | `FlowMatchEulerDiscreteScheduler` |
| Text encoder | CLIP text (unused) | — |
| Image encoder | `CLIPVisionModel` | `CLIPVisionModel` (unchanged) |
| Conditioning adapter | `Embedding_Adapter` | `WanEmbeddingAdapter` |
| Pose injection | extra conv_in channels | extra patch_embedding channels |
| Training target | noise (ε-prediction) | velocity (flow matching) |

### Acknowledgment

This code is based on the original [DreamPose](https://grail.cs.washington.edu/projects/dreampose) and the [Wan2.1](https://github.com/Wan-Video/Wan2.1) model. The pipeline structure is adapted from the [Hugging Face diffusers repo](https://github.com/huggingface/diffusers).
