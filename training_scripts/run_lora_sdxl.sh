#!/bin/bash
# SDXL LoRA DreamBooth training script
# Requires: accelerate, diffusers >= 0.21, transformers, bitsandbytes (optional)

export MODEL_NAME="/dataset/stable-diffusion-xl-base-1.0"   # e.g. stabilityai/stable-diffusion-xl-base-1.0
export INSTANCE_DIR="./data/mygo"
export OUTPUT_DIR="./output/mygo_sdxl"

mkdir -p "$OUTPUT_DIR"

accelerate launch training_scripts/train_lora_dreambooth_sdxl.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --instance_data_dir="$INSTANCE_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --instance_prompt="a photo of sks character in MYGO anime style" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --lora_rank=16 \
  --learning_rate=1e-4 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=100 \
  --max_train_steps=2000 \
  --save_steps=200 \
  --output_format="safe" \
  --mixed_precision="bf16" \
  --resize=True
# Optional flags:
#   --train_text_encoder          train CLIP-L text encoder with LoRA
#   --train_text_encoder_2        train OpenCLIP-G text encoder with LoRA
#   --use_8bit_adam               use 8-bit Adam (requires bitsandbytes)
#   --use_xformers                use memory-efficient attention
#   --with_prior_preservation     enable prior preservation loss
#   --class_data_dir=./data/class_images
#   --class_prompt="a photo of a character"
#   --num_class_images=200
