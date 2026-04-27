#!/bin/bash
# SDXL LoRA + Textual Inversion (PTI) training script
# Phase 1 (0~unfreeze_lora_step): train TI embeddings only
# Phase 2 (unfreeze_lora_step~end): train UNet LoRA only

export MODEL_NAME="/dataset/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="./data/mygo"
export OUTPUT_DIR="./output/mygo_sdxl_pti"

mkdir -p "$OUTPUT_DIR"

accelerate launch training_scripts/train_lora_w_ti_sdxl.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --instance_data_dir="$INSTANCE_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --placeholder_token="<mygo>" \
  --initializer_token="anime" \
  --learnable_property="style" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --lora_rank=16 \
  --learning_rate=1e-4 \
  --learning_rate_text=5e-6 \
  --learning_rate_ti=5e-4 \
  --unfreeze_lora_step=500 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --save_steps=200 \
  --output_format="safe" \
  --mixed_precision="bf16" \
  --resize=True
# Optional flags:
#   --train_text_encoder          also apply LoRA to text_encoder_1 in LoRA phase
#   --train_text_encoder_2        also apply LoRA to text_encoder_2 in LoRA phase
#   --just_ti                     only train TI embeddings, skip LoRA entirely
#   --stochastic_attribute="detailed,colorful"   random attrs added to prompt
#   --use_8bit_adam               lower memory (needs bitsandbytes)
#   --use_xformers                memory-efficient attention
#   --with_prior_preservation
#   --class_data_dir=./data/class_images
#   --class_prompt="anime style character"
# Recommended single-GPU preset for small datasets (~25 images):
#   launch with: accelerate launch --num_processes=1 training_scripts/train_lora_w_ti_sdxl.py
#   keep --train_text_encoder and --train_text_encoder_2 disabled at first
#   suggested overrides:
#     --train_batch_size=1
#     --gradient_accumulation_steps=1
#     --lora_rank=8
#     --learning_rate=5e-5
#     --learning_rate_text=5e-6
#     --learning_rate_ti=2e-4
#     --unfreeze_lora_step=500
#     --max_train_steps=2000
#     --save_steps=200
