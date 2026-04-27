#!/bin/bash
# SDXL LoRA + Textual Inversion (PTI) training script
# Phase 1 (0~unfreeze_lora_step): train TI embeddings only
# Phase 2 (unfreeze_lora_step~end): train UNet LoRA only

export MODEL_NAME="/dataset/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="./data/cyberpunk_all"
export OUTPUT_DIR="./output/cyberpunk_all_sdxl_pti"

mkdir -p "$OUTPUT_DIR"

accelerate launch --num_processes=1 \
  training_scripts/train_lora_w_ti_sdxl.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --instance_data_dir="$INSTANCE_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --placeholder_token="<cyberpunk2>" \
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
