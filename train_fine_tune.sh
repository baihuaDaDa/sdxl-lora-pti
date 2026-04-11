export MODEL_NAME="/dataset/stable-diffusion-v1-5"
export INSTANCE_DIR="./data/mygo"
export OUTPUT_DIR="./output/mygo"

lora_pti \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --scale_lr \
  --learning_rate_unet=1e-4 \
  --learning_rate_text=1e-5 \
  --learning_rate_ti=5e-4 \
  --lr_scheduler="linear" \
  --placeholder_tokens="MYGO" \
  --use_template="style" \
  --max_train_steps_ti=1000 \
  --max_train_steps_tuning=3000 \
  --perform_inversion=True \
  --lora_rank=1 \
  --device="cuda:0"