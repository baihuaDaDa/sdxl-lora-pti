lora_add ./output_2000_4/mygo_sdxl_pti/lora_weight.safetensors \
  ./output_2000_4/jojo_sdxl_pti/lora_weight.safetensors \
  ./output_2000_4/mygo_jojo_merged.safetensors \
  --mode ljl-sdxl

lora_add ./output_2000_4/wushan_sdxl_pti/lora_weight.safetensors \
  ./output_2000_4/mygo_jojo_merged.safetensors \
  ./output_2000_4/mygo_jojo_wushan_merged.safetensors \
  --mode ljl-sdxl

lora_add ./output_2000_4/cyberpunk_all_sdxl_pti/lora_weight.safetensors \
  ./output_2000_4/mygo_jojo_wushan_merged.safetensors \
  ./output_2000_4/four_merged_all.safetensors \
  --mode ljl-sdxl