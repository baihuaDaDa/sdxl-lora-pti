from lora_diffusion import patch_pipe, tune_lora_scale
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("/dataset/stable-diffusion-v1-5")
pipe = pipe.to("cuda")

# 注入 LoRA 权重
patch_pipe(pipe, "./output/mygo/final_lora.safetensors", patch_ti=True)

# 控制 LoRA 强度（alpha，0~1）
tune_lora_scale(pipe.unet, 0.8)
tune_lora_scale(pipe.text_encoder, 0.8)

image = pipe("A photo of a guitar player with style of MYGO").images[0]

image.save("guitar-mygo-3000.png")