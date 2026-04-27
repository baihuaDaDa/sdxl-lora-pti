from diffusers import StableDiffusionXLPipeline
import torch

model_id = "/dataset/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "A anime style photo of a man named kujo jotaro, muscular, tall, black gakuran, long coat, peaked cap, hat seamlessly merging with hair, gold chain on collar, pointing finger forward, serious expression, iconic jojo pose, dramatic lighting, dynamic angle"
image = pipe(prompt).images[0]  
    
image.save("origin-sdxl/jojo/kujo.png")
