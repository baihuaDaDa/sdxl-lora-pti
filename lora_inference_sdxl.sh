python lora_inference_sdxl.py \
    --model /dataset/stable-diffusion-xl-base-1.0 \
    --lora ./output/mygo_sdxl_pti/lora_weight.safetensors \
    --prompt "A <mygo> style photo of a girl, short grey hair, blue eyes, hanasakigawa girls' academy uniform, holding microphone, holding mic with both hands, singing earnestly, slightly hunched posture, expressive eyes, stage spotlight, upper body" \
    --lora_scale 0.8 \
    --output lora-sdxl/mygo_pti/takamatsu-2000-4.png

python lora_inference_sdxl.py \
    --model /dataset/stable-diffusion-xl-base-1.0 \
    --lora ./output/jojo_sdxl_pti/lora_weight.safetensors \
    --prompt "A <jojo> style photo of a man, muscular, tall, black gakuran, long coat, peaked cap, hat seamlessly merging with hair, gold chain on collar, pointing finger forward, serious expression, iconic jojo pose, dramatic lighting, dynamic angle" \
    --lora_scale 0.8 \
    --output lora-sdxl/jojo_pti/kujo-2000-4.png

python lora_inference_sdxl.py \
    --model /dataset/stable-diffusion-xl-base-1.0 \
    --lora ./output/wushan_sdxl_pti/lora_weight.safetensors \
    --prompt "A <wushan> style photo of a boy, messy black hair, ancient chinese clothes, martial arts attire, muscular, holding spear, fiery spear, blazing flames, dynamic combat stance, intense gaze, ink wash painting influence, action pose" \
    --lora_scale 0.8 \
    --output lora-sdxl/wushan_pti/wenren-2000-4.png

python lora_inference_sdxl.py \
    --model /dataset/stable-diffusion-xl-base-1.0 \
    --lora ./output/cyberpunk_portrait_sdxl_pti/lora_weight.safetensors \
    --prompt "A <cyberpunk1> style photo of a girl, bob cut, white hair, multicolored inner hair, pastel gradient hair, red eye makeup, white netrunner suit, holding glowing red monowire, dynamic swinging action, cyberpunk city background, neon lights, looking at viewer" \
    --lora_scale 0.8 \
    --output lora-sdxl/cyberpunk_portrait_pti/lucyna-2000-4.png

python lora_inference_sdxl.py \
    --model /dataset/stable-diffusion-xl-base-1.0 \
    --lora ./output/cyberpunk_all_sdxl_pti/lora_weight.safetensors \
    --prompt "A <cyberpunk2> style photo of a girl, bob cut, white hair, multicolored inner hair, pastel gradient hair, red eye makeup, white netrunner suit, holding glowing red monowire, dynamic swinging action, cyberpunk city background, neon lights, looking at viewer" \
    --lora_scale 0.8 \
    --output lora-sdxl/cyberpunk_all_pti/lucyna-2000-4.png