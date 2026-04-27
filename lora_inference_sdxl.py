"""
SDXL LoRA inference script.

Usage:
    python lora_inference_sdxl.py \
        --model /path/to/sdxl-base-1.0 \
        --lora ./output/mygo_sdxl/lora_weight.safetensors \
        --prompt "A photo of sks character in anime style" \
        --lora_scale 0.8 \
        --output result.png
"""

import argparse
import torch
from diffusers import StableDiffusionXLPipeline
from lora_diffusion import tune_lora_scale
from lora_diffusion.lora import (
    apply_learned_embed_in_clip_sdxl,
    monkeypatch_or_replace_safeloras,
    parse_safeloras,
    parse_safeloras_embeds,
)
from safetensors.torch import safe_open


def patch_pipe_sdxl(pipe, lora_path: str, lora_scale: float = 1.0):
    """
    Inject LoRA weights from a safetensors file into an SDXL pipeline.

    The safetensors file may contain keys for:
        "unet"            -> applied to pipe.unet
        "text_encoder"    -> applied to pipe.text_encoder  (CLIP-L)
        "text_encoder_2"  -> applied to pipe.text_encoder_2 (OpenCLIP-G)
    """
    safeloras = safe_open(lora_path, framework="pt", device="cpu")
    loras = parse_safeloras(safeloras)
    embeds = parse_safeloras_embeds(safeloras)

    for name, (lora_weights, ranks, target) in loras.items():
        if name == "unet":
            model = pipe.unet
        elif name == "text_encoder":
            model = pipe.text_encoder
        elif name == "text_encoder_2":
            model = pipe.text_encoder_2
        else:
            print(f"[WARN] Unknown model key '{name}' in safetensors - skipping.")
            continue

        from lora_diffusion.lora import monkeypatch_or_replace_lora_extended
        monkeypatch_or_replace_lora_extended(model, list(lora_weights), target, list(ranks))
        print(f"  Patched '{name}' with LoRA (target modules: {target})")

    runtime_tokens = {}
    if embeds:
        runtime_tokens = apply_learned_embed_in_clip_sdxl(
            embeds,
            pipe.text_encoder,
            pipe.tokenizer,
            pipe.text_encoder_2,
            pipe.tokenizer_2,
            idempotent=True,
        )
        if runtime_tokens:
            print("  Registered SDXL PTI tokens:")
            for original_token, runtime_token in runtime_tokens.items():
                print(f"    {original_token} -> {runtime_token}")

    # Apply scale
    tune_lora_scale(pipe.unet, lora_scale)
    if hasattr(pipe, "text_encoder"):
        tune_lora_scale(pipe.text_encoder, lora_scale)
    if hasattr(pipe, "text_encoder_2"):
        tune_lora_scale(pipe.text_encoder_2, lora_scale)

    return runtime_tokens


def main():
    parser = argparse.ArgumentParser(description="SDXL LoRA inference")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to SDXL-base model (local or HF hub).")
    parser.add_argument("--lora", type=str, required=True,
                        help="Path to LoRA .safetensors file.")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--lora_scale", type=float, default=0.8,
                        help="LoRA strength (0.0 = disabled, 1.0 = full).")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default="output_sdxl.png")
    parser.add_argument("--dtype", type=str, choices=["fp16", "bf16", "fp32"],
                        default="fp16")
    args = parser.parse_args()

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading SDXL pipeline from {args.model} ...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        use_safetensors=True,
    )
    pipe = pipe.to("cuda")

    print(f"Injecting LoRA from {args.lora} (scale={args.lora_scale}) ...")
    runtime_tokens = patch_pipe_sdxl(pipe, args.lora, lora_scale=args.lora_scale)
    if runtime_tokens:
        merged_tokens = sorted(set(runtime_tokens.values()))
        print("Prompt tokens available in this LoRA:", ", ".join(merged_tokens))

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(args.seed)

    print(f"Generating image for prompt: '{args.prompt}' ...")
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        generator=generator,
    ).images[0]

    image.save(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
