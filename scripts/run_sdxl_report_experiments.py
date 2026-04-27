import argparse
import csv
import json
from pathlib import Path
import sys

import torch
from diffusers import StableDiffusionXLPipeline
from lora_diffusion import tune_lora_scale

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lora_inference_sdxl import patch_pipe_sdxl


def parse_args():
    parser = argparse.ArgumentParser(description="Run reproducible SDXL LoRA report experiments.")
    parser.add_argument("--spec", type=str, required=True, help="Path to the experiment JSON spec.")
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Directory used to store generated images and metadata.",
    )
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        help="Only run selected group ids. May be passed multiple times.",
    )
    return parser.parse_args()


def load_spec(spec_path: Path):
    with spec_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(value: str | None, base_dir: Path):
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def build_dtype(dtype_name: str):
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    return dtype_map[dtype_name]


def set_lora_scale(pipe, scale: float):
    tune_lora_scale(pipe.unet, scale)
    if hasattr(pipe, "text_encoder"):
        tune_lora_scale(pipe.text_encoder, scale)
    if hasattr(pipe, "text_encoder_2"):
        tune_lora_scale(pipe.text_encoder_2, scale)


def sanitize_value(value):
    text = str(value)
    text = text.replace(" ", "_")
    text = text.replace("/", "-")
    text = text.replace("<", "")
    text = text.replace(">", "")
    return text


def group_selected(group_id: str, selected_groups: set[str]):
    return not selected_groups or group_id in selected_groups


def generate_records(pipe, group, shared, output_root: Path):
    group_dir = output_root / group["id"]
    group_dir.mkdir(parents=True, exist_ok=True)

    group_steps = group.get("steps", shared.get("steps", 40))
    group_guidance = group.get("guidance_scale", shared.get("guidance_scale", 7.5))
    group_width = group.get("width", shared.get("width", 1024))
    group_height = group.get("height", shared.get("height", 1024))
    group_negative = group.get("negative_prompt", shared.get("negative_prompt", ""))
    group_seeds = group.get("seeds", shared.get("seeds", [13, 37, 73]))
    group_scales = group.get("scales", shared.get("scales", [0.8]))

    records = []
    for case in group["cases"]:
        case_id = case["id"]
        case_dir = group_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        case_prompt = case["prompt"]
        case_negative = case.get("negative_prompt", group_negative)
        case_steps = case.get("steps", group_steps)
        case_guidance = case.get("guidance_scale", group_guidance)
        case_width = case.get("width", group_width)
        case_height = case.get("height", group_height)
        case_seeds = case.get("seeds", group_seeds)
        case_scales = case.get("scales", group_scales)

        for scale in case_scales:
            set_lora_scale(pipe, scale)
            scale_dir = case_dir / f"scale_{sanitize_value(scale)}"
            scale_dir.mkdir(parents=True, exist_ok=True)
            for seed in case_seeds:
                output_path = scale_dir / f"seed_{sanitize_value(seed)}.png"
                print(
                    f"[RUN] group={group['id']} case={case_id} scale={scale} seed={seed} -> {output_path}"
                )
                generator = torch.Generator(device="cuda").manual_seed(int(seed))
                with torch.inference_mode():
                    image = pipe(
                        prompt=case_prompt,
                        negative_prompt=case_negative,
                        num_inference_steps=case_steps,
                        guidance_scale=case_guidance,
                        width=case_width,
                        height=case_height,
                        generator=generator,
                    ).images[0]
                image.save(output_path)
                records.append(
                    {
                        "group_id": group["id"],
                        "group_title": group.get("title", group["id"]),
                        "lora_path": group.get("lora_path", ""),
                        "case_id": case_id,
                        "case_title": case.get("title", case_id),
                        "prompt": case_prompt,
                        "negative_prompt": case_negative,
                        "steps": case_steps,
                        "guidance_scale": case_guidance,
                        "width": case_width,
                        "height": case_height,
                        "lora_scale": scale,
                        "seed": seed,
                        "output_path": str(output_path.relative_to(output_root)),
                    }
                )

    return records


def main():
    args = parse_args()
    spec_path = Path(args.spec).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    spec = load_spec(spec_path)
    shared = spec.get("shared", {})
    shared_dtype = build_dtype(shared.get("dtype", "fp16"))
    selected_groups = set(args.group)
    spec_dir = spec_path.parent
    all_records = []

    for group in spec["groups"]:
        group_id = group["id"]
        if not group_selected(group_id, selected_groups):
            continue

        model_path = resolve_path(group.get("model", shared["model"]), spec_dir)
        lora_path = resolve_path(group.get("lora"), spec_dir)
        group["lora_path"] = str(lora_path) if lora_path else ""
        print(f"[LOAD] group={group_id} model={model_path}")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            str(model_path),
            torch_dtype=shared_dtype,
            use_safetensors=True,
        )
        pipe = pipe.to("cuda")

        if lora_path is not None:
            print(f"[PATCH] group={group_id} lora={lora_path}")
            runtime_tokens = patch_pipe_sdxl(pipe, str(lora_path), lora_scale=1.0)
            if runtime_tokens:
                print(f"[TOKENS] group={group_id} {sorted(set(runtime_tokens.values()))}")

        all_records.extend(generate_records(pipe, group, shared, output_root))
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metadata_path = output_root / "metadata.csv"
    if all_records:
        with metadata_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_records[0].keys()))
            writer.writeheader()
            writer.writerows(all_records)
        print(f"[DONE] wrote metadata to {metadata_path}")
    else:
        print("[DONE] no groups were selected, nothing generated")


if __name__ == "__main__":
    main()