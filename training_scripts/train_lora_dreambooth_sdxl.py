# Adapted from train_lora_dreambooth.py for SDXL-base
# Key differences from SD-v1.5:
#   - Dual text encoders (CLIP-L + OpenCLIP-G)
#   - SDXL UNet requires added_cond_kwargs (time IDs + pooled text embeds)
#   - VAE scaling factor: 0.13025 (not 0.18215)
#   - Default resolution: 1024
#   - Pipeline: StableDiffusionXLPipeline

import argparse
import hashlib
import inspect
import itertools
import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from lora_diffusion import (
    extract_lora_ups_down,
    inject_trainable_lora,
    safetensors_available,
    save_lora_weight,
    save_safeloras,
)
from lora_diffusion.xformers_utils import set_use_memory_efficient_attention_xformers
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import random
import re

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DreamBoothDatasetSDXL(Dataset):
    """
    Dataset for SDXL DreamBooth training.
    Encodes prompts with both tokenizers so the collate_fn can batch them easily.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer_1,
        tokenizer_2,
        class_data_root=None,
        class_prompt=None,
        size=1024,
        center_crop=False,
        color_jitter=False,
        h_flip=False,
        resize=True,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.resize = resize

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exist.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        img_transforms = []
        if resize:
            img_transforms.append(
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
            )
        if center_crop:
            img_transforms.append(transforms.CenterCrop(size))
        if color_jitter:
            img_transforms.append(transforms.ColorJitter(0.2, 0.1))
        if h_flip:
            img_transforms.append(transforms.RandomHorizontalFlip())

        self.image_transforms = transforms.Compose(
            [*img_transforms, transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )

    def _tokenize(self, prompt):
        ids_1 = self.tokenizer_1(
            prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer_1.model_max_length,
        ).input_ids
        ids_2 = self.tokenizer_2(
            prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer_2.model_max_length,
        ).input_ids
        return ids_1, ids_2

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(
            self.instance_images_path[index % self.num_instance_images]
        )
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        example["instance_images"] = self.image_transforms(instance_image)
        (
            example["instance_prompt_ids_1"],
            example["instance_prompt_ids_2"],
        ) = self._tokenize(self.instance_prompt)

        if self.class_data_root:
            class_image = Image.open(
                self.class_images_path[index % self.num_class_images]
            )
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            (
                example["class_prompt_ids_1"],
                example["class_prompt_ids_2"],
            ) = self._tokenize(self.class_prompt)

        return example


class PromptDataset(Dataset):
    """Simple dataset for generating class images."""

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return {"prompt": self.prompt, "index": index}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_time_ids(
    original_size,
    crops_coords_top_left,
    target_size,
    device,
    dtype,
):
    """Compute the additional conditioning used by SDXL (size/crop embeddings)."""
    add_time_ids = list(original_size) + list(crops_coords_top_left) + list(target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
    return add_time_ids


def encode_prompt_sdxl(
    text_encoders,
    tokenizers,
    prompt_batch,
    device,
    weight_dtype,
):
    """
    Encode a batch of already-padded token id tensors with both SDXL text encoders.

    Returns:
        prompt_embeds      : (B, seq_len, 2048)  concatenation of both encoder outputs
        pooled_prompt_embeds: (B, 1280)           pooled output of the second encoder
    """
    prompt_embeds_list = []

    for tokenizer, text_encoder, token_ids in zip(tokenizers, text_encoders, prompt_batch):
        prompt_embeds_out = text_encoder(
            token_ids.to(device),
            output_hidden_states=True,
        )
        # Use the second-to-last hidden state (standard SDXL conditioning)
        hidden_states = prompt_embeds_out.hidden_states[-2]
        prompt_embeds_list.append(hidden_states)

    # Pooled output comes from the *second* text encoder only
    pooled_prompt_embeds = text_encoder(
        prompt_batch[-1].to(device),
        output_hidden_states=True,
    ).text_embeds  # shape (B, 1280)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)   # (B, seq, 2048)
    return prompt_embeds.to(dtype=weight_dtype), pooled_prompt_embeds.to(dtype=weight_dtype)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="SDXL LoRA DreamBooth training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained SDXL model or HF model identifier.",
    )
    parser.add_argument(
        "--pretrained_vae_name_or_path",
        type=str,
        default=None,
        help="Optional: path to a different VAE.",
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        required=True,
        help="Folder with training instance images.",
    )
    parser.add_argument("--class_data_dir", type=str, default=None)
    parser.add_argument(
        "--instance_prompt",
        type=str,
        required=True,
        help="Prompt describing the instance (e.g. 'a photo of sks dog').",
    )
    parser.add_argument("--class_prompt", type=str, default=None)
    parser.add_argument("--with_prior_preservation", default=False, action="store_true")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0)
    parser.add_argument("--num_class_images", type=int, default=100)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora-sdxl-output",
        help="Output directory for checkpoints.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["pt", "safe", "both"],
        default="safe",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Training resolution (SDXL default: 1024).",
    )
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--color_jitter", action="store_true")
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train text encoder 1 with LoRA.",
    )
    parser.add_argument(
        "--train_text_encoder_2",
        action="store_true",
        help="Whether to train text encoder 2 with LoRA.",
    )
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--sample_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--learning_rate_text", type=float, default=5e-6)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--resume_unet", type=str, default=None)
    parser.add_argument("--resume_text_encoder", type=str, default=None)
    parser.add_argument("--resize", type=bool, default=True)
    parser.add_argument("--use_xformers", action="store_true")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
    )

    args = parser.parse_args(input_args) if input_args is not None else parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("--class_data_dir is required when using prior preservation.")
        if args.class_prompt is None:
            raise ValueError("--class_prompt is required when using prior preservation.")

    if not safetensors_available:
        if args.output_format in ("both", "safe"):
            print("safetensors not available - falling back to pt output format")
            args.output_format = "pt"

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=logging_dir,
    )

    if args.seed is not None:
        set_seed(args.seed)

    # ------------------------------------------------------------------
    # Prior preservation: generate class images if needed
    # ------------------------------------------------------------------
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        class_images_dir.mkdir(parents=True, exist_ok=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                revision=args.revision,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(
                sample_dataset, batch_size=args.sample_batch_size
            )
            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(sample_dataloader, desc="Generating class images",
                                disable=not accelerator.is_local_main_process):
                images = pipeline(example["prompt"]).images
                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = (
                        class_images_dir
                        / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    )
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load tokenizers and models
    # ------------------------------------------------------------------
    tokenizer_1 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision
    )

    text_encoder_1 = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,
        subfolder=None if args.pretrained_vae_name_or_path else "vae",
        revision=None if args.pretrained_vae_name_or_path else args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    # ------------------------------------------------------------------
    # Inject LoRA into UNet
    # ------------------------------------------------------------------
    unet.requires_grad_(False)
    unet_lora_params, _ = inject_trainable_lora(
        unet, r=args.lora_rank, loras=args.resume_unet
    )

    for _up, _down in extract_lora_ups_down(unet):
        print("Before training: UNet first LoRA up:", _up.weight.data)
        print("Before training: UNet first LoRA down:", _down.weight.data)
        break

    vae.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    te1_lora_params = None
    te2_lora_params = None

    if args.train_text_encoder:
        te1_lora_params, _ = inject_trainable_lora(
            text_encoder_1,
            target_replace_module=["CLIPAttention"],
            r=args.lora_rank,
        )
        for _up, _down in extract_lora_ups_down(
            text_encoder_1, target_replace_module=["CLIPAttention"]
        ):
            print("Before training: TE1 first LoRA up:", _up.weight.data)
            print("Before training: TE1 first LoRA down:", _down.weight.data)
            break

    if args.train_text_encoder_2:
        te2_lora_params, _ = inject_trainable_lora(
            text_encoder_2,
            target_replace_module=["CLIPAttention"],
            r=args.lora_rank,
        )
        for _up, _down in extract_lora_ups_down(
            text_encoder_2, target_replace_module=["CLIPAttention"]
        ):
            print("Before training: TE2 first LoRA up:", _up.weight.data)
            print("Before training: TE2 first LoRA down:", _down.weight.data)
            break

    if args.use_xformers:
        set_use_memory_efficient_attention_xformers(unet, True)
        set_use_memory_efficient_attention_xformers(vae, True)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_1.gradient_checkpointing_enable()
        if args.train_text_encoder_2:
            text_encoder_2.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Install bitsandbytes: `pip install bitsandbytes`.")
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    text_lr = args.learning_rate if args.learning_rate_text is None else args.learning_rate_text

    params_to_optimize = [
        {"params": itertools.chain(*unet_lora_params), "lr": args.learning_rate},
    ]
    if args.train_text_encoder and te1_lora_params is not None:
        params_to_optimize.append(
            {"params": itertools.chain(*te1_lora_params), "lr": text_lr}
        )
    if args.train_text_encoder_2 and te2_lora_params is not None:
        params_to_optimize.append(
            {"params": itertools.chain(*te2_lora_params), "lr": text_lr}
        )

    # Flatten to a single param group list when only unet is trained, for compatibility
    if len(params_to_optimize) == 1:
        params_to_optimize = itertools.chain(*unet_lora_params)

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_config(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # ------------------------------------------------------------------
    # Dataset & DataLoader
    # ------------------------------------------------------------------
    train_dataset = DreamBoothDatasetSDXL(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        size=args.resolution,
        center_crop=args.center_crop,
        color_jitter=args.color_jitter,
        resize=args.resize,
    )

    def collate_fn(examples):
        pixel_values = [ex["instance_images"] for ex in examples]
        input_ids_1 = [ex["instance_prompt_ids_1"] for ex in examples]
        input_ids_2 = [ex["instance_prompt_ids_2"] for ex in examples]

        if args.with_prior_preservation:
            pixel_values += [ex["class_images"] for ex in examples]
            input_ids_1 += [ex["class_prompt_ids_1"] for ex in examples]
            input_ids_2 += [ex["class_prompt_ids_2"] for ex in examples]

        pixel_values = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()

        input_ids_1 = tokenizer_1.pad(
            {"input_ids": input_ids_1},
            padding="max_length",
            max_length=tokenizer_1.model_max_length,
            return_tensors="pt",
        ).input_ids
        input_ids_2 = tokenizer_2.pad(
            {"input_ids": input_ids_2},
            padding="max_length",
            max_length=tokenizer_2.model_max_length,
            return_tensors="pt",
        ).input_ids

        return {
            "pixel_values": pixel_values,
            "input_ids_1": input_ids_1,
            "input_ids_2": input_ids_2,
        }

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=1,
    )

    # ------------------------------------------------------------------
    # LR scheduler
    # ------------------------------------------------------------------
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # ------------------------------------------------------------------
    # Accelerator prepare
    # ------------------------------------------------------------------
    models_to_prepare = [unet, optimizer, train_dataloader, lr_scheduler]
    if args.train_text_encoder:
        models_to_prepare = [unet, text_encoder_1, optimizer, train_dataloader, lr_scheduler]
        (unet, text_encoder_1, optimizer, train_dataloader, lr_scheduler) = accelerator.prepare(
            *models_to_prepare
        )
    elif args.train_text_encoder_2:
        models_to_prepare = [unet, text_encoder_2, optimizer, train_dataloader, lr_scheduler]
        (unet, text_encoder_2, optimizer, train_dataloader, lr_scheduler) = accelerator.prepare(
            *models_to_prepare
        )
    elif args.train_text_encoder and args.train_text_encoder_2:
        models_to_prepare = [unet, text_encoder_1, text_encoder_2, optimizer, train_dataloader, lr_scheduler]
        (unet, text_encoder_1, text_encoder_2, optimizer, train_dataloader, lr_scheduler) = accelerator.prepare(
            *models_to_prepare
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move frozen models to device with proper dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder_1.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder_2:
        text_encoder_2.to(accelerator.device, dtype=weight_dtype)

    # Recalculate steps
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth_sdxl", config=vars(args))

    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    print("***** Running SDXL LoRA training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num epochs = {args.num_train_epochs}")
    print(f"  Batch size (per device) = {args.train_batch_size}")
    print(f"  Total batch size = {total_batch_size}")
    print(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")
    global_step = 0
    last_save = 0

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder_1.train()
        if args.train_text_encoder_2:
            text_encoder_2.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):

                # 1. Encode images to latents
                # SDXL VAE scaling factor: 0.13025
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * 0.13025

                # 2. Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 3. Encode text with both encoders
                # text_encoder_1 returns hidden_states[-2], text_encoder_2 returns hidden_states[-2] + pooled
                te1_out = text_encoder_1(
                    batch["input_ids_1"].to(accelerator.device),
                    output_hidden_states=True,
                )
                te2_out = text_encoder_2(
                    batch["input_ids_2"].to(accelerator.device),
                    output_hidden_states=True,
                )

                # Concat hidden states along feature dim → (B, seq, 768+1280=2048)
                encoder_hidden_states = torch.cat(
                    [te1_out.hidden_states[-2], te2_out.hidden_states[-2]], dim=-1
                )
                pooled_prompt_embeds = te2_out.text_embeds  # (B, 1280)

                # 4. Build SDXL extra conditioning (time IDs)
                # original_size and target_size are both (resolution, resolution)
                # crops_coords_top_left defaults to (0, 0)
                resolution = args.resolution
                add_time_ids = torch.tensor(
                    [[resolution, resolution, 0, 0, resolution, resolution]] * bsz,
                    dtype=weight_dtype,
                    device=accelerator.device,
                )

                added_cond_kwargs = {
                    "text_embeds": pooled_prompt_embeds.to(dtype=weight_dtype),
                    "time_ids": add_time_ids,
                }

                # 5. Predict noise
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states.to(dtype=weight_dtype),
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

                # 6. Get target
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                # 7. Compute loss
                if args.with_prior_preservation:
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)
                    loss = (
                        F.mse_loss(model_pred.float(), target.float(), reduction="none")
                        .mean([1, 2, 3])
                        .mean()
                    )
                    prior_loss = F.mse_loss(
                        model_pred_prior.float(), target_prior.float(), reduction="mean"
                    )
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = unet.parameters()
                    if args.train_text_encoder:
                        params_to_clip = itertools.chain(
                            params_to_clip, text_encoder_1.parameters()
                        )
                    if args.train_text_encoder_2:
                        params_to_clip = itertools.chain(
                            params_to_clip, text_encoder_2.parameters()
                        )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if args.save_steps and global_step - last_save >= args.save_steps:
                    if accelerator.is_main_process:
                        accepts_keep_fp32_wrapper = "keep_fp32_wrapper" in set(
                            inspect.signature(accelerator.unwrap_model).parameters.keys()
                        )
                        extra_args = (
                            {"keep_fp32_wrapper": True}
                            if accepts_keep_fp32_wrapper
                            else {}
                        )
                        unet_unwrapped = accelerator.unwrap_model(unet, **extra_args)

                        checkpoint_path = f"{args.output_dir}/lora_weight_s{global_step}.safetensors"
                        loras = {
                            "unet": (unet_unwrapped, {"CrossAttention", "Attention", "GEGLU"}),
                        }
                        if args.train_text_encoder:
                            te1_unwrapped = accelerator.unwrap_model(text_encoder_1, **extra_args)
                            loras["text_encoder"] = (te1_unwrapped, {"CLIPAttention"})
                        if args.train_text_encoder_2:
                            te2_unwrapped = accelerator.unwrap_model(text_encoder_2, **extra_args)
                            loras["text_encoder_2"] = (te2_unwrapped, {"CLIPAttention"})

                        print(f"Saving checkpoint: {checkpoint_path}")
                        if args.output_format in ("safe", "both"):
                            save_safeloras(loras, checkpoint_path)
                        if args.output_format in ("pt", "both"):
                            save_lora_weight(
                                unet_unwrapped,
                                f"{args.output_dir}/lora_weight_s{global_step}.pt",
                            )

                        last_save = global_step

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()

    # ------------------------------------------------------------------
    # Save final weights
    # ------------------------------------------------------------------
    if accelerator.is_main_process:
        unet_unwrapped = accelerator.unwrap_model(unet)
        loras = {
            "unet": (unet_unwrapped, {"CrossAttention", "Attention", "GEGLU"}),
        }
        if args.train_text_encoder:
            loras["text_encoder"] = (
                accelerator.unwrap_model(text_encoder_1),
                {"CLIPAttention"},
            )
        if args.train_text_encoder_2:
            loras["text_encoder_2"] = (
                accelerator.unwrap_model(text_encoder_2),
                {"CLIPAttention"},
            )

        print("\n\nSDXL LoRA TRAINING DONE!\n\n")

        if args.output_format in ("pt", "both"):
            save_lora_weight(unet_unwrapped, args.output_dir + "/lora_weight.pt")
            if args.train_text_encoder:
                save_lora_weight(
                    accelerator.unwrap_model(text_encoder_1),
                    args.output_dir + "/lora_weight.text_encoder.pt",
                    target_replace_module=["CLIPAttention"],
                )
            if args.train_text_encoder_2:
                save_lora_weight(
                    accelerator.unwrap_model(text_encoder_2),
                    args.output_dir + "/lora_weight.text_encoder_2.pt",
                    target_replace_module=["CLIPAttention"],
                )

        if args.output_format in ("safe", "both"):
            save_safeloras(loras, args.output_dir + "/lora_weight.safetensors")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
