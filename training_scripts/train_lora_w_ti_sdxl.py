# SDXL LoRA + Textual Inversion (PTI) training script.
# Adapted from train_lora_w_ti.py for SDXL-base.
#
# Key differences from SD-v1.5:
#   - Dual tokenizers (tokenizer_1 + tokenizer_2)
#   - Dual text encoders (CLIP-L text_encoder_1 + OpenCLIP-G text_encoder_2)
#   - Placeholder token and embedding added to BOTH encoders
#   - SDXL UNet requires added_cond_kwargs (pooled text_embeds + time_ids)
#   - VAE scaling factor: 0.13025
#   - Default resolution: 1024
#
# PTI Phase schedule (same as original):
#   Steps 0 .. unfreeze_lora_step   -> train TI embeddings only (LoRA lr=0)
#   Steps unfreeze_lora_step .. end -> train LoRA only (TI lr=0)

import argparse
import hashlib
import inspect
import itertools
import math
import os
import random
import re
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from lora_diffusion import (
    extract_lora_ups_down,
    inject_trainable_lora,
    safetensors_available,
    save_lora_weight,
    save_safeloras_with_embeds,
)
from lora_diffusion.xformers_utils import set_use_memory_efficient_attention_xformers


# ---------------------------------------------------------------------------
# Templates (same as original)
# ---------------------------------------------------------------------------

imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


def _randomset(lis):
    return [x for x in lis if random.random() < 0.5]


def _shuffle(lis):
    return random.sample(lis, len(lis))


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DreamBoothTiDatasetSDXL(Dataset):
    """
    Dataset for SDXL PTI (DreamBooth + Textual Inversion) training.
    Tokenizes prompts with both SDXL tokenizers.
    """

    def __init__(
        self,
        instance_data_root,
        learnable_property,
        placeholder_token,
        stochastic_attribute,
        tokenizer_1,
        tokenizer_2,
        class_data_root=None,
        class_prompt=None,
        size=1024,
        center_crop=False,
        color_jitter=False,
        resize=True,
        h_flip=True,
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

        self.placeholder_token = placeholder_token
        self.stochastic_attribute = (
            stochastic_attribute.split(",") if stochastic_attribute else []
        )
        self.templates = (
            imagenet_style_templates_small
            if learnable_property == "style"
            else imagenet_templates_small
        )
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

    def _tokenize(self, text):
        ids_1 = self.tokenizer_1(
            text,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer_1.model_max_length,
        ).input_ids
        ids_2 = self.tokenizer_2(
            text,
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

        text = random.choice(self.templates).format(
            ", ".join(
                [self.placeholder_token] + _shuffle(_randomset(self.stochastic_attribute))
            )
        )
        example["instance_prompt_ids_1"], example["instance_prompt_ids_2"] = self._tokenize(text)

        if self.class_data_root:
            class_image = Image.open(
                self.class_images_path[index % self.num_class_images]
            )
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids_1"], example["class_prompt_ids_2"] = self._tokenize(
                self.class_prompt
            )

        return example


class PromptDataset(Dataset):
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

def save_progress(
    text_encoder_1,
    text_encoder_2,
    placeholder_token_id_1,
    placeholder_token_id_2,
    accelerator,
    args,
    save_path,
):
    """Save TI embedding from both SDXL text encoders."""
    logger.info("Saving embeddings")

    te1 = accelerator.unwrap_model(text_encoder_1)
    te2 = accelerator.unwrap_model(text_encoder_2)

    embed_1 = te1.get_input_embeddings().weight[placeholder_token_id_1]
    embed_2 = te2.get_input_embeddings().weight[placeholder_token_id_2]

    print(f"TE1 Learned Embed[:4]: {embed_1[:4]}")
    print(f"TE2 Learned Embed[:4]: {embed_2[:4]}")

    learned_embeds_dict = {
        args.placeholder_token: embed_1.detach().cpu(),
        args.placeholder_token + "_te2": embed_2.detach().cpu(),
    }
    torch.save(learned_embeds_dict, save_path)
    print(f"Saved TI embeddings to {save_path}")


def get_learned_embeds_sdxl(
    text_encoder_1,
    text_encoder_2,
    placeholder_token_id_1,
    placeholder_token_id_2,
    placeholder_token,
):
    embed_1 = text_encoder_1.get_input_embeddings().weight[placeholder_token_id_1]
    embed_2 = text_encoder_2.get_input_embeddings().weight[placeholder_token_id_2]
    return {
        placeholder_token: embed_1.detach().cpu(),
        placeholder_token + "_te2": embed_2.detach().cpu(),
    }


def ensure_finite(name, tensor):
    if torch.isfinite(tensor).all():
        return

    finite_values = tensor[torch.isfinite(tensor)]
    if finite_values.numel() > 0:
        min_value = float(finite_values.min().item())
        max_value = float(finite_values.max().item())
    else:
        min_value = None
        max_value = None

    raise RuntimeError(
        f"Non-finite values detected in {name}: "
        f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"finite_min={min_value} finite_max={max_value}"
    )


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def unfreeze_params(params):
    for param in params:
        param.requires_grad = True


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="SDXL LoRA + Textual Inversion (PTI) training.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--pretrained_vae_name_or_path", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--class_data_dir", type=str, default=None)
    parser.add_argument(
        "--placeholder_token",
        type=str,
        required=True,
        help="The special token to represent the learned concept (e.g. '<mygo>').",
    )
    parser.add_argument(
        "--initializer_token",
        type=str,
        required=True,
        help="A real word to initialize the placeholder embedding (e.g. 'anime').",
    )
    parser.add_argument(
        "--learnable_property",
        type=str,
        default="style",
        choices=["object", "style"],
        help="Whether to learn a style or an object.",
    )
    parser.add_argument(
        "--stochastic_attribute",
        type=str,
        default=None,
        help="Comma-separated attributes to randomly add to the prompt.",
    )
    parser.add_argument("--class_prompt", type=str, default=None)
    parser.add_argument("--with_prior_preservation", default=False, action="store_true")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0)
    parser.add_argument("--num_class_images", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="lora-sdxl-pti-output")
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["pt", "safe", "both"],
        default="safe",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--color_jitter", action="store_true")
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Train text_encoder_1 (CLIP-L) with LoRA after TI phase.",
    )
    parser.add_argument(
        "--train_text_encoder_2",
        action="store_true",
        help="Train text_encoder_2 (OpenCLIP-G) with LoRA after TI phase.",
    )
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--sample_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="LR for UNet LoRA (LoRA phase).")
    parser.add_argument("--learning_rate_text", type=float, default=5e-6,
                        help="LR for text encoder LoRA (LoRA phase).")
    parser.add_argument("--learning_rate_ti", type=float, default=5e-4,
                        help="LR for TI embeddings (TI phase).")
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument(
        "--unfreeze_lora_step",
        type=int,
        default=500,
        help="After this many steps, switch from TI training to LoRA training.",
    )
    parser.add_argument(
        "--just_ti",
        action="store_true",
        help="Only train TI embeddings, skip LoRA.",
    )
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
    parser.add_argument("--resize", type=bool, default=True)
    parser.add_argument("--use_xformers", action="store_true")
    parser.add_argument("--logging_dir", type=str, default="logs")

    args = parser.parse_args(input_args) if input_args is not None else parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("--class_data_dir required for prior preservation.")
        if args.class_prompt is None:
            raise ValueError("--class_prompt required for prior preservation.")

    if not safetensors_available:
        if args.output_format in ("safe", "both"):
            print("safetensors not available - falling back to pt format")
            args.output_format = "pt"

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    kwargs_handlers = None
    if args.train_text_encoder or args.train_text_encoder_2:
        kwargs_handlers = [
            DistributedDataParallelKwargs(find_unused_parameters=True)
        ]

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=logging_dir,
        kwargs_handlers=kwargs_handlers,
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
    # Load tokenizers
    # ------------------------------------------------------------------
    tokenizer_1 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision
    )

    # Add placeholder token to BOTH tokenizers
    num_added_1 = tokenizer_1.add_tokens(args.placeholder_token)
    if num_added_1 == 0:
        raise ValueError(
            f"tokenizer_1 already contains '{args.placeholder_token}'. Choose a different placeholder."
        )
    num_added_2 = tokenizer_2.add_tokens(args.placeholder_token)
    if num_added_2 == 0:
        raise ValueError(
            f"tokenizer_2 already contains '{args.placeholder_token}'. Choose a different placeholder."
        )

    # Resolve initializer token id for both tokenizers
    init_ids_1 = tokenizer_1.encode(args.initializer_token, add_special_tokens=False)
    init_ids_2 = tokenizer_2.encode(args.initializer_token, add_special_tokens=False)
    if len(init_ids_1) > 1:
        raise ValueError("initializer_token must map to a single token in tokenizer_1.")
    if len(init_ids_2) > 1:
        raise ValueError("initializer_token must map to a single token in tokenizer_2.")

    initializer_token_id_1 = init_ids_1[0]
    initializer_token_id_2 = init_ids_2[0]
    placeholder_token_id_1 = tokenizer_1.convert_tokens_to_ids(args.placeholder_token)
    placeholder_token_id_2 = tokenizer_2.convert_tokens_to_ids(args.placeholder_token)

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    text_encoder_1 = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision
    )

    # SDXL text-encoder weights load in fp16 by default. Keep them in fp32 for any
    # trainable path so AdamW does not update the placeholder embeddings in half precision.
    text_encoder_1.to(dtype=torch.float32)
    text_encoder_2.to(dtype=torch.float32)

    # Resize token tables for the new placeholder
    text_encoder_1.resize_token_embeddings(len(tokenizer_1))
    text_encoder_2.resize_token_embeddings(len(tokenizer_2))

    # Initialize placeholder from initializer token
    embeds_1 = text_encoder_1.get_input_embeddings().weight.data
    embeds_2 = text_encoder_2.get_input_embeddings().weight.data
    embeds_1[placeholder_token_id_1] = embeds_1[initializer_token_id_1].clone()
    embeds_2[placeholder_token_id_2] = embeds_2[initializer_token_id_2].clone()

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,
        subfolder=None if args.pretrained_vae_name_or_path else "vae",
        revision=None if args.pretrained_vae_name_or_path else args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    # ------------------------------------------------------------------
    # LoRA injection
    # ------------------------------------------------------------------
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    unet_lora_params, _ = inject_trainable_lora(unet, r=args.lora_rank)
    for _up, _down in extract_lora_ups_down(unet):
        print("Before training: UNet first LoRA up:", _up.weight.data)
        print("Before training: UNet first LoRA down:", _down.weight.data)
        break

    # Freeze all text encoder params EXCEPT the new token embedding
    # (TI phase: only embedding is updated; LoRA phase: LoRA params are updated)
    for te in [text_encoder_1, text_encoder_2]:
        freeze_params(te.parameters())
    # Unfreeze only the embedding layer (needed for TI training)
    unfreeze_params(text_encoder_1.get_input_embeddings().parameters())
    unfreeze_params(text_encoder_2.get_input_embeddings().parameters())

    # Only inject text-encoder LoRA when those branches are explicitly enabled.
    te1_lora_params = []
    te2_lora_params = []
    if args.train_text_encoder:
        te1_lora_params, _ = inject_trainable_lora(
            text_encoder_1, target_replace_module=["CLIPAttention"], r=args.lora_rank
        )
        for _up, _down in extract_lora_ups_down(
            text_encoder_1, target_replace_module=["CLIPAttention"]
        ):
            print("Before training: TE1 first LoRA up:", _up.weight.data)
            break
    if args.train_text_encoder_2:
        te2_lora_params, _ = inject_trainable_lora(
            text_encoder_2, target_replace_module=["CLIPAttention"], r=args.lora_rank
        )
        for _up, _down in extract_lora_ups_down(
            text_encoder_2, target_replace_module=["CLIPAttention"]
        ):
            print("Before training: TE2 first LoRA up:", _up.weight.data)
            break

    if args.use_xformers:
        set_use_memory_efficient_attention_xformers(unet, True)
        set_use_memory_efficient_attention_xformers(vae, True)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        text_encoder_1.gradient_checkpointing_enable()
        text_encoder_2.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # ------------------------------------------------------------------
    # Optimizer  (3 or 1 param groups depending on just_ti)
    # ------------------------------------------------------------------
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Install bitsandbytes for 8-bit Adam: `pip install bitsandbytes`.")
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Group 0: UNet LoRA  (lr=0 during TI phase, set to learning_rate in LoRA phase)
    # Group 1: TE1 LoRA   (lr=0 during TI phase)
    # Group 2: TE2 LoRA   (lr=0 during TI phase)
    # Group 3: TE1 embedding (lr=learning_rate_ti during TI phase, 0 in LoRA phase)
    # Group 4: TE2 embedding (lr=learning_rate_ti during TI phase, 0 in LoRA phase)

    if args.just_ti:
        params_to_optimize = [
            {"params": text_encoder_1.get_input_embeddings().parameters(),
             "lr": args.learning_rate_ti},
            {"params": text_encoder_2.get_input_embeddings().parameters(),
             "lr": args.learning_rate_ti},
        ]
        # group indices: 0=TE1 embed, 1=TE2 embed
        IDX_UNET = None
        IDX_TE1_LORA = None
        IDX_TE2_LORA = None
        IDX_TE1_EMBED = 0
        IDX_TE2_EMBED = 1
    else:
        params_to_optimize = [
            {"params": itertools.chain(*unet_lora_params), "lr": args.learning_rate},
        ]
        IDX_UNET = 0
        IDX_TE1_LORA = None
        IDX_TE2_LORA = None

        if args.train_text_encoder:
            IDX_TE1_LORA = len(params_to_optimize)
            params_to_optimize.append(
                {"params": itertools.chain(*te1_lora_params), "lr": args.learning_rate_text}
            )

        if args.train_text_encoder_2:
            IDX_TE2_LORA = len(params_to_optimize)
            params_to_optimize.append(
                {"params": itertools.chain(*te2_lora_params), "lr": args.learning_rate_text}
            )

        IDX_TE1_EMBED = len(params_to_optimize)
        params_to_optimize.append(
            {"params": text_encoder_1.get_input_embeddings().parameters(),
             "lr": args.learning_rate_ti}
        )
        IDX_TE2_EMBED = len(params_to_optimize)
        params_to_optimize.append(
            {"params": text_encoder_2.get_input_embeddings().parameters(),
             "lr": args.learning_rate_ti}
        )

    optimizer = optimizer_class(
        params_to_optimize,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_config(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    train_dataset = DreamBoothTiDatasetSDXL(
        instance_data_root=args.instance_data_dir,
        placeholder_token=args.placeholder_token,
        stochastic_attribute=args.stochastic_attribute,
        learnable_property=args.learnable_property,
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
    # LR scheduler + accelerator prepare
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

    (
        unet,
        text_encoder_1,
        text_encoder_2,
        train_dataloader,
    ) = accelerator.prepare(unet, text_encoder_1, text_encoder_2, train_dataloader)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth_sdxl_pti", config=vars(args))

    total_batch_size = (
        args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )
    print("***** Running SDXL PTI training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num epochs = {args.num_train_epochs}")
    print(f"  Batch size (per device) = {args.train_batch_size}")
    print(f"  Total batch size = {total_batch_size}")
    print(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    print(f"  TI phase: 0 .. {args.unfreeze_lora_step}")
    print(f"  LoRA phase: {args.unfreeze_lora_step} .. {args.max_train_steps}")

    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")
    global_step = 0
    last_save = 0

    # Save original embeddings so we can protect non-placeholder tokens
    orig_embeds_1 = (
        accelerator.unwrap_model(text_encoder_1)
        .get_input_embeddings()
        .weight.data.clone()
    )
    orig_embeds_2 = (
        accelerator.unwrap_model(text_encoder_2)
        .get_input_embeddings()
        .weight.data.clone()
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(args.num_train_epochs):
        unet.train()
        text_encoder_1.train()
        text_encoder_2.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):

                # *** Phase gating: switch lr between TI and LoRA phases ***
                if not args.just_ti:
                    if global_step < args.unfreeze_lora_step:
                        # TI phase: freeze LoRA, train embeddings
                        optimizer.param_groups[IDX_UNET]["lr"] = 0.0
                        if IDX_TE1_LORA is not None:
                            optimizer.param_groups[IDX_TE1_LORA]["lr"] = 0.0
                        if IDX_TE2_LORA is not None:
                            optimizer.param_groups[IDX_TE2_LORA]["lr"] = 0.0
                        optimizer.param_groups[IDX_TE1_EMBED]["lr"] = args.learning_rate_ti
                        optimizer.param_groups[IDX_TE2_EMBED]["lr"] = args.learning_rate_ti
                    else:
                        # LoRA phase: train LoRA, freeze embeddings
                        optimizer.param_groups[IDX_UNET]["lr"] = args.learning_rate
                        if IDX_TE1_LORA is not None:
                            optimizer.param_groups[IDX_TE1_LORA]["lr"] = args.learning_rate_text
                        if IDX_TE2_LORA is not None:
                            optimizer.param_groups[IDX_TE2_LORA]["lr"] = args.learning_rate_text
                        optimizer.param_groups[IDX_TE1_EMBED]["lr"] = 0.0
                        optimizer.param_groups[IDX_TE2_EMBED]["lr"] = 0.0

                # 1. Encode images to latents (SDXL scaling 0.13025)
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * 0.13025
                ensure_finite("latents", latents)

                # 2. Sample noise + timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 3. Dual text encoder forward pass
                te1_out = text_encoder_1(
                    batch["input_ids_1"].to(accelerator.device),
                    output_hidden_states=True,
                )
                te2_out = text_encoder_2(
                    batch["input_ids_2"].to(accelerator.device),
                    output_hidden_states=True,
                )
                # Concat hidden_states[-2] from both: (B, seq, 768+1280=2048)
                encoder_hidden_states = torch.cat(
                    [te1_out.hidden_states[-2], te2_out.hidden_states[-2]], dim=-1
                ).to(dtype=weight_dtype)
                # Pooled from text_encoder_2
                pooled_prompt_embeds = te2_out.text_embeds.to(dtype=weight_dtype)
                ensure_finite("encoder_hidden_states", encoder_hidden_states)
                ensure_finite("pooled_prompt_embeds", pooled_prompt_embeds)

                # 4. SDXL time IDs
                resolution = args.resolution
                add_time_ids = torch.tensor(
                    [[resolution, resolution, 0, 0, resolution, resolution]] * bsz,
                    dtype=weight_dtype,
                    device=accelerator.device,
                )
                added_cond_kwargs = {
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids,
                }

                # 5. Predict noise
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample
                ensure_finite("model_pred", model_pred)

                # 6. Loss target
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

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

                ensure_finite("loss", loss.reshape(1))

                accelerator.backward(loss)

                te1_unwrapped = accelerator.unwrap_model(text_encoder_1)
                te2_unwrapped = accelerator.unwrap_model(text_encoder_2)
                ensure_finite(
                    "placeholder_embed_te1_pre_step",
                    te1_unwrapped.get_input_embeddings().weight[placeholder_token_id_1],
                )
                ensure_finite(
                    "placeholder_embed_te2_pre_step",
                    te2_unwrapped.get_input_embeddings().weight[placeholder_token_id_2],
                )
                ensure_finite(
                    "placeholder_embed_te1_grad",
                    te1_unwrapped.get_input_embeddings().weight.grad[placeholder_token_id_1],
                )
                ensure_finite(
                    "placeholder_embed_te2_grad",
                    te2_unwrapped.get_input_embeddings().weight.grad[placeholder_token_id_2],
                )

                if accelerator.sync_gradients:
                    params_to_clip = itertools.chain(
                        unet.parameters(),
                        text_encoder_1.parameters(),
                        text_encoder_2.parameters(),
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Protect non-placeholder embeddings from being updated
                index_no_updates_1 = torch.arange(len(tokenizer_1)) != placeholder_token_id_1
                index_no_updates_2 = torch.arange(len(tokenizer_2)) != placeholder_token_id_2
                with torch.no_grad():
                    te1_unwrapped.get_input_embeddings().weight[
                        index_no_updates_1
                    ] = orig_embeds_1[index_no_updates_1]
                    te2_unwrapped.get_input_embeddings().weight[
                        index_no_updates_2
                    ] = orig_embeds_2[index_no_updates_2]
                    ensure_finite(
                        "placeholder_embed_te1",
                        te1_unwrapped.get_input_embeddings().weight[placeholder_token_id_1],
                    )
                    ensure_finite(
                        "placeholder_embed_te2",
                        te2_unwrapped.get_input_embeddings().weight[placeholder_token_id_2],
                    )

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if args.save_steps and global_step - last_save >= args.save_steps:
                    if accelerator.is_main_process:
                        accepts_keep_fp32_wrapper = "keep_fp32_wrapper" in set(
                            inspect.signature(accelerator.unwrap_model).parameters.keys()
                        )
                        extra_args = (
                            {"keep_fp32_wrapper": True} if accepts_keep_fp32_wrapper else {}
                        )

                        unet_uw = accelerator.unwrap_model(unet, **extra_args)
                        te1_uw = accelerator.unwrap_model(text_encoder_1, **extra_args)
                        te2_uw = accelerator.unwrap_model(text_encoder_2, **extra_args)
                        checkpoint_base = (
                            f"{args.output_dir}/lora_weight_e{epoch}_s{global_step}"
                        )
                        embeds = get_learned_embeds_sdxl(
                            te1_uw,
                            te2_uw,
                            placeholder_token_id_1,
                            placeholder_token_id_2,
                            args.placeholder_token,
                        )

                        if args.output_format in ("pt", "both"):
                            save_lora_weight(
                                unet_uw,
                                checkpoint_base + ".pt",
                            )
                            if args.train_text_encoder:
                                save_lora_weight(
                                    te1_uw,
                                    checkpoint_base + ".text_encoder.pt",
                                    target_replace_module=["CLIPAttention"],
                                )
                            if args.train_text_encoder_2:
                                save_lora_weight(
                                    te2_uw,
                                    checkpoint_base + ".text_encoder_2.pt",
                                    target_replace_module=["CLIPAttention"],
                                )
                            torch.save(embeds, checkpoint_base + ".ti.pt")
                            print(f"Saved TI embeddings to {checkpoint_base}.ti.pt")

                        if args.output_format in ("safe", "both"):
                            loras = {"unet": (unet_uw, {"CrossAttention", "Attention", "GEGLU"})}
                            if args.train_text_encoder:
                                loras["text_encoder"] = (te1_uw, {"CLIPAttention"})
                            if args.train_text_encoder_2:
                                loras["text_encoder_2"] = (te2_uw, {"CLIPAttention"})

                            save_safeloras_with_embeds(
                                loras, embeds, checkpoint_base + ".safetensors"
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
    # Final save
    # ------------------------------------------------------------------
    if accelerator.is_main_process:
        unet_uw = accelerator.unwrap_model(unet)
        te1_uw = accelerator.unwrap_model(text_encoder_1)
        te2_uw = accelerator.unwrap_model(text_encoder_2)

        print("\n\nSDXL PTI TRAINING DONE!\n\n")

        embeds = get_learned_embeds_sdxl(
            te1_uw,
            te2_uw,
            placeholder_token_id_1,
            placeholder_token_id_2,
            args.placeholder_token,
        )

        if args.output_format in ("pt", "both"):
            save_lora_weight(unet_uw, args.output_dir + "/lora_weight.pt")
            if args.train_text_encoder:
                save_lora_weight(
                    te1_uw,
                    args.output_dir + "/lora_weight.text_encoder.pt",
                    target_replace_module=["CLIPAttention"],
                )
            if args.train_text_encoder_2:
                save_lora_weight(
                    te2_uw,
                    args.output_dir + "/lora_weight.text_encoder_2.pt",
                    target_replace_module=["CLIPAttention"],
                )
            torch.save(embeds, args.output_dir + "/lora_weight.ti.pt")

        if args.output_format in ("safe", "both"):
            loras = {"unet": (unet_uw, {"CrossAttention", "Attention", "GEGLU"})}
            if args.train_text_encoder:
                loras["text_encoder"] = (te1_uw, {"CLIPAttention"})
            if args.train_text_encoder_2:
                loras["text_encoder_2"] = (te2_uw, {"CLIPAttention"})

            save_safeloras_with_embeds(
                loras, embeds, args.output_dir + "/lora_weight.safetensors"
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
