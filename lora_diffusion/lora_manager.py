from typing import List, Optional
import torch
from safetensors import safe_open
from diffusers import StableDiffusionPipeline
from .lora import (
    monkeypatch_or_replace_safeloras,
    apply_learned_embed_in_clip,
    set_lora_diag,
    parse_safeloras_embeds,
    split_sdxl_learned_embeds,
)


def lora_join(lora_safetenors: list):
    metadatas = [dict(safelora.metadata()) for safelora in lora_safetenors]
    _total_metadata = {}
    total_metadata = {}
    total_tensor = {}
    total_rank = 0
    ranklist = []
    for _metadata in metadatas:
        rankset = []
        for k, v in _metadata.items():
            if k.endswith("rank"):
                rankset.append(int(v))

        assert len(set(rankset)) <= 1, "Rank should be the same per model"
        if len(rankset) == 0:
            rankset = [0]

        total_rank += rankset[0]
        _total_metadata.update(_metadata)
        ranklist.append(rankset[0])

    # remove metadata about tokens
    for k, v in _total_metadata.items():
        if v != "<embed>":
            total_metadata[k] = v

    tensorkeys = set()
    for safelora in lora_safetenors:
        tensorkeys.update(safelora.keys())

    for keys in tensorkeys:
        if keys.startswith("text_encoder") or keys.startswith("unet"):
            tensorset = [safelora.get_tensor(keys) for safelora in lora_safetenors]

            is_down = keys.endswith("down")

            if is_down:
                _tensor = torch.cat(tensorset, dim=0)
                assert _tensor.shape[0] == total_rank
            else:
                _tensor = torch.cat(tensorset, dim=1)
                assert _tensor.shape[1] == total_rank

            total_tensor[keys] = _tensor
            keys_rank = ":".join(keys.split(":")[:-1]) + ":rank"
            total_metadata[keys_rank] = str(total_rank)
    token_size_list = []
    for idx, safelora in enumerate(lora_safetenors):
        tokens = [k for k, v in safelora.metadata().items() if v == "<embed>"]
        for jdx, token in enumerate(sorted(tokens)):

            total_tensor[f"<s{idx}-{jdx}>"] = safelora.get_tensor(token)
            total_metadata[f"<s{idx}-{jdx}>"] = "<embed>"

            print(f"Embedding {token} replaced to <s{idx}-{jdx}>")

        token_size_list.append(len(tokens))

    return total_tensor, total_metadata, ranklist, token_size_list


def _derive_join_token(token: str, suffix_index: int) -> str:
    if token.startswith("<") and token.endswith(">"):
        return f"{token[:-1]}-{suffix_index}>"
    return f"{token}_{suffix_index}"


def lora_join_sdxl(
    lora_safetenors: list,
    token_aliases: Optional[List[str]] = None,
):
    if token_aliases is not None and len(token_aliases) != len(lora_safetenors):
        raise ValueError("token_aliases must match the number of input LoRAs")

    metadatas = [dict(safelora.metadata()) for safelora in lora_safetenors]
    _total_metadata = {}
    total_metadata = {}
    total_tensor = {}
    total_rank = 0
    ranklist = []

    for _metadata in metadatas:
        rankset = []
        for k, v in _metadata.items():
            if k.endswith("rank"):
                rankset.append(int(v))

        assert len(set(rankset)) <= 1, "Rank should be the same per model"
        if len(rankset) == 0:
            rankset = [0]

        total_rank += rankset[0]
        _total_metadata.update(_metadata)
        ranklist.append(rankset[0])

    for k, v in _total_metadata.items():
        if v != "<embed>":
            total_metadata[k] = v

    tensorkeys = set()
    for safelora in lora_safetenors:
        tensorkeys.update(safelora.keys())

    for keys in tensorkeys:
        if keys.startswith("text_encoder") or keys.startswith("unet"):
            tensorset = [safelora.get_tensor(keys) for safelora in lora_safetenors]

            is_down = keys.endswith("down")

            if is_down:
                _tensor = torch.cat(tensorset, dim=0)
                assert _tensor.shape[0] == total_rank
            else:
                _tensor = torch.cat(tensorset, dim=1)
                assert _tensor.shape[1] == total_rank

            total_tensor[keys] = _tensor
            keys_rank = ":".join(keys.split(":")[:-1]) + ":rank"
            total_metadata[keys_rank] = str(total_rank)

    used_tokens = set()
    token_records = []

    for idx, safelora in enumerate(lora_safetenors):
        learned_embeds = parse_safeloras_embeds(safelora)
        paired_embeds = split_sdxl_learned_embeds(learned_embeds)
        preferred_alias = None
        if token_aliases is not None:
            preferred_alias = token_aliases[idx].strip() or None

        if preferred_alias and len(paired_embeds) > 1:
            raise ValueError(
                "A single token alias cannot represent multiple concepts in one SDXL LoRA"
            )

        for base_token, pair in sorted(paired_embeds.items()):
            merged_token = preferred_alias or base_token
            suffix_index = 1

            while merged_token in used_tokens:
                merged_token = _derive_join_token(preferred_alias or base_token, suffix_index)
                suffix_index += 1

            total_tensor[merged_token] = pair["te1"]
            total_metadata[merged_token] = "<embed>"
            used_tokens.add(merged_token)

            if "te2" in pair:
                total_tensor[f"{merged_token}_te2"] = pair["te2"]
                total_metadata[f"{merged_token}_te2"] = "<embed>"

            token_records.append(
                {
                    "source_index": idx,
                    "source_token": base_token,
                    "merged_token": merged_token,
                }
            )
            print(
                f"Embedding pair {base_token} / {base_token}_te2 saved as "
                f"{merged_token} / {merged_token}_te2"
            )

    return total_tensor, total_metadata, ranklist, token_records


class DummySafeTensorObject:
    def __init__(self, tensor: dict, metadata):
        self.tensor = tensor
        self._metadata = metadata

    def keys(self):
        return self.tensor.keys()

    def metadata(self):
        return self._metadata

    def get_tensor(self, key):
        return self.tensor[key]


class LoRAManager:
    def __init__(self, lora_paths_list: List[str], pipe: StableDiffusionPipeline):

        self.lora_paths_list = lora_paths_list
        self.pipe = pipe
        self._setup()

    def _setup(self):

        self._lora_safetenors = [
            safe_open(path, framework="pt", device="cpu")
            for path in self.lora_paths_list
        ]

        (
            total_tensor,
            total_metadata,
            self.ranklist,
            self.token_size_list,
        ) = lora_join(self._lora_safetenors)

        self.total_safelora = DummySafeTensorObject(total_tensor, total_metadata)

        monkeypatch_or_replace_safeloras(self.pipe, self.total_safelora)
        tok_dict = parse_safeloras_embeds(self.total_safelora)

        apply_learned_embed_in_clip(
            tok_dict,
            self.pipe.text_encoder,
            self.pipe.tokenizer,
            token=None,
            idempotent=True,
        )

    def tune(self, scales):

        assert len(scales) == len(
            self.ranklist
        ), "Scale list should be the same length as ranklist"

        diags = []
        for scale, rank in zip(scales, self.ranklist):
            diags = diags + [scale] * rank

        set_lora_diag(self.pipe.unet, torch.tensor(diags))

    def prompt(self, prompt):
        if prompt is not None:
            for idx, tok_size in enumerate(self.token_size_list):
                prompt = prompt.replace(
                    f"<{idx + 1}>",
                    "".join([f"<s{idx}-{jdx}>" for jdx in range(tok_size)]),
                )
        # TODO : Rescale LoRA + Text inputs based on prompt scale params

        return prompt
