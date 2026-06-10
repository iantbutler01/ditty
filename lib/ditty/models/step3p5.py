from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM
from transformers.dynamic_module_utils import get_class_from_dynamic_module


CONFIG_NAME = "config.json"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
SAFE_WEIGHTS_NAME = "model.safetensors"
MTP_SPECIAL_NAMES = ("enorm", "hnorm", "eh_proj", "shared_head")
STEP3P5_AUTO_CONFIG_CLASS = "configuration_step3p5.Step3p5Config"


@dataclass(frozen=True)
class Step3p5MTPWeightLoadReport:
    loaded: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]


@dataclass(frozen=True)
class Step3p5MTPKeySummary:
    layers: tuple[int, ...]
    has_embed_tokens: bool
    key_classes_by_layer: dict[int, tuple[str, ...]]


def _download_kwargs_from_load_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "cache_dir": kwargs.get("cache_dir"),
        "force_download": bool(kwargs.get("force_download", False)),
        "local_files_only": bool(kwargs.get("local_files_only", False)),
        "proxies": kwargs.get("proxies"),
        "revision": kwargs.get("revision"),
        "token": kwargs.get("token"),
    }


def _resolve_config_path(
    model_path: str,
    *,
    revision: str | None = None,
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> Path:
    local_path = Path(model_path)
    if local_path.is_dir():
        config_path = local_path / (subfolder or "") / CONFIG_NAME
        if config_path.exists():
            return config_path
    return Path(
        hf_hub_download(
            repo_id=model_path,
            filename=CONFIG_NAME,
            subfolder=subfolder,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
        )
    )


def _step3p5_config_class(model_path: str, config_payload: dict[str, Any], kwargs: dict[str, Any]):
    auto_map = config_payload.get("auto_map") or {}
    class_reference = auto_map.get("AutoConfig") or STEP3P5_AUTO_CONFIG_CLASS
    return get_class_from_dynamic_module(
        class_reference,
        model_path,
        **_download_kwargs_from_load_kwargs(kwargs),
        code_revision=kwargs.get("code_revision"),
    )


def _step3p5_config_from_payload(
    config_payload: dict[str, Any],
    config_cls: type,
) -> Any:
    payload = copy.deepcopy(config_payload)
    layer_types = payload.get("layer_types")
    num_hidden_layers = int(payload.get("num_hidden_layers", 0) or 0)
    num_mtp_layers = int(payload.get("num_nextn_predict_layers", 0) or 0)
    if (
        isinstance(layer_types, list)
        and num_hidden_layers > 0
        and len(layer_types) == num_hidden_layers + num_mtp_layers
    ):
        payload["layer_types"] = layer_types[:num_hidden_layers]

    config = config_cls(**payload)
    if isinstance(layer_types, list):
        config._ditty_full_layer_types = tuple(layer_types)
    return config


def _load_step3p5_config(model_path: str, kwargs: dict[str, Any]) -> Any:
    config_path = _resolve_config_path(
        model_path,
        revision=kwargs.get("revision"),
        token=kwargs.get("token"),
        cache_dir=kwargs.get("cache_dir"),
        force_download=bool(kwargs.get("force_download", False)),
        local_files_only=bool(kwargs.get("local_files_only", False)),
        subfolder=kwargs.get("subfolder"),
    )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    config_cls = _step3p5_config_class(model_path, payload, kwargs)
    return _step3p5_config_from_payload(payload, config_cls)


def _step3p5_mtp_config_view(config: Any) -> Any:
    mtp_config = copy.copy(config)
    full_layer_types = getattr(config, "_ditty_full_layer_types", None)
    if full_layer_types is not None:
        mtp_config.layer_types = list(full_layer_types)
    return mtp_config


def get_step3p5_mtp_layer_index(config: Any, weight_name: str) -> int | None:
    depth = int(getattr(config, "num_nextn_predict_layers", 0) or 0)
    start = int(getattr(config, "num_hidden_layers", 0) or 0)
    for layer_idx in range(start, start + depth):
        if weight_name.startswith(f"model.layers.{layer_idx}.") or weight_name.startswith(
            f"layers.{layer_idx}."
        ):
            return layer_idx
    return None


def rewrite_step3p5_mtp_weight_name(config: Any, weight_name: str) -> str | None:
    if "rotary_emb.inv_freq" in weight_name:
        return None
    if weight_name.endswith("embed_tokens.weight"):
        return "mtp_embed_tokens.weight"

    spec_layer = get_step3p5_mtp_layer_index(config, weight_name)
    if spec_layer is None:
        return None

    for prefix in (f"model.layers.{spec_layer}.", f"layers.{spec_layer}."):
        if weight_name.startswith(prefix):
            suffix = weight_name[len(prefix) :]
            break
    else:
        return None

    suffix = suffix.removeprefix("transformer.")
    if suffix.startswith("embed_tokens."):
        return "mtp_embed_tokens.weight"
    if suffix.startswith(MTP_SPECIAL_NAMES):
        return f"mtp_layers.{spec_layer}.{suffix}"
    return f"mtp_layers.{spec_layer}.mtp_block.{suffix}"


def _step3p5_mtp_key_class(config: Any, weight_name: str) -> tuple[int | None, str | None]:
    if weight_name.endswith("embed_tokens.weight"):
        return None, "embed_tokens"

    spec_layer = get_step3p5_mtp_layer_index(config, weight_name)
    if spec_layer is None:
        return None, None
    for prefix in (f"model.layers.{spec_layer}.", f"layers.{spec_layer}."):
        if weight_name.startswith(prefix):
            suffix = weight_name[len(prefix) :].removeprefix("transformer.")
            break
    else:
        return spec_layer, None
    if suffix.startswith("enorm."):
        return spec_layer, "enorm"
    if suffix.startswith("hnorm."):
        return spec_layer, "hnorm"
    if suffix.startswith("eh_proj."):
        return spec_layer, "eh_proj"
    if suffix.startswith("shared_head."):
        return spec_layer, "shared_head"
    return spec_layer, "mtp_block"


def _resolve_safetensors_index(model_path: str, *, revision: str | None = None, token: str | None = None) -> Path:
    local_path = Path(model_path)
    if local_path.is_dir():
        index_path = local_path / SAFE_WEIGHTS_INDEX_NAME
        if index_path.exists():
            return index_path
        single_path = local_path / SAFE_WEIGHTS_NAME
        if single_path.exists():
            return single_path
    return Path(
        hf_hub_download(
            repo_id=model_path,
            filename=SAFE_WEIGHTS_INDEX_NAME,
            revision=revision,
            token=token,
        )
    )


def _iter_safetensor_keys(index_or_file: Path) -> Iterable[tuple[str, Path]]:
    if index_or_file.name == SAFE_WEIGHTS_NAME:
        with safe_open(str(index_or_file), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                yield key, index_or_file
        return

    payload = json.loads(index_or_file.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{index_or_file} did not contain a safetensors weight_map.")
    for key, filename in weight_map.items():
        yield key, index_or_file.parent / str(filename)


def _resolve_shard_path(
    model_path: str,
    shard_path: Path,
    *,
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    if shard_path.exists():
        return shard_path
    return Path(
        hf_hub_download(
            repo_id=model_path,
            filename=shard_path.name,
            revision=revision,
            token=token,
        )
    )


def _mtp_parameter_names(model: nn.Module) -> set[str]:
    return {
        name
        for name, _ in model.named_parameters()
        if name == "mtp_embed_tokens.weight" or name.startswith("mtp_layers.")
    }


def summarize_step3p5_mtp_checkpoint_keys(
    model_path: str,
    config: Any,
    *,
    revision: str | None = None,
    token: str | None = None,
) -> Step3p5MTPKeySummary:
    index_or_file = _resolve_safetensors_index(model_path, revision=revision, token=token)
    has_embed_tokens = False
    classes_by_layer: dict[int, set[str]] = {}
    for remote_name, _ in _iter_safetensor_keys(index_or_file):
        layer_idx, key_class = _step3p5_mtp_key_class(config, remote_name)
        if key_class == "embed_tokens":
            has_embed_tokens = True
            continue
        if layer_idx is None or key_class is None:
            continue
        classes_by_layer.setdefault(layer_idx, set()).add(key_class)
    return Step3p5MTPKeySummary(
        layers=tuple(sorted(classes_by_layer)),
        has_embed_tokens=has_embed_tokens,
        key_classes_by_layer={
            layer_idx: tuple(sorted(key_classes))
            for layer_idx, key_classes in sorted(classes_by_layer.items())
        },
    )


def load_step3p5_mtp_weights(
    model: nn.Module,
    model_path: str,
    *,
    strict: bool = True,
    revision: str | None = None,
    token: str | None = None,
) -> Step3p5MTPWeightLoadReport:
    index_or_file = _resolve_safetensors_index(model_path, revision=revision, token=token)
    params = dict(model.named_parameters())
    config = getattr(model, "config", None)
    selected: dict[Path, list[tuple[str, str]]] = {}
    unexpected: list[str] = []

    for remote_name, shard_path in _iter_safetensor_keys(index_or_file):
        local_name = rewrite_step3p5_mtp_weight_name(config, remote_name)
        if local_name is None:
            continue
        if local_name not in params:
            unexpected.append(f"{remote_name} -> {local_name}")
            continue
        selected.setdefault(shard_path, []).append((remote_name, local_name))

    loaded: set[str] = set()
    for shard_path, names in selected.items():
        resolved_path = _resolve_shard_path(model_path, shard_path, revision=revision, token=token)
        with safe_open(str(resolved_path), framework="pt", device="cpu") as handle:
            for remote_name, local_name in names:
                tensor = handle.get_tensor(remote_name)
                param = params[local_name]
                if tuple(tensor.shape) != tuple(param.shape):
                    raise RuntimeError(
                        f"MTP parameter shape mismatch for {local_name}: "
                        f"checkpoint {tuple(tensor.shape)} vs model {tuple(param.shape)}"
                    )
                with torch.no_grad():
                    param.copy_(tensor.to(device=param.device, dtype=param.dtype))
                loaded.add(local_name)

    expected = _mtp_parameter_names(model)
    missing = expected - loaded
    if strict and missing:
        example = sorted(missing)[0]
        raise RuntimeError(
            f"Missing Step3p5 MTP checkpoint parameters; {example} was not loaded."
        )

    return Step3p5MTPWeightLoadReport(
        loaded=tuple(sorted(loaded)),
        missing=tuple(sorted(missing)),
        unexpected=tuple(sorted(unexpected)),
    )


class Step3p5MTPSharedHead(nn.Module):
    def __init__(self, config: Any, norm_cls: type[nn.Module]):
        super().__init__()
        self.norm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


class Step3p5MTPPredictorLayer(nn.Module):
    def __init__(
        self,
        config: Any,
        *,
        layer_idx: int,
        decoder_layer_cls: type[nn.Module],
        norm_cls: type[nn.Module],
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.enorm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.shared_head = Step3p5MTPSharedHead(config, norm_cls)
        self.mtp_block = decoder_layer_cls(config, layer_idx)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None,
        previous_hidden_states: torch.Tensor,
        embed_tokens: nn.Embedding,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = embed_tokens(input_ids.to(embed_tokens.weight.device))
        previous_hidden_states = previous_hidden_states.to(device=inputs_embeds.device)
        hidden_states = self.eh_proj(
            torch.cat(
                [self.enorm(inputs_embeds), self.hnorm(previous_hidden_states)],
                dim=-1,
            )
        )
        if positions is not None:
            positions = positions.to(device=hidden_states.device)
        causal_mask = self._build_attention_mask(hidden_states, positions, attention_mask)
        return self.mtp_block(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=positions,
            use_cache=False,
        )

    def _build_attention_mask(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
    ) -> Any:
        module = import_module(self.mtp_block.__class__.__module__)
        create_causal_mask = getattr(module, "create_causal_mask", None)
        if create_causal_mask is None:
            return attention_mask

        if positions is None:
            positions = torch.arange(
                hidden_states.shape[1],
                device=hidden_states.device,
                dtype=torch.long,
            ).unsqueeze(0)
        cache_position = positions[0] if positions.dim() == 2 else positions
        mask_kwargs = {
            "config": self.config,
            "input_embeds": hidden_states,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": None,
            "position_ids": positions,
        }
        attention_type = getattr(self.mtp_block, "attention_type", "full_attention")
        if attention_type == "sliding_attention":
            create_sliding_mask = getattr(module, "create_sliding_window_causal_mask", None)
            if create_sliding_mask is not None:
                return create_sliding_mask(**mask_kwargs)
        return create_causal_mask(**mask_kwargs)


class Step3p5ForMTPTraining(nn.Module):
    """HF Step3p5 wrapper that exposes train-time MTP heads for Ditty losses."""

    base_model_prefix = "base_model"

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.num_mtp_heads = int(getattr(self.config, "num_nextn_predict_layers", 0) or 0)
        if self.num_mtp_heads < 1:
            raise ValueError("Step3p5ForMTPTraining requires num_nextn_predict_layers > 0.")

        decoder = self.get_decoder()
        decoder_layer_cls = type(decoder.layers[0])
        norm_cls = type(decoder.norm)
        self.mtp_embed_tokens = nn.Embedding(
            self.config.vocab_size,
            self.config.hidden_size,
            getattr(self.config, "pad_token_id", None),
        )
        start = int(self.config.num_hidden_layers)
        self.mtp_layers = nn.ModuleDict(
            {
                str(layer_idx): Step3p5MTPPredictorLayer(
                    _step3p5_mtp_config_view(self.config),
                    layer_idx=layer_idx,
                    decoder_layer_cls=decoder_layer_cls,
                    norm_cls=norm_cls,
                )
                for layer_idx in range(start, start + self.num_mtp_heads)
            }
        )
        self._mtp_load_report: Step3p5MTPWeightLoadReport | None = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *args,
        load_mtp_weights: bool = True,
        mtp_strict: bool = True,
        **kwargs,
    ) -> "Step3p5ForMTPTraining":
        kwargs = dict(kwargs)
        revision = kwargs.get("revision")
        token = kwargs.get("token")
        kwargs["config"] = kwargs.get("config") or _load_step3p5_config(
            pretrained_model_name_or_path,
            kwargs,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            *args,
            **kwargs,
        )
        model = cls(base_model)
        if load_mtp_weights:
            model._mtp_load_report = load_step3p5_mtp_weights(
                model,
                pretrained_model_name_or_path,
                strict=mtp_strict,
                revision=revision,
                token=token,
            )
        return model

    @property
    def mtp_load_report(self) -> Step3p5MTPWeightLoadReport | None:
        return self._mtp_load_report

    def forward(self, input_ids: torch.Tensor | None = None, **kwargs):
        kwargs.pop("labels", None)
        kwargs.pop("logits_to_keep", None)
        kwargs["return_dict"] = kwargs.get("return_dict", True)
        return self.get_decoder()(input_ids=input_ids, **kwargs)

    def get_decoder(self):
        if hasattr(self.base_model, "model"):
            return self.base_model.model
        if hasattr(self.base_model, "get_decoder"):
            try:
                decoder = self.base_model.get_decoder()
            except (AttributeError, NotImplementedError):
                decoder = None
            if decoder is not None:
                return decoder
        raise RuntimeError(f"Could not resolve decoder for {type(self.base_model).__name__}.")

    def get_input_embeddings(self):
        return self.get_decoder().embed_tokens

    def get_output_embeddings(self):
        return self.base_model.lm_head

    @property
    def lm_head(self):
        return self.base_model.lm_head

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def forward_mtp_step(
        self,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None,
        previous_hidden_states: torch.Tensor,
        step_idx: int,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layer_idx = int(self.config.num_hidden_layers) + int(step_idx)
        layer = self.mtp_layers[str(layer_idx)]
        return layer(
            input_ids=input_ids,
            positions=positions,
            previous_hidden_states=previous_hidden_states,
            embed_tokens=self.mtp_embed_tokens,
            attention_mask=attention_mask,
        )

    def prepare_mtp_hidden_for_output(self, hidden_states: torch.Tensor, *, step_idx: int):
        layer_idx = int(self.config.num_hidden_layers) + int(step_idx)
        return self.mtp_layers[str(layer_idx)].shared_head(hidden_states)

    def get_mtp_output_embeddings(self, step_idx: int):
        layer_idx = int(self.config.num_hidden_layers) + int(step_idx)
        return self.mtp_layers[str(layer_idx)].shared_head.output

    def clone_mtp_step(
        self,
        src_step: int = 0,
        dst_steps: Iterable[int] = (1, 2),
        *,
        overwrite: bool = False,
    ) -> None:
        src_layer_idx = int(self.config.num_hidden_layers) + int(src_step)
        src_layer = self.mtp_layers[str(src_layer_idx)]
        src_state = src_layer.state_dict()
        for dst_step in dst_steps:
            dst_layer_idx = int(self.config.num_hidden_layers) + int(dst_step)
            dst_layer = self.mtp_layers[str(dst_layer_idx)]
            if not overwrite:
                dst_state = dst_layer.state_dict()
                if any(not torch.equal(dst_state[name], src_state[name]) for name in src_state):
                    raise RuntimeError(
                        f"MTP step {dst_step} differs from source step {src_step}; "
                        "pass overwrite=True to replace it."
                    )
            dst_layer.load_state_dict(src_state)

    def load_mtp_weights(
        self,
        model_path: str,
        *,
        strict: bool = True,
        revision: str | None = None,
        token: str | None = None,
    ) -> Step3p5MTPWeightLoadReport:
        self._mtp_load_report = load_step3p5_mtp_weights(
            self,
            model_path,
            strict=strict,
            revision=revision,
            token=token,
        )
        return self._mtp_load_report

    def gradient_checkpointing_enable(self, *args, **kwargs) -> None:
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable(*args, **kwargs)
        for layer in self.mtp_layers.values():
            block = getattr(layer, "mtp_block", None)
            if hasattr(block, "gradient_checkpointing"):
                block.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            self.base_model.gradient_checkpointing_disable()
        for layer in self.mtp_layers.values():
            block = getattr(layer, "mtp_block", None)
            if hasattr(block, "gradient_checkpointing"):
                block.gradient_checkpointing = False
