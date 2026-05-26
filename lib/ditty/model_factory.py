import os
import types
from dataclasses import dataclass, field, replace
from importlib import import_module
from logging import getLogger
from typing import Optional, List, Type, Dict, Any, Union

import torch
import torch.nn as nn
import safetensors
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import hub, SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME
from accelerate import init_empty_weights
from fastcore.parallel import parallel
from tqdm.auto import tqdm

logger = getLogger("ditty_model_factory")


class ModelTransform:
    """Transform applied to a model after loading.

    Use for operations like wrapping models, freezing layers, etc.
    """
    def transform(self, model: nn.Module) -> nn.Module:
        raise NotImplementedError


class ChainedModelTransform(ModelTransform):
    """Apply multiple model transforms in order."""

    def __init__(self, *transforms: Optional[ModelTransform]):
        self.transforms = [transform for transform in transforms if transform is not None]

    def transform(self, model: nn.Module) -> nn.Module:
        for transform in self.transforms:
            model = transform.transform(model)
        return model


class CausalLMBackboneTransform(ModelTransform):
    """Patch HF causal LM forward to return backbone outputs instead of dense vocab logits."""

    def transform(self, model: nn.Module) -> nn.Module:
        original_forward = model.forward

        def backbone_forward(bound_model, *args, **kwargs):
            decoder = self._resolve_decoder(bound_model)
            filtered_kwargs = dict(kwargs)
            filtered_kwargs.pop("labels", None)
            filtered_kwargs.pop("logits_to_keep", None)
            filtered_kwargs["return_dict"] = filtered_kwargs.get("return_dict", True)
            return decoder(*args, **filtered_kwargs)

        model._ditty_original_forward = original_forward
        model.forward = types.MethodType(backbone_forward, model)
        return model

    @staticmethod
    def _resolve_decoder(model: nn.Module) -> nn.Module:
        if hasattr(model, "get_decoder"):
            try:
                decoder = model.get_decoder()
                if decoder is not None:
                    return decoder
            except (AttributeError, NotImplementedError):
                pass
        real_model = getattr(model, "_orig_mod", model)
        base_model_prefix = getattr(real_model, "base_model_prefix", None)
        if base_model_prefix and hasattr(real_model, base_model_prefix):
            return getattr(real_model, base_model_prefix)
        if hasattr(real_model, "base_model"):
            return real_model.base_model
        if hasattr(real_model, "model"):
            return real_model.model
        raise RuntimeError(
            f"Could not resolve decoder/backbone module for model {type(real_model).__name__}."
        )


@dataclass
class Float8TrainingTransform(ModelTransform):
    """Convert eligible dense and packed MoE matmuls to TorchAO float8 before FSDP2."""

    recipe_name: str = "tensorwise"
    enable_fsdp_float8_all_gather: bool = True
    pad_inner_dim: bool = True
    force_recompute_fp8_weight_in_bwd: bool = True
    skip_fqn_fragments: tuple[str, ...] = ()

    def transform(self, model: nn.Module) -> nn.Module:
        try:
            from torchao.float8 import Float8LinearConfig, convert_to_float8_training
            from torchao.float8.float8_linear import (
                GemmInputRole,
                LinearMMConfig,
                ScaledMMConfig,
                WeightWithDynamicFloat8CastTensor,
                matmul_with_hp_or_float8_args,
            )
            from torchao.float8.float8_training_tensor import Float8TrainingTensor
        except ImportError as exc:
            raise ModuleNotFoundError(
                "Float8TrainingTransform requires torchao.float8. Install a torchao "
                "build compatible with the active PyTorch/CUDA stack."
            ) from exc

        config = replace(
            Float8LinearConfig.from_recipe_name(self.recipe_name),
            enable_fsdp_float8_all_gather=self.enable_fsdp_float8_all_gather,
            pad_inner_dim=self.pad_inner_dim,
            force_recompute_fp8_weight_in_bwd=self.force_recompute_fp8_weight_in_bwd,
        )
        self._install_raw_float8_storage_support(
            weight_wrapper_cls=WeightWithDynamicFloat8CastTensor,
            float8_training_tensor_cls=Float8TrainingTensor,
            gemm_input_role=GemmInputRole,
        )

        converted = 0
        skipped_shape = 0
        skipped_shape_examples: list[str] = []

        def module_filter_fn(module: nn.Module, fqn: str) -> bool:
            nonlocal converted, skipped_shape
            if not isinstance(module, nn.Linear):
                return True
            if any(fragment and fragment in fqn for fragment in self.skip_fqn_fragments):
                return False
            in_features = int(getattr(module, "in_features", 0))
            out_features = int(getattr(module, "out_features", 0))
            shape_supported = out_features % 16 == 0
            if not self.pad_inner_dim:
                shape_supported = shape_supported and in_features % 16 == 0
            if not shape_supported:
                skipped_shape += 1
                if len(skipped_shape_examples) < 8:
                    skipped_shape_examples.append(f"{fqn}({in_features}->{out_features})")
                return False
            converted += 1
            return True

        logger.info(
            "Applying TorchAO float8 training transform: recipe=%s "
            "fsdp_float8_all_gather=%s pad_inner_dim=%s "
            "force_recompute_fp8_weight_in_bwd=%s skip=%s",
            self.recipe_name,
            self.enable_fsdp_float8_all_gather,
            self.pad_inner_dim,
            self.force_recompute_fp8_weight_in_bwd,
            ",".join(self.skip_fqn_fragments),
        )
        model = convert_to_float8_training(
            model,
            module_filter_fn=module_filter_fn,
            config=config,
        )
        logger.info(
            "Converted %s Linear module(s) to TorchAO Float8Linear; skipped %s shape-ineligible "
            "Linear module(s)%s.",
            converted,
            skipped_shape,
            f": {', '.join(skipped_shape_examples)}" if skipped_shape_examples else "",
        )
        wrapped_experts = self._wrap_packed_moe_experts(
            model,
            config=config,
            linear_mm_config=LinearMMConfig(
                ScaledMMConfig(config.emulate, config.gemm_config_output.use_fast_accum, False, config.pad_inner_dim),
                ScaledMMConfig(config.emulate, config.gemm_config_grad_input.use_fast_accum, False, config.pad_inner_dim),
                ScaledMMConfig(config.emulate, config.gemm_config_grad_weight.use_fast_accum, False, config.pad_inner_dim),
            ),
            weight_wrapper_cls=WeightWithDynamicFloat8CastTensor,
            matmul_fn=matmul_with_hp_or_float8_args,
        )
        if wrapped_experts:
            logger.info("Wrapped %s packed MoE expert module(s) for TorchAO float8 FSDP all-gather.", wrapped_experts)
        model._ditty_float8_training_enabled = True
        return model

    @staticmethod
    def _install_raw_float8_storage_support(
        *,
        weight_wrapper_cls: Any,
        float8_training_tensor_cls: Any,
        gemm_input_role: Any,
    ) -> None:
        if getattr(weight_wrapper_cls, "_ditty_raw_float8_storage_patch", False):
            return

        original_pre_all_gather = weight_wrapper_cls.fsdp_pre_all_gather
        original_post_all_gather = weight_wrapper_cls.fsdp_post_all_gather
        float8_dtypes = {torch.float8_e4m3fn, torch.float8_e5m2}

        def fsdp_pre_all_gather(self, mesh):
            tensor = getattr(self, "_tensor", None)
            if isinstance(tensor, torch.Tensor) and tensor.dtype in float8_dtypes:
                scale = torch.ones((), device=tensor.device, dtype=torch.float32)
                return (tensor,), (scale,)
            return original_pre_all_gather(self, mesh)

        def fsdp_post_all_gather(self, all_gather_outputs, metadata, param_dtype, *, out=None):
            tensor = getattr(self, "_tensor", None)
            if not (isinstance(tensor, torch.Tensor) and tensor.dtype in float8_dtypes):
                return original_post_all_gather(self, all_gather_outputs, metadata, param_dtype, out=out)

            (data,) = all_gather_outputs
            (scale,) = metadata
            if out is not None:
                from torch.distributed._tensor import DTensor

                if isinstance(out, float8_training_tensor_cls):
                    out._data = data
                    out._scale = scale
                elif isinstance(out, DTensor) and isinstance(out._local_tensor, float8_training_tensor_cls):
                    out._local_tensor._data = data
                    out._local_tensor._scale = scale
                else:
                    raise RuntimeError(
                        "raw-FP8 FSDP post-all-gather expected Float8TrainingTensor output, "
                        f"got {type(out).__name__}"
                    )
                return

            return float8_training_tensor_cls(
                data,
                scale,
                param_dtype,
                self._linear_mm_config,
                gemm_input_role.WEIGHT,
            ), (data,)

        weight_wrapper_cls.fsdp_pre_all_gather = fsdp_pre_all_gather
        weight_wrapper_cls.fsdp_post_all_gather = fsdp_post_all_gather
        weight_wrapper_cls._ditty_raw_float8_storage_patch = True

    @staticmethod
    def _wrap_packed_moe_experts(
        model: nn.Module,
        *,
        config: Any,
        linear_mm_config: Any,
        weight_wrapper_cls: Any,
        matmul_fn: Any,
    ) -> int:
        wrapped = 0
        for module in model.modules():
            gate_up_proj = getattr(module, "gate_up_proj", None)
            down_proj = getattr(module, "down_proj", None)
            if not isinstance(gate_up_proj, nn.Parameter) or not isinstance(down_proj, nn.Parameter):
                continue
            if gate_up_proj.dim() != 3 or down_proj.dim() != 3:
                continue
            if getattr(module, "_ditty_float8_packed_moe_wrapped", False):
                continue

            module.gate_up_proj = nn.Parameter(
                weight_wrapper_cls(
                    gate_up_proj,
                    linear_mm_config,
                    config.cast_config_weight.target_dtype,
                ),
                requires_grad=gate_up_proj.requires_grad,
            )
            module.down_proj = nn.Parameter(
                weight_wrapper_cls(
                    down_proj,
                    linear_mm_config,
                    config.cast_config_weight.target_dtype,
                ),
                requires_grad=down_proj.requires_grad,
            )
            module._ditty_float8_expert_config = config
            module._ditty_float8_expert_linear_mm_config = linear_mm_config
            module._ditty_float8_expert_matmul = matmul_fn
            module._ditty_float8_packed_moe_wrapped = True
            module.forward = types.MethodType(_float8_packed_moe_experts_forward, module)
            wrapped += 1
        return wrapped


def _float8_packed_moe_experts_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    final_hidden_states = torch.zeros_like(hidden_states)
    num_experts = int(getattr(self, "num_experts"))
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    matmul = self._ditty_float8_expert_matmul
    mm_config = self._ditty_float8_expert_linear_mm_config
    config = self._ditty_float8_expert_config

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        expert_i = int(expert_idx.item())
        if expert_i == num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate_up_weight = self.gate_up_proj[expert_i : expert_i + 1].reshape(
            self.gate_up_proj.shape[1],
            self.gate_up_proj.shape[2],
        )
        gate_up = matmul.apply(current_state, gate_up_weight.t(), mm_config, config)
        gate, up = gate_up.chunk(2, dim=-1)
        current_hidden_states = self.act_fn(gate) * up
        down_weight = self.down_proj[expert_i : expert_i + 1].reshape(
            self.down_proj.shape[1],
            self.down_proj.shape[2],
        )
        current_hidden_states = matmul.apply(current_hidden_states, down_weight.t(), mm_config, config)
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

    return final_hidden_states


@dataclass
class FSDPConfig:
    enabled: bool = False
    transformer_layers: List[Type[nn.Module]] = field(default_factory=list)
    param_dtype: Optional[torch.dtype] = None  # e.g. torch.bfloat16
    reduce_dtype: Optional[torch.dtype] = None  # None = match param_dtype, torch.float32 for accuracy
    original_param_dtype: Optional[torch.dtype] = None  # dtype for sharded optimizer params after fully_shard
    reshard_after_forward: bool = True  # True = FULL_SHARD, False = SHARD_GRAD_OP
    cpu_offload: bool = False
    cpu_offload_pin_memory: bool = True


@dataclass
class QuantConfig:
    enabled: bool = False
    bits: int = 4  # 4 or 8
    use_double_quant: bool = True
    quant_type: str = "nf4"
    compute_dtype: torch.dtype = torch.bfloat16
    quant_storage: torch.dtype = torch.bfloat16
    use_dora: bool = False


@dataclass
class PeftConfig:
    enabled: bool = False
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"])
    use_dora: bool = False


@dataclass
class ModelConfig:
    model_path: str
    model_class_name: Optional[str] = None
    text_only: bool = False
    load_kwargs: Dict[str, Any] = field(default_factory=dict)


class ModelFactory:
    """
    Factory for loading models and preparing them for distributed training.

    Handles:
    - Loading from HuggingFace Hub
    - Loading from local checkpoints
    - Wrapping existing model instances
    - FSDP2 sharding via fully_shard()
    - QLoRA 4bit/8bit quantization
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        model_path: Optional[str] = None,
        model_class: Optional[Type[nn.Module]] = None,
        fsdp_config: Optional[Union[FSDPConfig, Dict[str, Any]]] = None,
        quant_config: Optional[Union[QuantConfig, Dict[str, Any]]] = None,
        peft_config: Optional[Union[PeftConfig, Dict[str, Any]]] = None,
        load_kwargs: Optional[Dict[str, Any]] = None,
        contract: str = "",
        model_transform: Optional[ModelTransform] = None,
        use_compile: bool = False,
        compile_mode: str = "default",
    ):
        self._model = model
        self._model_path = model_path
        self._model_class = model_class
        self._load_kwargs = load_kwargs or {}
        self.contract = contract
        self._model_transform = model_transform
        self.use_compile = use_compile
        self.compile_mode = compile_mode
        # Injected by Pipeline when resuming from checkpoint
        self._checkpoint_state: Optional[Dict[str, Any]] = None

        if isinstance(fsdp_config, dict):
            self.fsdp_config = FSDPConfig(**fsdp_config)
        else:
            self.fsdp_config = fsdp_config or FSDPConfig()

        if isinstance(quant_config, dict):
            self.quant_config = QuantConfig(**quant_config)
        else:
            self.quant_config = quant_config or QuantConfig()

        if isinstance(peft_config, dict):
            self.peft_config = PeftConfig(**peft_config)
        else:
            self.peft_config = peft_config or PeftConfig()

        if model is None and model_path is None:
            raise ValueError("Must provide either model or model_path")

    @classmethod
    def from_huggingface(
        cls,
        model_path: Union[str, ModelConfig, Dict[str, Any]],
        fsdp_config: Optional[Union[FSDPConfig, Dict[str, Any]]] = None,
        quant_config: Optional[Union[QuantConfig, Dict[str, Any]]] = None,
        peft_config: Optional[Union[PeftConfig, Dict[str, Any]]] = None,
        model_transform: Optional[ModelTransform] = None,
        use_compile: bool = False,
        compile_mode: str = "default",
        **load_kwargs,
    ) -> "ModelFactory":
        if isinstance(model_path, dict):
            model_config = ModelConfig(**model_path)
        elif isinstance(model_path, ModelConfig):
            model_config = model_path
        else:
            model_config = ModelConfig(model_path=model_path)

        merged_load_kwargs = dict(model_config.load_kwargs)
        merged_load_kwargs.update(load_kwargs)

        return cls(
            model_path=model_config.model_path,
            model_class=cls._resolve_huggingface_model_class(model_config, merged_load_kwargs),
            fsdp_config=fsdp_config,
            quant_config=quant_config,
            peft_config=peft_config,
            load_kwargs=merged_load_kwargs,
            model_transform=model_transform,
            use_compile=use_compile,
            compile_mode=compile_mode,
        )

    @classmethod
    def _resolve_huggingface_model_class(
        cls,
        model_config: ModelConfig,
        load_kwargs: Dict[str, Any],
    ) -> Type[nn.Module]:
        model_class_name = model_config.model_class_name
        if model_class_name is None and model_config.text_only:
            model_class_name = cls._infer_text_only_model_class_name(
                model_path=model_config.model_path,
                load_kwargs=load_kwargs,
            )
        if model_class_name is None:
            return AutoModelForCausalLM
        return cls._lookup_huggingface_model_class(
            model_path=model_config.model_path,
            model_class_name=model_class_name,
            load_kwargs=load_kwargs,
        )

    @classmethod
    def _infer_text_only_model_class_name(
        cls,
        *,
        model_path: str,
        load_kwargs: Dict[str, Any],
    ) -> str:
        cfg = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=load_kwargs.get("trust_remote_code", False),
        )
        architectures = [str(name) for name in (getattr(cfg, "architectures", None) or [])]
        candidates: List[str] = []
        for architecture in architectures:
            if architecture.endswith("ForConditionalGeneration"):
                candidates.append(
                    architecture.removesuffix("ForConditionalGeneration") + "ForCausalLM"
                )
            elif architecture.endswith("ForCausalLM"):
                candidates.append(architecture)

        if not candidates:
            model_type = str(getattr(cfg, "model_type", "")).replace("-", "_")
            class_stem = "".join(part.capitalize() for part in model_type.split("_") if part)
            if class_stem:
                candidates.append(f"{class_stem}ForCausalLM")

        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                cls._lookup_huggingface_model_class(
                    model_path=model_path,
                    model_class_name=candidate,
                    load_kwargs=load_kwargs,
                )
            except LookupError:
                continue
            return candidate

        raise RuntimeError(
            f"Could not resolve a text-only HuggingFace model class for {model_path!r}. "
            f"Tried candidates: {candidates or ['<none>']}."
        )

    @classmethod
    def _lookup_huggingface_model_class(
        cls,
        *,
        model_path: str,
        model_class_name: str,
        load_kwargs: Dict[str, Any],
    ) -> Type[nn.Module]:
        if "." in model_class_name:
            module_name, _, class_name = model_class_name.rpartition(".")
            module = import_module(module_name)
            resolved = getattr(module, class_name, None)
            if resolved is None:
                raise LookupError(
                    f"Could not resolve HuggingFace model class {model_class_name!r}."
                )
            return resolved

        cfg = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=load_kwargs.get("trust_remote_code", False),
        )
        model_type = str(getattr(cfg, "model_type", "")).replace("-", "_")
        candidate_modules = [
            f"transformers.models.{model_type}.modeling_{model_type}",
            "transformers",
        ]
        for module_name in candidate_modules:
            try:
                module = import_module(module_name)
            except ImportError:
                continue
            resolved = getattr(module, model_class_name, None)
            if resolved is not None:
                return resolved

        raise LookupError(
            f"Could not resolve HuggingFace model class {model_class_name!r} "
            f"for model {model_path!r}."
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_class: Type[nn.Module],
        fsdp_config: Optional[Union[FSDPConfig, Dict[str, Any]]] = None,
        model_transform: Optional[ModelTransform] = None,
        use_compile: bool = False,
        compile_mode: str = "default",
        **model_kwargs,
    ) -> "ModelFactory":
        return cls(
            model_path=checkpoint_path,
            model_class=model_class,
            fsdp_config=fsdp_config,
            load_kwargs=model_kwargs,
            model_transform=model_transform,
            use_compile=use_compile,
            compile_mode=compile_mode,
        )

    @classmethod
    def from_instance(
        cls,
        model: nn.Module,
        fsdp_config: Optional[Union[FSDPConfig, Dict[str, Any]]] = None,
        use_compile: bool = False,
        compile_mode: str = "default",
    ) -> "ModelFactory":
        return cls(
            model=model,
            fsdp_config=fsdp_config,
            use_compile=use_compile,
            compile_mode=compile_mode,
        )

    def _replace_linear(self, model: nn.Module, skip_modules: List[str] = None):
        from bitsandbytes.nn import Linear4bit

        skip_modules = skip_modules or ["lm_head"]
        for name, module in model.named_children():
            if name in skip_modules:
                continue
            if len(list(module.children())) > 0:
                self._replace_linear(module, skip_modules)
            if isinstance(module, nn.Linear):
                model._modules[name] = Linear4bit(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    compute_dtype=self.quant_config.compute_dtype,
                    quant_type=self.quant_config.quant_type,
                    quant_storage=self.quant_config.quant_storage,
                )
        return model

    def _n_loading_workers(self, param_count: float):
        devprops = torch.cuda.get_device_properties(torch.cuda.current_device())
        left = int(os.cpu_count() / torch.cuda.device_count())
        right = int(8 * (devprops.total_memory / 1e9 / 40) * (70 / (param_count / 1e9)))
        return min(left, right)

    def _load_and_quantize(self, module: nn.Module, name: str, value: torch.Tensor,
                           device=None, dtype=None, skip_names=None, to_cpu=False, to_meta=False):
        from bitsandbytes.nn import Params4bit

        skip_names = skip_names or []

        def place_on_device(value):
            if to_meta:
                return value.to(device="meta", dtype=dtype)
            elif to_cpu:
                return value.to(device="cpu", dtype=dtype)
            return value.to(device=device, dtype=dtype)

        if any(skip_name in name for skip_name in skip_names):
            return

        module_key, _, value_key = name.rpartition(".")
        try:
            submodule = module.get_submodule(module_key)
        except AttributeError:
            return

        try:
            param = submodule.get_parameter(value_key)
            if isinstance(param, Params4bit):
                if self.quant_config.use_dora:
                    setattr(submodule, "dora_scale", value.norm(p=2, dim=1).to(dtype=dtype).to("cpu"))
                value = type(param)(value.to(device=device, dtype=dtype).data, **param.__dict__).cuda(device)
                if to_meta:
                    value = type(param)(value.data.to("meta"), **value.__dict__)
                elif to_cpu:
                    value = type(param)(value.data.to("cpu"), **value.__dict__)
            else:
                value = type(param)(place_on_device(value).data)
        except AttributeError:
            value = place_on_device(value)

        setattr(submodule, value_key, value)

    def _load_quantized_model(self) -> nn.Module:
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        cfg = AutoConfig.from_pretrained(self._model_path, **self._load_kwargs)
        cfg.use_cache = False
        if self._load_kwargs.get("attn_implementation"):
            cfg.attn_implementation = self._load_kwargs["attn_implementation"]

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(cfg)
            model.model = self._replace_linear(model.model)

        model.is_loaded_in_4bit = True

        try:
            idx = hub.cached_file(self._model_path, SAFE_WEIGHTS_INDEX_NAME)
            files, _ = hub.get_checkpoint_shard_files(self._model_path, idx)
        except OSError:
            try:
                files = [hub.cached_file(self._model_path, SAFE_WEIGHTS_NAME)]
            except OSError as e:
                raise e

        def load_and_quantize_parallel(name_param, model, **kwargs):
            name, param = name_param
            self._load_and_quantize(model, name, param, **kwargs)

        param_count = sum(p.numel() for p in model.parameters())
        if local_rank == 0:
            logger.info(f"Total model params: {param_count}")

        n_workers = self._n_loading_workers(param_count)
        if rank == 0:
            logger.info(f"Using n_workers: {n_workers} for loading")

        for filename in tqdm(files, desc="Loading & Quantizing", disable=rank != 0):
            weights = safetensors.torch.load_file(filename)
            parallel(
                load_and_quantize_parallel,
                iter(weights.items()),
                n_workers=n_workers,
                threadpool=True,
                model=model,
                dtype=self.quant_config.compute_dtype,
                device=torch.cuda.current_device(),
                skip_names=[],
                to_cpu=(local_rank == 0),
                to_meta=(local_rank != 0),
            )

        torch.cuda.empty_cache()
        return model

    def _load_model(self) -> nn.Module:
        if self._model is not None:
            model = self._model
            # Apply checkpoint state if injected (for resuming training)
            if self._checkpoint_state is not None:
                logger.info("Loading model weights from checkpoint state")
                model.load_state_dict(self._checkpoint_state)
            return model

        if self.quant_config.enabled and self.quant_config.bits == 4 and self.fsdp_config.enabled:
            logger.info(f"Loading 4bit quantized model: {self._model_path}")
            return self._load_quantized_model()

        if (
            self._model_path is not None
            and not self._model_path.endswith(".pt")
            and not self._model_path.endswith(".pth")
            and self._model_class is not None
            and hasattr(self._model_class, "from_pretrained")
        ):
            logger.info(f"Loading model from HuggingFace: {self._model_path}")
            bnb_config = None
            if self.quant_config.enabled:
                if self.quant_config.bits == 4:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=self.quant_config.use_double_quant,
                        bnb_4bit_quant_type=self.quant_config.quant_type,
                        bnb_4bit_quant_storage=self.quant_config.quant_storage,
                        bnb_4bit_compute_dtype=self.quant_config.compute_dtype,
                    )
                elif self.quant_config.bits == 8:
                    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

            return self._model_class.from_pretrained(
                self._model_path,
                quantization_config=bnb_config,
                **self._load_kwargs,
            )

        # For custom model classes, create model then optionally load checkpoint
        if self._model_path is None or self._model_path.endswith(".pt") or self._model_path.endswith(".pth"):
            # Determine which state dict to use
            if self._checkpoint_state is not None:
                # Use injected checkpoint state (from Pipeline resume)
                logger.info("Loading model weights from checkpoint state")
                model = self._model_class(**self._load_kwargs)
                model.load_state_dict(self._checkpoint_state)
                return model
            elif self._model_path is not None:
                # Load from explicit checkpoint path
                logger.info(f"Loading model from checkpoint: {self._model_path}")
                state_dict = torch.load(self._model_path, map_location="cpu", weights_only=False)
                if "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                model = self._model_class(**self._load_kwargs)
                model.load_state_dict(state_dict)
                return model
            else:
                # Fresh model, no weights to load
                model = self._model_class(**self._load_kwargs)
                return model

        raise ValueError(f"Cannot load model from {self._model_path}")

    def _apply_fsdp(self, model: nn.Module) -> nn.Module:
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        logger.info(f"Applying FSDP2 sharding (rank {rank}, local_rank {local_rank})")

        torch.cuda.set_device(local_rank)
        model = model.to("cpu")

        mp_policy = None
        if self.fsdp_config.param_dtype is not None:
            mp_policy = MixedPrecisionPolicy(
                param_dtype=self.fsdp_config.param_dtype,
                reduce_dtype=self.fsdp_config.reduce_dtype,
            )

        fsdp_kwargs = {
            "reshard_after_forward": self.fsdp_config.reshard_after_forward,
        }
        if mp_policy:
            fsdp_kwargs["mp_policy"] = mp_policy
        if self.fsdp_config.cpu_offload:
            fsdp_kwargs["offload_policy"] = CPUOffloadPolicy(
                pin_memory=self.fsdp_config.cpu_offload_pin_memory,
            )

        for module in model.modules():
            if any(
                isinstance(module, layer_cls)
                for layer_cls in self.fsdp_config.transformer_layers
            ):
                fully_shard(module, **fsdp_kwargs)

        fully_shard(model, **fsdp_kwargs)
        if self.fsdp_config.original_param_dtype is not None:
            logger.info(
                "Casting FSDP2 sharded original parameters to %s",
                self.fsdp_config.original_param_dtype,
            )
            model.to(self.fsdp_config.original_param_dtype)
        return model

    def _setup_quantized_meta_for_peft(self, model: nn.Module):
        from bitsandbytes.nn import Params4bit

        def temp_to_method(self, *args, **kwargs):
            return self
        for param in model.parameters():
            if isinstance(param, Params4bit):
                param.quant_state._orig_to = param.quant_state.to
                param.quant_state.to = types.MethodType(temp_to_method, param.quant_state)

    def _setup_quantized_peft_meta_for_training(self, model: nn.Module):
        from bitsandbytes.nn import Params4bit

        for param in model.parameters():
            if isinstance(param, Params4bit) and hasattr(param.quant_state, "_orig_to"):
                param.quant_state.to = param.quant_state._orig_to
                param.quant_state._orig_to = None

    def _apply_peft(self, model: nn.Module) -> nn.Module:
        from peft import TaskType, LoraConfig, get_peft_model

        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.peft_config.target_modules,
            inference_mode=False,
            r=self.peft_config.r,
            lora_alpha=self.peft_config.lora_alpha,
            lora_dropout=self.peft_config.lora_dropout,
            bias="none",
            use_dora=self.peft_config.use_dora,
        )

        model.enable_input_require_grads()

        if self.quant_config.enabled and local_rank != 0:
            self._setup_quantized_meta_for_peft(model)

        model = get_peft_model(model, lora_config)

        if self.quant_config.enabled:
            self._setup_quantized_peft_meta_for_training(model)

        return model

    def build(self) -> nn.Module:
        model = self._load_model()

        if self._model_transform is not None:
            model = self._model_transform.transform(model)

        if self.peft_config.enabled:
            model = self._apply_peft(model)

        if self.use_compile:
            logger.info(f"Compiling model with torch.compile(mode={self.compile_mode})")
            model = torch.compile(model, mode=self.compile_mode)

        if not self.fsdp_config.enabled:
            logger.info("FSDP disabled, returning unwrapped model")
            return model

        return self._apply_fsdp(model)


class TokenizerFactory:
    def __init__(
        self,
        tokenizer_path: str,
        pad_token: Optional[str] = None,
        token: Optional[str] = None,
        **load_kwargs,
    ):
        self._tokenizer_path = tokenizer_path
        self._pad_token = pad_token
        self._token = token or os.environ.get("HF_TOKEN")
        self._load_kwargs = load_kwargs

    @classmethod
    def from_pretrained(cls, tokenizer_path: str, **kwargs) -> "TokenizerFactory":
        return cls(tokenizer_path=tokenizer_path, **kwargs)

    def build(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self._tokenizer_path,
            token=self._token,
            **self._load_kwargs,
        )
        if tokenizer.pad_token_id is None:
            if self._pad_token:
                tokenizer.pad_token = self._pad_token
            else:
                logger.warning("Tokenizer did not have a pad_token_id, set to EOS.")
                tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer
