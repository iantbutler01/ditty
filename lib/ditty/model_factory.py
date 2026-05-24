import os
import types
from dataclasses import dataclass, field
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
