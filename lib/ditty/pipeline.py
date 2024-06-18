from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
)
from bitsandbytes.nn import Linear4bit, Params4bit
from transformers.utils import hub, SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME

import logging
import os
import functools
import types
from logging import getLogger
from typing import Optional, List, Dict, Any
import bitsandbytes as bnb
from accelerate import Accelerator, infer_auto_device_map, init_empty_weights
from accelerate.utils import ProjectConfiguration
from fastcore.parallel import parallel

import torch
from tqdm.auto import tqdm
import safetensors
from torch.utils.data import DataLoader
from .trainer import Trainer
from .data import Data
from .hf_utils import push_to_hub
from peft import prepare_model_for_kbit_training 
from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy
from torch.distributed.fsdp import MixedPrecision, FullyShardedDataParallel as FSDP
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LLAMA_ATTENTION_CLASSES, LlamaMLP
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MISTRAL_ATTENTION_CLASSES, MistralMLP


logging.basicConfig(level=logging.INFO)

logger = getLogger("ditty_pipeline")


class Pipeline:
    def __init__(
        self,
        output_dir: str = "./output",
        dataset_namespace: str = "code_search_net",
        dataset_path: str = "python",
        model_name_or_path: str = "theblackcat102/pythia-3b-deduped-sft-r1",
        hf_hub_token: Optional[str] = None,
        hf_hub_model_id: Optional[str] = None,
        fp16: bool = True,
        l8bit: bool = True,
        l4bit: bool = False,
        seed: int = None,
        batch_size: int = 4,
        grad_accum: int = 4,
        push_to_hub: bool = True,
        fp32_cpu_offload: bool = False,
        load_checkpoint: bool = True,
        checkpoint_every: int = 1000,
        gradient_checkpointing: bool = True,
        block_size: int = 2048,
        use_bfloat16: int = False,
        model_load_kwargs: Dict[str, Any] = {"device_map": "auto"},
        accelerator_kwargs: Dict[str, Any] = {},
        use_fsdp: bool = False,
        use_deep_speed: bool = False,
        use_8bit_optim: bool = False,
        use_qdora: bool = False,
        peft_config: bool = None,
        epochs: int = 1,
        max_steps: Optional[int] = None,
        use_flash_attn_2: bool = True,
        model_token: Optional[str] = True,
        output_hub_repo: Optional[str] = None,
        merge_adapters: bool = False,
        private_repo: bool = True
    ):
        self.output_dir = output_dir
        self.dataset_namespace = dataset_namespace
        self.dataset_path = dataset_path
        self.model_name_or_path = model_name_or_path
        self.hf_hub_token = hf_hub_token or os.environ.get("HF_TOKEN")
        self.hf_hub_model_id = hf_hub_model_id
        self.fp16 = fp16
        self.seed = seed
        self.epochs = epochs
        self.max_steps = max_steps
        self.l8bit = l8bit
        self.l4bit = l4bit
        self.push_to_hub = push_to_hub
        self.output_hub_repo = output_hub_repo
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.block_size = block_size
        self.fp32_cpu_offload = fp32_cpu_offload
        self.use_bfloat16 = use_bfloat16
        self.accelerator_kwargs = accelerator_kwargs
        self.use_8bit_optim=use_8bit_optim
        self.use_fsdp=use_fsdp
        self.use_deep_speed = use_deep_speed
        self.peft_config = peft_config
        self.use_qdora = use_qdora
        self.use_flash_attn_2 = use_flash_attn_2
        self.model_token = model_token or os.environ.get("HF_TOKEN")
        self.model_load_kwargs = model_load_kwargs
        self.merge_adapters = merge_adapters
        self.private_repo = private_repo

        if not model_load_kwargs.get("token"):
            model_load_kwargs["token"] = self.model_token

        if self.use_fsdp and self.use_deep_speed:
            raise ValueError("Cannot set both use_fsdp and use_deep_speed to True.")

        if self.l8bit and self.l4bit:
            raise ValueError("Cannot set both l8bit and l4bit to True.")

        if self.push_to_hub and not self.output_hub_repo:
            raise ValueError("Cannot enable push to hub without providing output_hub_repo.")

        if self.l4bit:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_quant_storage=torch.bfloat16,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif self.l8bit:
            self.bnb_config = BitsAndBytesConfig(
                load_in_8bit=l8bit, llm_int8_enable_fp32_cpu_offload=fp32_cpu_offload
            )
        else:
            self.bnb_config = None

        self.checkpoint_every = checkpoint_every
        self.load_checkpoint = load_checkpoint
        self.gradient_checkpointing = gradient_checkpointing

    def dataset(self) -> DataLoader:
        """
        Subclass Pipeline and customize for your own dataset.
        """

        data = Data(
            load_kwargs={"path": self.dataset_namespace, "name": self.dataset_path},
            tokenizer=self.tokenizer,
            seed=self.seed,
            batch_size=self.batch_size,
            grad_accum=self.grad_accum,
        )

        columns = data.dataset.features

        def filter_longer(sample):
            tokens, _attn_mask = self.tokenizer(sample["whole_func_string"])

            return len(tokens) <= self.block_size

        def truncate(sample):
            sample["attention_mask"] = sample["attention_mask"][: self.block_size]
            sample["input_ids"] = sample["input_ids"][: self.block_size]
            return sample

        dataloader = data.prepare(
            [
                ("filter", filter_longer, {}),
                (
                    "map",
                    lambda sample: self.tokenizer(
                        sample["whole_func_string"],
                        padding="max_length",
                        max_length=self.block_size,
                    ),
                    dict(batched=True, remove_columns=columns, num_proc=8),
                ),
                ("map", truncate, dict(num_proc=8)),
            ]
        )
        return dataloader

    def _replace_linear(self, model: torch.nn.Module, linear_replacement: torch.nn.Module, quant_config:dict|None=None, skip_modules:List[str]=["lm_head"], **kwargs):
        """
        Replace linear modules with a new Linear module.
        Parameters:
            model (`torch.nn.Module`):
                Input model or `torch.nn.Module` as the function is run recursively.
            linear_replacement (`torch.nn.Module`):
                The linear module that replaces the old one. Only expects standard arguments.
                If other arguments need to be passed, use a lambda.
            skip_modules (`List[str]`, *optional*, defaults to `lm_head`):
                List of modules names not to convert. Defaults to `lm_head`.
        """
        for name, module in model.named_children():
            if name in skip_modules:
                continue

            if len(list(module.children())) > 0:
                self._replace_linear(module, linear_replacement, quant_config, skip_modules, **kwargs)

            if isinstance(module, torch.nn.Linear):
                if issubclass(linear_replacement, Linear4bit):
                    model._modules[name] = linear_replacement(
                        module.in_features,
                        module.out_features,
                        module.bias is not None,
                        **kwargs
                    )
                else:
                    raise ValueError(f"Unsupported linear replacement: {type(linear_replacement)}")
        return model
    
    def _n_loading_workers(self, quant_method: str, param_count: float):
        devprops = torch.cuda.get_device_properties(torch.cuda.current_device())
        left = int(os.cpu_count()/torch.cuda.device_count())
        right = int((4 if quant_method == "hqq" else 8) * (devprops.total_memory/1e9/40) * (70/(param_count/1e9)))
        return min(left, right)

    # Credit to Answer.ai for this loading and quantizing code. In testing it has been much more efficient when working with limited GPU compute. 
    def _load_and_quantize(self, module: torch.nn.Module, name: str, value: torch.Tensor, device: torch.device=None, dtype: torch.dtype = None, skip_names: list[str] = [], to_cpu: bool = False, to_meta: bool = False, verbose: bool = False, quant_method: str = 'bnb', is_dora: bool = False):
        """
        Loads `value` tensor into submodule of `module`, optionally skipping `skip_names` and converting to `dtype`.

        Quantizes `Params4bit` on `device` then places on "cpu" if to_cpu=True or "meta" if to_meta=True.
        """

        def place_on_device(value):
            if to_meta:
                device = 'meta'
            elif to_cpu:
                device = 'cpu'
            return value.to(device=device, dtype=dtype)

        if any([skip_name in name for skip_name in skip_names]):
            if verbose:
                logger.info(f"Skipping {name} because it is in skip_names")
            return

        module_key, _, value_key = name.rpartition('.')
        try:
            submodule = module.get_submodule(module_key)
        except AttributeError as e:
            logger.info(f"Module {module_key} not found:\n{e}")
            return

        try:
            if quant_method=='bnb':
                param = submodule.get_parameter(value_key)
                if isinstance(param, Params4bit):
                    # With `sync_module_states=True`, a meta device Params4bit needs to be the same
                    # shape as the quantized Params4bit with an initialized quant_state. However,
                    # FSDP only syncs parameters and buffers, so the quant_state isn't copied. This
                    # workaround quantizes Params4bit to initialize quant_state on all ranks, then
                    # replaces Params4bit's data with a meta tensor to free memory on non-rank 0.
                    if is_dora:
                        setattr(submodule, "dora_scale", value.norm(p=2, dim=1).to(dtype=dtype).to("cpu"))
                    value = type(param)(value.to(device=device, dtype=dtype).data, **param.__dict__).cuda(device)
                    if to_meta:
                        value = type(param)(value.data.to("meta"), **value.__dict__)
                    elif to_cpu:
                        value = type(param)(value.data.to("cpu"), **value.__dict__)
                else:
                    value = type(param)(place_on_device(value).data)

        except AttributeError:
            # it's a buffer
            value = place_on_device(value)
            pass

        setattr(submodule, value_key, value)

    # All credit for this wrapping policy code goes to AnswerAI, specifically from https://github.com/AnswerDotAI/fsdp_qlora/blob/ed431272fd95b8ff57b5b12aff0f0cbdbd29cf96/train.py#L444C1-L479C60
    # released under Apache V2 and slightly modified to support the goals here in the Ditty library.
    def _get_wrapping_policy(self, custom_policy:bool=False, vanilla_policy:bool=False):
        from peft.tuners import PromptEncoder, PromptEmbedding, PrefixEncoder

        # if custom_policy:
        #     def lambda_policy_fn(module):
        #         # LoRA and DoRA trainable layers.
        #         return (isinstance(module, nn.Sequential) and all(m.weight.requires_grad for m in module)) or (isinstance(module, (DORALayer, MagnitudeLayer)))
        # else:
        def lambda_policy_fn(module):
            return (
                len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad
            )

        def self_attn_policy_fn(module):
            # Check module name is self_attn.
            return isinstance(module, tuple((*LLAMA_ATTENTION_CLASSES.values(), *MISTRAL_ATTENTION_CLASSES.values())))

        def mlp_policy_fn(module):
            # Check module name is self_attn.
            return isinstance(module, (LlamaMLP, MistralMLP))

        lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
        self_attn_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=self_attn_policy_fn)
        mlp_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=mlp_policy_fn)
        transformer_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=(LlamaDecoderLayer, MistralDecoderLayer),
        )
        if vanilla_policy:
            return transformer_wrap_policy

        policies=[lambda_policy, transformer_wrap_policy]
        if custom_policy:
            policies.extend([self_attn_policy, mlp_policy])
        return functools.partial(_or_policy, policies=policies)

    def run(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, token=self.model_token, torch_dtype=torch.bfloat16)
        if self.tokenizer.pad_token_id is None:
            logger.warn("Tokenizer did not have a pad_token_id, this was set to EOS which can cause a model to ignore producing an EOS token after finetuning.")
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        data = self.dataset()

        if self.use_fsdp or self.use_deep_speed:
            num_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES","").split(","))
            local_rank = int(os.environ.get("LOCAL_RANK"))
            rank = int(os.environ.get("RANK"))
            world_size = int(os.environ.get("WORLD_SIZE"))

            logger.info(f"I am rank: {rank} and local rank {local_rank}!")

        acc_kwargs = {
            "gradient_accumulation_steps": self.grad_accum,
            "project_dir": self.output_dir,
            "project_config": ProjectConfiguration(
                project_dir=self.output_dir,
                automatic_checkpoint_naming=True,
                save_on_each_node=True
            ),
            "mixed_precision": "bf16" if self.use_bfloat16 else "fp16",
        } 

        acc_kwargs = {**acc_kwargs, **self.accelerator_kwargs}

        self.accelerator = Accelerator(**acc_kwargs)

        # Experienced a weird case where upon saving an FSDP checkpoint, accelerate was initializing deepspeed for some reason and then hanging
        if self.use_fsdp:
            self.accelerator.state.deepspeed_plugin = None

        modified_load_kwargs = self.model_load_kwargs

        if self.use_flash_attn_2:
            modified_load_kwargs["attn_implementation"] = "flash_attention_2"

        if self.use_fsdp or self.use_deep_speed:
            if self.use_fsdp:
                modified_load_kwargs["low_cpu_mem_usage"] = True
                my_auto_wrap_policy = self._get_wrapping_policy(custom_policy=False, vanilla_policy=(not self.l8bit and not self.l4bit))

                self.accelerator.state.fsdp_plugin.auto_wrap_policy = my_auto_wrap_policy

                self.accelerator.state.device = torch.cuda.current_device()

            del modified_load_kwargs["device_map"]
            modified_load_kwargs["torch_dtype"] = torch.bfloat16 if self.use_bfloat16 else torch.float16


        if self.use_deep_speed or not (self.l8bit or self.l4bit):
            if self.use_fsdp and local_rank != 0:
                with init_empty_weights():
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name_or_path,
                        quantization_config=self.bnb_config,
                        **modified_load_kwargs,
                    )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    quantization_config=self.bnb_config,
                    **modified_load_kwargs,
                )
        elif self.use_fsdp:
            from bitsandbytes.nn import Linear4bit, Params4bit

            cfg = AutoConfig.from_pretrained(self.model_name_or_path)
            cfg.use_cache = False
            cfg.attn_implementation = "flash_attention_2"
            skip_modules = ["lm_head"]
    
            # load model on meta device without calling init and replace nn.Linear with Linear4bit
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(cfg)
                model.model = self._replace_linear(model.model, Linear4bit, compute_dtype=torch.bfloat16, quant_type='nf4', quant_storage=torch.bfloat16, skip_modules=skip_modules)

            model.is_loaded_in_4bit = True

            self.model = model

            # Grab the safetensors files that hold the weights
            try:
                idx = hub.cached_file(self.model_name_or_path, SAFE_WEIGHTS_INDEX_NAME)
                files, _ = hub.get_checkpoint_shard_files(self.model_name_or_path, idx)
            except OSError:
                try:
                    # This means the model doesn't have a model.safetensors.index.json because it is not sharded
                    files = []
                    files.append(hub.cached_file(self.model_name_or_path, SAFE_WEIGHTS_NAME))
                except OSError as e:
                    # This means the model probably doesn't have a safetensors file
                    raise e

            # Load in the weights, using our custom load_and_quantize method which quantizes Params4bit on the fly
            # and then places each layer on CPU or meta if using low_memory to minimize GPU memory usage
            def load_and_quantize_parallel(name_param, model, **kwargs):
                name, param = name_param
                self._load_and_quantize(model, name, param, **kwargs)

            quant_method = "bnb"
            param_count = sum((p.numel() for n,p in self.model.named_parameters()))
            if local_rank == 0:
                logger.info("Loading model", rank)
            if local_rank == 0:
                logger.info(f"Total model params: {param_count}")

            n_workers = self._n_loading_workers(quant_method, param_count)
            if rank == 0:
                logger.info(f"Using n_workers: {n_workers} for loading")

            for filename in tqdm(files, desc="Loading & Quantizing Model Shards", disable=rank!=0, position=0):
                weights = safetensors.torch.load_file(filename)
                parallel(load_and_quantize_parallel, iter(weights.items()), n_workers=n_workers, threadpool=True,
                         model=self.model, dtype=torch.bfloat16, device=torch.cuda.current_device(), skip_names=[],
                         to_cpu=(local_rank==0), to_meta=(local_rank!=0),
                         verbose=True, quant_method=quant_method, is_dora=(self.use_qdora))

            # cleanup any extra memory usage from parallel loading
            torch.cuda.empty_cache()


        target_modules = ["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]

        if self.accelerator.is_main_process:
            logger.info(self.model)

        if "gpt-neox" in self.model_name_or_path:
            target_modules = ["query_key_value", "xxx"]

        if "rwkv" in self.model_name_or_path:
            target_modules = ["key", "value", "receptance", "xxx"]

        if (self.l8bit or self.l4bit) and not self.use_fsdp:
            self.model = prepare_model_for_kbit_training(
                self.model, use_gradient_checkpointing=self.gradient_checkpointing
            )

        if hasattr(self.model, "embed_out"):
            output_embedding_layer = getattr(self.model, "embed_out")
            input_dtype = output_embedding_layer.weight.dtype

            class CastOutputToFloat(torch.nn.Sequential):
                r"""
                Manually cast to the expected dtype of the lm_head as sometimes there is a final layer norm that is casted
                in fp32
                """

                def forward(self, x):
                    return super().forward(x.to(input_dtype)).to(torch.float32)

            setattr(self.model, "embed_out", CastOutputToFloat(output_embedding_layer))

        if self.l8bit or self.l4bit:
            from peft import (
                TaskType,
                LoraConfig,
                get_peft_model,
            )

            from bitsandbytes.nn import Linear4bit, Params4bit

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                bias="none",
                use_dora=self.use_qdora
            )

            self.model.enable_input_require_grads()


            # Credit for this method goes to AnswerAI, specifically https://github.com/AnswerDotAI/fsdp_qlora/blob/ed431272fd95b8ff57b5b12aff0f0cbdbd29cf96/train.py#L164
            def setup_quantized_meta_for_peft(model: torch.nn.Module):
                """Replaces `quant_state.to` with a dummy function to prevent PEFT from moving `quant_state` to meta device"""
                def temp_to_method(self, *args, **kwargs):
                    return self
                for param in model.parameters():
                    if isinstance(param, Params4bit):
                        param.quant_state._orig_to = param.quant_state.to
                        param.quant_state.to = types.MethodType(temp_to_method, param.quant_state)

            # Credit for this method goes to AnswerAI, specifically https://github.com/AnswerDotAI/fsdp_qlora/blob/ed431272fd95b8ff57b5b12aff0f0cbdbd29cf96/train.py#L173 
            def setup_quantized_peft_meta_for_training(model: torch.nn.Module):
                """Replaces dummy `quant_state.to` method with the original function to allow training to continue"""
                for param in model.parameters():
                    if isinstance(param, Params4bit) and hasattr(param.quant_state, '_orig_to'):
                        param.quant_state.to = param.quant_state._orig_to
                        param.quant_state._orig_to = None

            if local_rank != 0:
                setup_quantized_meta_for_peft(self.model)

            self.model = get_peft_model(self.model, peft_config)

            setup_quantized_peft_meta_for_training(self.model)

            if self.use_fsdp:
                fsdp_plugin = self.accelerator.state.fsdp_plugin
                kwargs = {
                    "sharding_strategy": fsdp_plugin.sharding_strategy,
                    "cpu_offload": fsdp_plugin.cpu_offload,
                    "auto_wrap_policy": fsdp_plugin.auto_wrap_policy,
                    "mixed_precision": fsdp_plugin.mixed_precision_policy,
                    "sync_module_states": fsdp_plugin.sync_module_states,
                    "backward_prefetch": fsdp_plugin.backward_prefetch,
                    "forward_prefetch": fsdp_plugin.forward_prefetch,
                    "use_orig_params": fsdp_plugin.use_orig_params,
                    "ignored_modules": fsdp_plugin.ignored_modules,
                }
                self.model = FSDP(
                    self.model,
                    limit_all_gathers=True, # See https://github.com/pytorch/pytorch/issues/91165
                    device_id = torch.cuda.current_device(),
                    param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
                        if (rank!=0) else None, # TODO note about meta device and why we need this
                    **kwargs
                )

        ### Training
        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.config.use_cache = (
            not self.gradient_checkpointing
        )

        adam_beta1 = 0.9
        adam_beta2 = 0.999
        adam_epsilon = 1e-08
        adam_kwargs = {
            "betas": (adam_beta1, adam_beta2),
            "eps": adam_epsilon,
        }

        lr=0.97e-5

        if(self.use_fsdp):
            lr = lr * world_size

        if self.use_8bit_optim:
            optimizer = bnb.optim.Adam8bit(self.model.parameters(), lr=lr, **adam_kwargs)
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, **adam_kwargs
            )

        trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            accelerator=self.accelerator,
            dataset=data,
            device="cuda",
            grad_accum=self.grad_accum,
            fp16=self.fp16,
            output_dir=self.output_dir,
            checkpoint_every=self.checkpoint_every,
            load_checkpoint=self.load_checkpoint,
            use_bfloat16=self.use_bfloat16,
            seed=self.seed,
            hf_hub_token=self.hf_hub_token
        )

        trainer.train(
            epochs=self.epochs, max_steps=self.max_steps if self.max_steps else None
        )

        self.accelerator.wait_for_everyone()

        # Share adapters on the 🤗 Hub
        if self.push_to_hub:
            model = self.model

            if self.use_fsdp or self.use_deep_speed:
                logger.info("Unwrapping sharded model.")
                model = self.accelerator.unwrap_model(model)

            if self.merge_adapters and (self.l4bit or self.l8bit):
                logger.info("Merging adapters and unloading.")
                model = model.merge_and_unload(True)

            if self.accelerator.is_main_process:
                logger.info("Pushing to hub!")

            #Ian: Monkey patch existing push_to_hub with our push_to_hub that handles saving FSDP correctly.
            model.push_to_hub = types.MethodType(push_to_hub, model)
            model.push_to_hub(self.output_hub_repo, token=self.hf_hub_token, accelerator=self.accelerator, private=self.private_repo)
