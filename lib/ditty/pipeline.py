from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

import os
import functools
import types
from typing import Optional, List, Dict, Any
import bitsandbytes as bnb
from accelerate import Accelerator, infer_auto_device_map, init_empty_weights
from accelerate.utils import ProjectConfiguration

import torch
from torch.utils.data import DataLoader
from .trainer import Trainer
from .data import Data
from peft import prepare_model_for_kbit_training 
from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LLAMA_ATTENTION_CLASSES, LlamaMLP
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MISTRAL_ATTENTION_CLASSES, MistralMLP
import logging

logging.basicConfig(level=logging.INFO)


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
        merge_adapters: bool = True
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
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        data = self.dataset()

        if self.use_fsdp or self.use_deep_speed:
            num_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES","").split(","))
            local_rank = int(os.environ.get("LOCAL_RANK"))
            rank = int(os.environ.get("RANK"))
            world_size = int(os.environ.get("WORLD_SIZE"))

            print(f"I am rank: {rank} and local rank {local_rank}!")

        acc_kwargs = {
            "gradient_accumulation_steps": self.grad_accum,
            "project_dir": self.output_dir,
            "project_config": ProjectConfiguration(
                project_dir=self.output_dir,
                automatic_checkpoint_naming=True,
                save_on_each_node=True
            ),
            # "mixed_precision": "bf16" if self.use_bfloat16 else "fp16"
            "mixed_precision": "no"
        } 

        acc_kwargs = {**acc_kwargs, **self.accelerator_kwargs}

        self.accelerator = Accelerator(**acc_kwargs)


        modified_load_kwargs = self.model_load_kwargs

        if self.use_flash_attn_2:
            modified_load_kwargs["attn_implementation"] = "flash_attention_2"

        if self.use_fsdp or self.use_deep_speed:
            if self.use_fsdp:
                modified_load_kwargs["low_cpu_mem_usage"] = True
                my_auto_wrap_policy = self._get_wrapping_policy(custom_policy=False, vanilla_policy=(not self.l8bit and not self.l4bit))

                self.accelerator.state.fsdp_plugin.auto_wrap_policy = my_auto_wrap_policy

                print(self.accelerator.state.fsdp_plugin.limit_all_gathers)

            del modified_load_kwargs["device_map"]
            modified_load_kwargs["torch_dtype"] = torch.bfloat16 if self.use_bfloat16 else torch.float16

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

        target_modules = ["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]

        print(self.model)

        if "gpt-neox" in self.model_name_or_path:
            target_modules = ["query_key_value", "xxx"]

        if "rwkv" in self.model_name_or_path:
            target_modules = ["key", "value", "receptance", "xxx"]

        if self.l8bit or self.l4bit and not self.use_fsdp:
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

            setup_quantized_meta_for_peft(self.model)

            self.model = get_peft_model(self.model, peft_config)

            setup_quantized_peft_meta_for_training(self.model)

        print("FINISHED LOADING MODEL")
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

        # ## Share adapters on the 🤗 Hub
        if self.push_to_hub and self.accelerator.is_main_process:
            model = self.model

            if self.merge_adapters and (self.l4bit or self.l8bit):
                model = model.merge_and_unload()

            self.model.push_to_hub(self.output_hub_repo, token=self.hf_hub_token)
