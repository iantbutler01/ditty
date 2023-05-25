from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)
import torch
from torch.utils.data import DataLoader
from .trainer import Trainer
from .data import Data
import logging

logging.basicConfig(level=logging.INFO)

class Pipeline():
    def __init__(
        self,
        output_dir="./output",
        dataset_name="code_search_net",
        dataset_language="python",
        model_name_or_path="theblackcat102/pythia-3b-deduped-sft-r1",
        hf_hub_token=None,
        hf_hub_model_id=None,
        fp16=True,
        l8bit=True,
        seed=None,
        batch_size=4,
        grad_accum=4,
        push_to_hub=True,
        fp32_cpu_offload=False,
        load_checkpoint=True,
        checkpoint_every=1000,
    ):
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.dataset_language = dataset_language
        self.model_name_or_path = model_name_or_path
        self.hf_hub_token = hf_hub_token
        self.hf_hub_model_id = hf_hub_model_id
        self.fp16 = fp16
        self.seed = seed
        self.epochs = 1
        self.max_steps = None
        self.l8bit = l8bit
        self.push_to_hub = push_to_hub
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.block_size = 2048
        self.fp32_cpu_offload = fp32_cpu_offload
        self.bb_conf = BitsAndBytesConfig(load_in_8bit=l8bit, llm_int8_enable_fp32_cpu_offload=fp32_cpu_offload)
        self.checkpoint_every = checkpoint_every
        self.load_checkpoint = load_checkpoint

    def dataset(self) -> DataLoader:
        '''
        Subclass Pipeline and customize for your own dataset.
        '''

        data = Data(
            load_args=(self.dataset_name, self.dataset_language),
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
            sample["attention_mask"] = sample["attention_mask"][:self.block_size]
            sample["input_ids"] = sample["input_ids"][:self.block_size]

            return sample


        dataloader = data.prepare(
            [
                ("filter", filter_longer, {}),
                (
                    "map",
                    lambda sample: self.tokenizer(
                        sample["whole_func_string"], padding="max_length", max_length=self.block_size
                    ),
                    dict(batched=True, remove_columns=columns, num_proc=8),
                ),
                ("map", truncate, dict(num_proc=8)),
            ]
        )
        return dataloader

    def run(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        data = self.dataset()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            device_map="auto",
            load_in_8bit=self.l8bit,
        )

        target_modules = None
        if "gpt-neox" in self.model_name_or_path:
            target_modules = ["query_key_value", "xxx"]

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
        )

        if self.l8bit:
            self.model = prepare_model_for_int8_training(self.model)

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

        self.model = get_peft_model(self.model, peft_config)

        ### Training


        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

        adam_beta1 = 0.9
        adam_beta2 = 0.999
        adam_epsilon = 1e-08
        adam_kwargs = {
            "betas": (adam_beta1, adam_beta2),
            "eps": adam_epsilon,
        }
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.97e-5, **adam_kwargs)

        trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            dataset=data,
            device="cuda",
            grad_accum=self.grad_accum,
            fp16=self.fp16,
            output_dir=self.output_dir,
            checkpoint_every=self.checkpoint_every,
            load_checkpoint=self.load_checkpoint,
            seed=self.seed,
        )

        trainer.train(epochs=self.epochs, max_steps=self.max_steps if self.max_steps else None)

        # ## Share adapters on the 🤗 Hub
        if self.push_to_hub:
            self.model.push_to_hub(self.output_dir, use_auth_token=True)