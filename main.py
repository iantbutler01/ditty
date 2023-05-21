import transformers
from itertools import chain
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import TaskType, LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_int8_training
from typing import Optional
import torch
from dataclasses import dataclass, field
BitsAndBytesConfig()

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default="theblackcat102/pythia-3b-deduped-sft-r1",
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )

@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default="code_search_net", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_language: Optional[str] = field(
        default="python", metadata={"help": "The name of the specific language within the dataset to use."}
    )
    block_size: Optional[int] = field(
        default=1024, metadata={"help": "The maximum length of a training sample (in tokens)."}
    ),

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

model_args, data_args, training_args = parser.parse_args_into_dataclasses()

tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, load_in_8bit=True ,device_map="auto").cuda()

target_modules = ["query_key_value", "xxx"]
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, target_modules=target_modules, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.05
)

# Enable grads for checkpointing
# model.enable_input_require_grads()
model = prepare_model_for_int8_training(model)

if hasattr(model, "embed_out"):
    output_embedding_layer = getattr(model, "embed_out")
    input_dtype = output_embedding_layer.weight.dtype

    class CastOutputToFloat(torch.nn.Sequential):
        r"""
        Manually cast to the expected dtype of the lm_head as sometimes there is a final layer norm that is casted
        in fp32
        """

        def forward(self, x):
            return super().forward(x.to(input_dtype)).to(torch.float32)

    setattr(model, "embed_out", CastOutputToFloat(output_embedding_layer))

model = get_peft_model(model, peft_config)

block_size = 1024

# ### Training
data = load_dataset(data_args.dataset_name, data_args.dataset_language)
columns = data["train"].features

def filter_longer(sample):
    tokens, _attn_mask = tokenizer(sample["whole_func_string"])

    return len(tokens) <= block_size

data = data.filter(filter_longer)
data = data.map(lambda sample: tokenizer(sample["whole_func_string"],
                                         padding='max_length',
                                         max_length=block_size), batched=True, remove_columns=columns)

def truncate(sample):
    sample["attention_mask"] = sample["attention_mask"][:block_size]
    sample["input_ids"] = sample["input_ids"][:block_size]

    return sample

data = data.map(truncate)

model.gradient_checkpointing_enable()
trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        # max_steps=4,
        max_steps=200,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir='outputs'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

# ## Share adapters on the 🤗 Hub
model.push_to_hub(training_args.output_dir, use_auth_token=True)

# Load adapters from the Hub and generate some output texts:

peft_model_id = training_args.output_dir
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)
# You can then directly use the trained model or the model that you have loaded from the 🤗 Hub for inference

model.config.use_cache = True
batch = tokenizer("def print_hello(): ", return_tensors="pt")

with torch.cuda.amp.autocast():
    output_tokens = model.generate(**batch, max_new_tokens=50)

print("\n\n", tokenizer.decode(output_tokens[0], skip_special_tokens=True))
