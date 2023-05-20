import transformers
from itertools import chain
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import TaskType, LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_int8_training
from typing import Optional
import torch
from dataclasses import dataclass, field
BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

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
    )

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

model_args, data_args, training_args = parser.parse_args_into_dataclasses()

tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, load_in_8bit=True, device_map="auto")

target_modules = ["query_key_value", "xxx"]
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, target_modules=target_modules, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.05
)

model = prepare_model_for_int8_training(model)
model = get_peft_model(model, peft_config)

block_size = data_args.block_size

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


# ### Training
data = load_dataset(data_args.dataset_name, data_args.dataset_language)
columns = data["train"].features
data = data.map(lambda samples: tokenizer(samples["whole_func_string"]), batched=True, remove_columns=columns)
data = data.map(group_texts, batched=True)

model.gradient_checkpointing_enable()
trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=training_args,
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