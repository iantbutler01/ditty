# Ditty

A distributed training library for PyTorch.

## What

A flexible library for training and finetuning models with modern distributed training support. Integrates with the HuggingFace ecosystem (Accelerate, Transformers, Datasets, PEFT, Hub) while providing a custom training loop and pipeline architecture. Works with any PyTorch model - from pretrained HuggingFace models to custom architectures like diffusion models.

Ditty has support for:
- Full training and finetuning
- LORA, QLORA
- 8bit, 4bit quantization
- FP16, BFLOAT16, FP8 (via transformer-engine)
- 8bit Adam (torchao or bitsandbytes backends)
- FSDP2 with DTensor-based sharding
- FSDP + QLORA (needs testing with FSDP2)
- torch.compile compatible
- Checkpointing and resume
- Pushing to HuggingFace Hub

### FSDP2

Ditty uses PyTorch's FSDP2 with per-parameter DTensor sharding. This provides:
- Memory-efficient training across multiple GPUs
- Compatible with torchao's 8-bit optimizers
- Works with torch.compile

To enable FSDP2, pass an `FSDPConfig` to your `ModelFactory`:

```python
from ditty import ModelFactory, FSDPConfig

fsdp_config = FSDPConfig(
    enabled=True,
    transformer_layers=[MyTransformerBlock],  # Layers to shard
)

model_factory = ModelFactory.from_instance(
    my_model,
    fsdp_config=fsdp_config,
)
```

### 8-bit Optimizers

Two backends are available for 8-bit Adam:

- `torchao` (default) - Works with FSDP2/DTensor, torch.compile compatible
- `bnb` - bitsandbytes, does not work with FSDP2

```python
pipeline = Pipeline(
    model_factory=model_factory,
    dataset=dataset,
    use_8bit_optim=True,
    optim_backend="torchao",  # or "bnb"
    ...
)
```

### FP8 Training

FP8 training is supported via [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine). This provides compute speedups on supported GPUs (H100, Ada Lovelace).

To use FP8:
1. Install transformer-engine: `pip install transformer-engine[pytorch]`
2. Pass `accelerator_kwargs={"mixed_precision": "fp8"}` to Pipeline

## Architecture

Ditty uses a pipeline pattern for training:

```
batch -> preprocessors -> model.forward -> postprocessors -> loss_calculator
```

This allows flexible composition of training workflows without modifying the core trainer.

## Classes

### Pipeline

The main entry point. Pass a `ModelFactory`, dataset, `LossCalculator`, and optional pre/post processors:

```python
from ditty import Pipeline, ModelFactory, CompositeLoss

model_factory = ModelFactory.from_instance(my_model)
# or: ModelFactory.from_checkpoint(path, model_class, **kwargs)

pipeline = Pipeline(
    model_factory=model_factory,
    dataset=my_dataset,
    collate_fn=my_collate_fn,
    loss_calculator=my_loss,
    preprocessors=[...],
    postprocessors=[...],
    output_dir="./output",
    fp16=True,
    use_8bit_optim=True,
    lr=2e-4,
    epochs=10,
)
pipeline.run()
```

### ModelFactory

Handles model creation, checkpoint loading, and FSDP wrapping:

- `ModelFactory.from_instance(model)` - Wrap an existing model instance
- `ModelFactory.from_checkpoint(path, model_class, **kwargs)` - Load from checkpoint

### PreProcessor / PostProcessor

Transform data before the model or outputs after:

```python
from ditty.processors import PreProcessor, PostProcessor, Context

class MyPreProcessor(PreProcessor):
    def __init__(self):
        super().__init__(contract="batch:3:i64 -> batch:3:i64 | ctx.my_key:0:i64")

    def process(self, batch, ctx: Context):
        ctx["forward_kwargs"] = ctx.get("forward_kwargs", {})
        ctx["forward_kwargs"]["my_param"] = some_value
        return batch, ctx

class MyPostProcessor(PostProcessor):
    def process(self, model_output, ctx: Context):
        ctx["target"] = extract_target(model_output, ctx["original_batch"])
        return model_output, ctx
```

### LossCalculator

Compute loss from model outputs. Use `output_index` to select from tuple outputs:

```python
from ditty.loss import LossCalculator, LossOutput, CompositeLoss

class MyLoss(LossCalculator):
    def __init__(self):
        super().__init__(output_index=0, target_key="target", mask_key="mask")

    def compute(self, model_output, ctx) -> LossOutput:
        pred = self.get_prediction(model_output)
        target = self.get_target(ctx)
        mask = self.get_mask(ctx)
        loss = F.mse_loss(pred, target)
        return LossOutput(loss=loss, metrics={"mse": loss.item()})

# Combine multiple losses with weights
loss_calculator = CompositeLoss([
    (MSELoss(output_index=0), 1.0),
    (CrossEntropyLoss(output_index=1), 0.1),
])
```

### Contracts (Optional)

Processors and losses can declare contracts for validation:

```python
# Terse syntax: "input_shape -> output_shape | ctx.key:shape:dtype"
contract = "batch:3:i64 -> batch:3:i64 | ctx.t:1:i64"
```

Pipeline validates that contracts chain together correctly at initialization.

## Setup

```
pip install ditty
```

## Attribution

### Huggingface

Portions of this library reference Huggingface's transformers Trainer class and in some cases re-implement functions from Trainer.

### Answer.ai

Portions of this library implement Answer.ai's method for FSDP+QLORA. The original work can be found at: https://github.com/AnswerDotAI/fsdp_qlora

## License

Apache V2 - see the LICENSE file for full text.
