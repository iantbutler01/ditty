# Ditty

A simple fine-tune.

## Who

Ditty powers finetuning the models at [BismuthOS](https://www.bismuthos.com) an AI enabled cloud platform. Build faster, Launch instantly. This library, like much work at Bismuth is part of our commitment to Open Source and contributing back to the communities we participate in.

## What
A very simple library for finetuning Huggingface Pretrained AutoModelForCausalLM such as GPTNeoX, Llama3, Mistral, etc. leveraging Huggingface Accelerate, Transformers, Datasets and Peft

Ditty has support for:
- LORA, QLORA
- 8bit, 4bit
- FP16, BFLOAT16, FP8 (via transformer-engine)
- 8bit Adam
- fp32 cpu offloading
- FSDP, FSDP + QLORA
- DeepSpeed
- Checkpointing
- Pushing to the hub

### FP8 Training

FP8 training is supported via [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine). This provides compute speedups on supported GPUs (H100, Ada Lovelace) by running matmuls in 8-bit floating point.

To use FP8:
1. Install transformer-engine: `pip install transformer-engine[pytorch]`
2. Pass `accelerator_kwargs={"mixed_precision": "fp8"}` to Pipeline

Note: FP8 primarily benefits transformer-based models where TransformerEngine can replace attention and linear layers. CNNs and other architectures may see limited benefit.

All other features work right out of the box and assumes you are running with a single GPU or distributed over multiple GPUs by default.

This has been tested on a 3 node cluster with 9 gpus.

## What Not
- Ditty does not support ASICs like TPU or Trainium.
- Ditty does not handle Sagemaker
- Ditty does not by default run with the CPU, except in cases where offloading is enabled (FSDP, DeepSpeed)
- Ditty does not handle evaluation sets or benchmarking, this may or may not change.

## Architecture

Ditty uses a pipeline pattern for training:

```
batch -> preprocessors -> model.forward -> postprocessors -> loss_calculator
```

This allows flexible composition of training workflows without modifying the core trainer.

## Classes

### Pipeline

The main entry point. Pass a `ModelFactory`, `DataLoader`, `LossCalculator`, and optional pre/post processors:

```python
from ditty import Pipeline, ModelFactory, CompositeLoss

model_factory = ModelFactory.from_instance(my_model)
# or: ModelFactory.from_checkpoint(path, model_class, **kwargs)

pipeline = Pipeline(
    model_factory=model_factory,
    dataloader=my_dataloader,
    loss_calculator=my_loss,
    preprocessors=[...],
    postprocessors=[...],
    output_dir="./output",
    bf16=True,
    use_8bit_optim=True,
    lr=2e-4,
    epochs=10,
)
pipeline.run()
```

### ModelFactory

Handles model creation and checkpoint loading:

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
        # Add data to forward_kwargs for the model
        ctx["forward_kwargs"] = ctx.get("forward_kwargs", {})
        ctx["forward_kwargs"]["my_param"] = some_value
        return batch, ctx

class MyPostProcessor(PostProcessor):
    def process(self, model_output, ctx: Context):
        # Extract targets for loss computation
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
        pred = self.get_prediction(model_output)  # model_output[output_index]
        target = self.get_target(ctx)             # ctx[target_key]
        mask = self.get_mask(ctx)                 # ctx[mask_key] or None
        loss = F.mse_loss(pred, target)
        return LossOutput(loss=loss, metrics={"mse": loss.item()})

# Combine multiple losses with weights
loss_calculator = CompositeLoss([
    (MSELoss(output_index=0), 1.0),
    (CrossEntropyLoss(output_index=1), 0.1),
])
```

### Trainer

Handles the training loop. You typically don't interact with this directly - Pipeline creates it internally.

### Data

Wraps HF Datasets with preprocessing support:

```python
data = Data(
    load_kwargs={"path": "dataset_name"},
    tokenizer=tokenizer,
    batch_size=8,
)

dataloader = data.prepare([
    ("filter", filter_fn, {}),
    ("map", transform_fn, {"batched": True}),
])
```

## Diffusion Support

Ditty includes utilities for diffusion model training:

```python
from ditty.diffusion import Scheduler

scheduler = Scheduler(
    max_timesteps=1000,
    schedule_type="cosine",  # or "linear"
    beta_start=0.0001,
    beta_end=0.02,
)
```

## Contracts (Optional)

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


## Tips

https://github.com/google/python-fire is a tool for autogenerating CLIs from Python functions, dicts and objects.

It can be combined with Pipeline to make a very quick cli for launching your process.

## Attribution / Statement of Changes

### Huggingface

Portions of this library look to Huggingface's transformers Trainer class as a reference and in some cases re-implements functions from Trainer, simplified to only account for GPU based work and overall narrower supported scope.

This statement is both to fulfill the obligations of the ApacheV2 licencse, but also because those folks do super cool work and I appreciate all they've done for the community and its just right to call this out.

Portions of this library modify Huggingface's hub code to support properly saving FSDP sharded models, for this code the appropriate license is reproduced in file and modifications are stated.

### Answer.ai

Portions of this library implement Answer.ai's method of quantizing and loading model layers + placing them on device manually as well as wrapping code for FSDP. 

Thanks so much for the Answer.ai team, without their work it would have been significantly harder to implement FSDP+QLORA in Ditty

Our implementation is basically a complete reproduction with slight changes to make it work with Ditty nuances, the original work can be found here, it is really good work and you should check it out:
https://github.com/AnswerDotAI/fsdp_qlora

## License

Apache V2 see the LICENSE file for full text.
