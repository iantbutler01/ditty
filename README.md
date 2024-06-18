# Ditty

A simple fine-tune.

## Who

Ditty powers finetuning the models at [BismuthOS](https://www.bismuthos.com) an AI enabled cloud platform. Build faster, Launch instantly. This library, like much work at Bismuth is part of our commitment to Open Source and contributing back to the communities we participate in.

## What
A very simple library for finetuning Huggingface Pretrained AutoModelForCausalLM such as GPTNeoX, Llama3, Mistral, etc. leveraging Huggingface Accelerate, Transformers, Datasets and Peft

Ditty has support for LORA, 8bit, and fp32 cpu offloading right out of the box and assumes you are running with a single GPU or distributed over multiple GPUs by default.

Checkpointing supported.

We now also support FSDP, QLORA, FSDP + QLORA and DeepSpeed Z3! This has been tested on a 3 node cluster with 9 gpus.

FP16, BFLOAT16 supported.

## What Not
- Ditty does not support ASICs like TPU or Trainium.
- Ditty does not handle Sagemaker
- Ditty does not by default run with the CPU, except in cases where offloading is enabled (FSDP, DeepSpeed)
- Ditty does not handle evaluation sets or benchmarking, this may or may not change.

## Classes

### Pipeline

Pipeline is responsible for running the entire show. Simply subclass Pipeline and implement the `dataset` method for your custom data, this must return a `torch.utils.data.DataLoader`

Instantiate with your chosen config and then simply call `run`.

### Trainer

Trainer does what it's name implies, which is to train the model. You may never need to touch this if you're not interested in customizing the training loop.

### Data

Data wraps an HF Dataset and can configure length grouped sampling and random sampling, as well as handling collation, batching, seeds, removing unused columns and a few other things.

The primary way of using this class is through the `prepare` method which takes a list of operations to perform against the dataset. These are normal operations like `map` and `filter`.

Example:
```python
data = Data(
    load_kwargs={"path": self.dataset_name, "name":
                 self.dataset_language},
    tokenizer=self.tokenizer,
    seed=self.seed,
    batch_size=self.batch_size,
    grad_accum=self.grad_accum,
)

....sic

dataloader = data.prepare(
        [
            ("filter", filter_longer, {}),
            ("map", do_something, dict(batched=True, remove_columns=columns)), 
            ("map", truncate, {}),
        ]
    )
```

This can be used to great effect when overriding the `dataset` method in a subclass of `Pipeline`.

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
