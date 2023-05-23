# Ditty

A simple fine-tune.

## What
A very simple library for finetuning Huggingface Pretrained AutoModelForCausalLM such as GPTNeoX, leveraging Huggingface Accelerate, Transformers, Datasets and Peft

Ditty has support for LORA, 8bit, and fp32 cpu offloading right out of the box and assumes you are running with a single GPU or distributed over multiple GPUs by default.

## What Not
- Ditty does not support ASICs like TPU or Trainium.
- Ditty does not handle Sagemaker
- Ditty does not by default run with the CPU
- Ditty does not handle evaluation sets or benchmarking, this may or may not change.

## Soon
- Ditty will handle FP16
- Ditty may handle distributed cluster finetuning
- Ditty will support DeepSpeed

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
    load_args=(self.dataset_name, self.dataset_language),
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

## Tips

https://github.com/google/python-fire is a tool for autogenerating CLIs from Python functions, dicts and objects.

It can be combined with Pipeline to make a very quick cli for launching your process.

## Attribution / Statement of Changes

Portions of this library look to Huggingface's transformers Trainer class as a reference and in some cases re-implements functions from Trainer, simplified to only account for GPU based work and overall narrower supported scope.

This statement is both to fulfill the obligations of the ApacheV2 licencse, but also because those folks do super cool work and I appreciate all they've done for the community and its just right to call this out.

## License

Apache V2 see the LICENSE file for full text.