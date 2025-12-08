# FSDP2 Implementation Notes

## Current State (Manual FSDP2)

ModelFactory manually applies FSDP2 via `fully_shard()` in `_apply_fsdp()`. This requires:
- DTensor detection hacks in trainer.py (`_has_dtensor_params()`)
- Manual device placement with `torch.cuda.set_device(local_rank)`
- Skipping model in `accelerator.prepare()` to avoid DDP wrapping conflict

## Why Manual Was Used

The `FSDPConfig.transformer_layers` lets you pass actual class objects to specify which layers get sharded:
```python
fsdp_config = FSDPConfig(
    enabled=True,
    transformer_layers=[ResidualBlock, TransformerBlock],  # actual classes
)
```

## Better Approach: Accelerate's Native FSDP2

Accelerate supports FSDP2 via `FullyShardedDataParallelPlugin` with `fsdp_version=2`.

```python
from accelerate import FullyShardedDataParallelPlugin, Accelerator

fsdp_plugin = FullyShardedDataParallelPlugin(
    fsdp_version=2,
    transformer_layer_cls_to_wrap="ResidualBlock,TransformerBlock",  # class names as strings
    use_orig_params=True,  # needed for frozen/trainable parameter mixing
)
accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
```

## Why Accelerate FSDP2 is Better

1. No DTensor detection hacks needed
2. No manual device placement
3. No skipping model in `prepare()`
4. Accelerate handles everything automatically
5. `use_orig_params=True` supports frozen layers (like our frozen decoder)

## Migration TODO

1. Remove manual FSDP2 from ModelFactory (`_apply_fsdp()`)
2. Remove DTensor detection from trainer.py
3. Remove `torch.cuda.set_device()` hack
4. Add `fsdp_plugin` parameter to Pipeline
5. Pass layer class names as comma-separated string instead of class objects
6. Update train_ditty.py to use new API

## Why Manual FSDP2 is Required for QLoRA

QLoRA + FSDP2 requires special handling that accelerate's plugin doesn't fully support:

1. **Rank-based device loading**: Rank 0 loads quantized weights to CPU, other ranks load to meta device
2. **FSDP2 distributes from rank 0**: After loading, FSDP2's `fully_shard()` distributes shards to all ranks
3. **`bnb_4bit_quant_storage`**: Must be set to enable FSDP-QLoRA compatibility (e.g., `torch.bfloat16`)

From `_load_quantized_model()` in model_factory.py:
```python
parallel(
    load_and_quantize_parallel,
    iter(weights.items()),
    model=model,
    to_cpu=(local_rank == 0),      # rank 0 loads to CPU
    to_meta=(local_rank != 0),     # others load to meta
)
```

This pattern is documented in Answer.AI's fsdp_qlora implementation and bitsandbytes docs.

## References

- https://huggingface.co/docs/accelerate/en/usage_guides/fsdp
- https://huggingface.co/docs/accelerate/en/concept_guides/fsdp1_vs_fsdp2
- https://github.com/huggingface/accelerate/issues/2873
- https://huggingface.co/docs/bitsandbytes/main/en/fsdp_qlora
- https://www.answer.ai/posts/2024-03-14-fsdp-qlora-deep-dive.html
- https://github.com/AnswerDotAI/fsdp_qlora
