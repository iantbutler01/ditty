# Spec: `Pipeline.print()` Method

## Location
`Pipeline` class in `ditty/lib/ditty/pipeline.py`

## Signature
```python
def print(self) -> None:
```

## Behavior

### 1. Build the model
- Call `self.model_factory.build()` (lazy - only if `self.model` not already set)
- Store result in `self.model` for reuse

### 2. Print model architecture
- Class name
- Total / trainable / frozen parameter counts
- Model attributes (vocab_size, embed_dim, hidden_dim, latent_dim, num_layers, num_heads)
- Model contract from `self.model_factory.contract`
- Actual `repr(model)` or nn.Module layer structure

### 3. Print data flow diagram
```
Dataset
    │
    ▼
PreProcessor 1: token_masker {'mask_prob': 0.15}
    │  batch:2:i64 -> batch:2:i64
    ▼
PreProcessor 2: forward_kwargs_injector
    │  batch:2:i64 -> batch:2:i64
    ▼
╔═══════════════════════════════════════╗
║  MODEL: ExampleModel                  ║
║  batch:2:i64 -> logits:3:f, hidden:3:f║
╚═══════════════════════════════════════╝
    │
    ▼
PostProcessor 1: target_extractor
    │  logits:3:f, hidden:3:f -> logits:3:f, hidden:3:f
    ▼
LOSS: CompositeLoss
  • masked_cross_entropy_loss (weight=1.0)
    logits:3:f | ctx.target:2:i64, ctx.mask:2:f -> loss:0:f
  • hidden_regularizer (weight=1.0)
    | ctx.hidden_states:3:f -> loss:0:f
```

### 4. Run contract validation
- Parse all contracts from preprocessors, model_factory, postprocessors, loss_calculator
- Run `validate_pipeline_chain()`
- Print PASSED / FAILED with specific errors

## Output Format
```
======================================================================
=================== DITTY PIPELINE ARCHITECTURE ======================
======================================================================

>>> MODEL
    Class: ExampleModel
    Parameters: 1,234,567 total, 1,000,000 trainable, 234,567 frozen
    Config: {'vocab_size': 1000, 'embed_dim': 256, 'hidden_dim': 512}
    Contract: batch:2:i64 -> logits:3:f, hidden:3:f

>>> DATA FLOW
    [diagram as above]

>>> CONTRACT VALIDATION
    Status: PASSED
    All tensor shapes and dtypes chain correctly through the pipeline.

======================================================================
```

## Changes Required

1. **Add `print()` method to `Pipeline` class** in `pipeline.py`
   - Access `self.model_factory.contract` for model contract
   - Build model if needed to get actual architecture
   - Use `self.preprocessors`, `self.postprocessors`, `self.loss_calculator`

2. **Remove from `example.py`:**
   - Delete `print_pipeline()` function
   - Update example functions to use `pipeline.print()` after construction

3. **Remove from `__init__.py`:**
   - Delete `from .example import print_pipeline`
   - Delete `"print_pipeline"` from `__all__`

## Usage
```python
pipeline = Pipeline(
    model_factory=model_factory,
    dataset=dataset,
    ...
)
pipeline.print()  # prints full architecture
pipeline.run()    # runs training
```
