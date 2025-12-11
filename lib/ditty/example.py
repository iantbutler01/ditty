"""
End-to-end example demonstrating the ditty training pipeline.

Components demonstrated:
- Data: Dataset loading and preparation
- PreProcessor: Transform batch before model forward
- PostProcessor: Transform model output before loss calculation
- LossCalculator: Compute loss from model output
- ModelFactory: Create and configure models
- Pipeline: Orchestrate the entire training flow
- Contracts: Declarative tensor shape/dtype validation
"""
import torch
import torch.nn as nn
from typing import Any, Dict, Tuple, List
from datasets import Dataset

from .processors import PreProcessor, PostProcessor, Context
from .loss import LossCalculator, LossOutput, CompositeLoss
from .model_factory import ModelFactory, ModelTransform, FSDPConfig
from .pipeline import Pipeline
from .contract import parse_contract, format_pipeline_contracts, validate_pipeline_chain


# --- Example Model ---


class ExampleModel(nn.Module):
    """Simple encoder-decoder model for demonstration."""

    def __init__(self, vocab_size: int = 1000, embed_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.Linear(embed_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        emb = self.embedding(input_ids)
        hidden = self.encoder(emb)
        logits = self.decoder(hidden)
        if return_hidden:
            return logits, hidden
        return (logits,)


# --- Example Preprocessors ---


class TokenMasker(PreProcessor):
    """Randomly mask tokens for masked language modeling."""

    def __init__(self, mask_prob: float = 0.15, mask_token_id: int = 0):
        super().__init__(contract="batch:2:i64 -> batch:2:i64")
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id

    def config(self) -> Dict[str, Any]:
        return {"mask_prob": self.mask_prob}

    def process(self, batch: torch.Tensor, ctx: Context) -> Tuple[torch.Tensor, Context]:
        ctx["original_input_ids"] = batch.clone()
        mask = torch.rand_like(batch.float()) < self.mask_prob
        masked_batch = batch.clone()
        masked_batch[mask] = self.mask_token_id
        ctx["mask"] = mask.float()
        return masked_batch, ctx


class ForwardKwargsInjector(PreProcessor):
    """Inject additional kwargs into model forward call."""

    def __init__(self, contract: str = "", **kwargs):
        super().__init__(contract=contract)
        self.kwargs = kwargs

    def config(self) -> Dict[str, Any]:
        return self.kwargs

    def process(self, batch: Any, ctx: Context) -> Tuple[Any, Context]:
        ctx["forward_kwargs"] = ctx.get("forward_kwargs", {})
        ctx["forward_kwargs"].update(self.kwargs)
        return batch, ctx


# --- Example Postprocessors ---


class TargetExtractor(PostProcessor):
    """Extract targets from context for loss computation."""

    def __init__(self, target_key: str = "original_input_ids", contract: str = ""):
        super().__init__(contract=contract)
        self.target_key = target_key

    def process(
        self, model_output: Tuple[Any, ...], ctx: Context
    ) -> Tuple[Tuple[Any, ...], Context]:
        ctx["target"] = ctx[self.target_key]
        return model_output, ctx


class HiddenStateExtractor(PostProcessor):
    """Extract hidden states for auxiliary losses."""

    def __init__(self, output_index: int = 1, contract: str = ""):
        super().__init__(contract=contract)
        self.output_index = output_index

    def process(
        self, model_output: Tuple[Any, ...], ctx: Context
    ) -> Tuple[Tuple[Any, ...], Context]:
        if len(model_output) > self.output_index:
            ctx["hidden_states"] = model_output[self.output_index]
        return model_output, ctx


# --- Example Loss Calculators ---


class MaskedCrossEntropyLoss(LossCalculator):
    """Cross-entropy loss only on masked positions."""

    def __init__(self, vocab_size: int = 1000, contract: str = ""):
        super().__init__(
            output_index=0,
            target_key="target",
            mask_key="mask",
            contract=contract,
        )
        self.vocab_size = vocab_size

    def compute(self, model_output: Tuple[Any, ...], ctx: Context) -> LossOutput:
        logits = self.get_prediction(model_output)
        target = self.get_target(ctx)
        mask = self.get_mask(ctx)

        logits_flat = logits.reshape(-1, self.vocab_size)
        target_flat = target.reshape(-1)
        mask_flat = mask.reshape(-1) if mask is not None else torch.ones_like(target_flat, dtype=torch.float)

        loss_per_token = torch.nn.functional.cross_entropy(
            logits_flat, target_flat, reduction="none"
        )
        loss = (loss_per_token * mask_flat).sum() / mask_flat.sum().clamp(min=1)
        return LossOutput(loss=loss, metrics={"masked_ce": loss.item()})


class HiddenRegularizer(LossCalculator):
    """L2 regularization on hidden states."""

    def __init__(self, weight: float = 0.01, contract: str = ""):
        super().__init__(contract=contract)
        self.weight = weight

    def compute(self, model_output: Tuple[Any, ...], ctx: Context) -> LossOutput:
        hidden = ctx.get("hidden_states")
        if hidden is None:
            device = ctx.get("device", "cuda")
            return LossOutput(loss=torch.tensor(0.0, device=device), metrics={"hidden_reg": 0.0})
        loss = self.weight * (hidden ** 2).mean()
        return LossOutput(loss=loss, metrics={"hidden_reg": loss.item()})


# --- Example Model Transform ---


class FreezeEmbeddings(ModelTransform):
    """Freeze embedding layer during training."""

    def transform(self, model: nn.Module) -> nn.Module:
        if hasattr(model, "embedding"):
            for param in model.embedding.parameters():
                param.requires_grad = False
        return model


# --- Architecture Printing ---


def print_pipeline(
    model: nn.Module,
    preprocessors: List[PreProcessor],
    postprocessors: List[PostProcessor],
    loss_calculator: LossCalculator,
    model_contract: str = "",
):
    """
    Print complete pipeline architecture with data flow and contract validation.

    This gives you an end-to-end view of:
    - Model architecture and parameters
    - Data flow through preprocessors -> model -> postprocessors -> loss
    - Contract specifications at each stage
    - Contract chain validation results
    """
    width = 70

    print("\n" + "=" * width)
    print(" DITTY PIPELINE ARCHITECTURE ".center(width, "="))
    print("=" * width)

    # --- Model ---
    print("\n>>> MODEL")
    print(f"    Class: {model.__class__.__name__}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"    Parameters: {total_params:,} total, {trainable_params:,} trainable, {frozen_params:,} frozen")

    attrs = ["vocab_size", "embed_dim", "hidden_dim", "latent_dim", "num_layers", "num_heads"]
    model_attrs = {a: getattr(model, a) for a in attrs if hasattr(model, a)}
    if model_attrs:
        print(f"    Config: {model_attrs}")

    if model_contract:
        print(f"    Contract: {model_contract}")

    # --- Data Flow ---
    print("\n>>> DATA FLOW")
    print("    ┌─────────────────────────────────────────────────────────────┐")
    print("    │  Dataset                                                    │")
    print("    └─────────────────────────────────────────────────────────────┘")
    print("                                  │")
    print("                                  ▼")

    for i, p in enumerate(preprocessors):
        cfg = p.config()
        cfg_str = f" {cfg}" if cfg else ""
        print(f"    ┌─ PreProcessor {i+1}: {p.name}{cfg_str}")
        if p.contract:
            print(f"    │  {p.contract}")
        print("    └──────────────────────────────────────────────────────────────")
        print("                                  │")
        print("                                  ▼")

    print("    ╔═════════════════════════════════════════════════════════════╗")
    print(f"    ║  MODEL: {model.__class__.__name__:<53} ║")
    if model_contract:
        contract_display = model_contract[:55] + "..." if len(model_contract) > 55 else model_contract
        print(f"    ║  {contract_display:<61} ║")
    print("    ╚═════════════════════════════════════════════════════════════╝")
    print("                                  │")
    print("                                  ▼")

    for i, p in enumerate(postprocessors):
        print(f"    ┌─ PostProcessor {i+1}: {p.name}")
        if p.contract:
            print(f"    │  {p.contract}")
        print("    └──────────────────────────────────────────────────────────────")
        print("                                  │")
        print("                                  ▼")

    print("    ┌─────────────────────────────────────────────────────────────┐")
    if isinstance(loss_calculator, CompositeLoss):
        print("    │  LOSS: CompositeLoss                                        │")
        for calc, weight in loss_calculator.losses:
            print(f"    │    • {calc.name} (weight={weight})")
            if calc.contract:
                contract_short = calc.contract[:50] + "..." if len(calc.contract) > 50 else calc.contract
                print(f"    │      {contract_short}")
    else:
        print(f"    │  LOSS: {loss_calculator.name:<56} │")
        if loss_calculator.contract:
            print(f"    │  {loss_calculator.contract:<61} │")
    print("    └─────────────────────────────────────────────────────────────┘")

    # --- Contract Validation ---
    print("\n>>> CONTRACT VALIDATION")

    pre_contracts = [(p.name, parse_contract(p.contract)) for p in preprocessors if p.contract]
    model_c = parse_contract(model_contract) if model_contract else None
    post_contracts = [(p.name, parse_contract(p.contract)) for p in postprocessors if p.contract]

    loss_contract_str = ""
    if isinstance(loss_calculator, CompositeLoss):
        for calc, _ in loss_calculator.losses:
            if calc.contract:
                loss_contract_str = calc.contract
                break
    else:
        loss_contract_str = loss_calculator.contract

    loss_c = parse_contract(loss_contract_str) if loss_contract_str else None

    if model_c and loss_c:
        errors = validate_pipeline_chain(
            [c for _, c in pre_contracts],
            model_c,
            [c for _, c in post_contracts],
            loss_c,
        )
        if errors:
            print("    Status: FAILED")
            for err in errors:
                print(f"      ✗ {err}")
        else:
            print("    Status: PASSED")
            print("    All tensor shapes and dtypes chain correctly through the pipeline.")
    else:
        print("    Status: SKIPPED (model or loss contract not specified)")

    print("\n" + "=" * width + "\n")


# --- Example Data ---


def create_example_dataset(num_samples: int = 1000, seq_len: int = 32, vocab_size: int = 1000):
    """Create a synthetic dataset for demonstration."""
    return Dataset.from_dict({
        "input_ids": [
            torch.randint(1, vocab_size, (seq_len,)).tolist()
            for _ in range(num_samples)
        ]
    })


def example_collate_fn(batch):
    """Collate function that stacks input_ids."""
    return torch.tensor([item["input_ids"] for item in batch])


# --- Main Example ---


def run_example():
    """
    Complete end-to-end training example with contracts.

    Flow:
        Dataset -> DataLoader
            -> TokenMasker (masks random tokens, stores originals in ctx)
            -> ForwardKwargsInjector (adds return_hidden=True)
            -> model.forward(batch, **forward_kwargs)
            -> TargetExtractor (moves original tokens to ctx["target"])
            -> HiddenStateExtractor (extracts hidden states to ctx)
            -> CompositeLoss([MaskedCrossEntropyLoss, HiddenRegularizer])
            -> backward + optimize
    """
    vocab_size = 1000
    embed_dim = 256
    hidden_dim = 512

    dataset = create_example_dataset(num_samples=1000, seq_len=32, vocab_size=vocab_size)

    model = ExampleModel(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim)

    preprocessors = [
        TokenMasker(
            mask_prob=0.15,
            mask_token_id=0,
        ),
        ForwardKwargsInjector(
            contract="batch:2:i64 -> batch:2:i64",
            return_hidden=True,
        ),
    ]

    postprocessors = [
        TargetExtractor(
            target_key="original_input_ids",
            contract="logits:3:f, hidden:3:f -> logits:3:f, hidden:3:f",
        ),
        HiddenStateExtractor(
            output_index=1,
            contract="logits:3:f, hidden:3:f -> logits:3:f, hidden:3:f",
        ),
    ]

    loss_calculator = CompositeLoss([
        (MaskedCrossEntropyLoss(
            vocab_size=vocab_size,
            contract="logits:3:f | ctx.target:2:i64, ctx.mask:2:f -> loss:0:f",
        ), 1.0),
        (HiddenRegularizer(
            weight=0.01,
            contract="| ctx.hidden_states:3:f -> loss:0:f",
        ), 1.0),
    ])

    model_contract = "batch:2:i64 -> logits:3:f, hidden:3:f"

    print_pipeline(model, preprocessors, postprocessors, loss_calculator, model_contract)

    model_factory = ModelFactory.from_instance(model)

    pipeline = Pipeline(
        model_factory=model_factory,
        dataset=dataset,
        collate_fn=example_collate_fn,
        loss_calculator=loss_calculator,
        preprocessors=preprocessors,
        postprocessors=postprocessors,
        output_dir="./example_output",
        batch_size=16,
        epochs=1,
        lr=1e-3,
        fp16=False,
        checkpoint_every=100,
        log_every=10,
    )

    pipeline.run()


def run_example_with_fsdp():
    """Example with FSDP for distributed training."""
    vocab_size = 1000
    embed_dim = 256
    hidden_dim = 512

    dataset = create_example_dataset(num_samples=1000, seq_len=32, vocab_size=vocab_size)
    model = ExampleModel(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim)

    fsdp_config = FSDPConfig(
        enabled=True,
        transformer_layers=[nn.Linear],
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    preprocessors = [TokenMasker(mask_prob=0.15)]
    postprocessors = [TargetExtractor()]
    loss = MaskedCrossEntropyLoss(vocab_size=vocab_size)

    print_pipeline(model, preprocessors, postprocessors, loss)

    model_factory = ModelFactory.from_instance(model, fsdp_config=fsdp_config)

    pipeline = Pipeline(
        model_factory=model_factory,
        dataset=dataset,
        collate_fn=example_collate_fn,
        loss_calculator=loss,
        preprocessors=preprocessors,
        postprocessors=postprocessors,
        output_dir="./example_fsdp_output",
        batch_size=16,
        epochs=1,
        use_bfloat16=True,
    )

    pipeline.run()


def run_example_with_transform():
    """Example with model transform to freeze layers."""
    vocab_size = 1000
    dataset = create_example_dataset(num_samples=500, seq_len=32, vocab_size=vocab_size)

    model = ExampleModel(vocab_size=vocab_size)
    transform = FreezeEmbeddings()
    transformed_model = transform.transform(model)

    preprocessors = [TokenMasker(mask_prob=0.15)]
    postprocessors = [TargetExtractor()]
    loss = MaskedCrossEntropyLoss(vocab_size=vocab_size)

    print("\n[Before Transform]")
    print_pipeline(model, preprocessors, postprocessors, loss)

    print("\n[After Transform - Embeddings Frozen]")
    print_pipeline(transformed_model, preprocessors, postprocessors, loss)

    model_factory = ModelFactory.from_instance(transformed_model)

    pipeline = Pipeline(
        model_factory=model_factory,
        dataset=dataset,
        collate_fn=example_collate_fn,
        loss_calculator=loss,
        preprocessors=preprocessors,
        postprocessors=postprocessors,
        output_dir="./example_frozen_output",
        batch_size=16,
        epochs=1,
    )

    pipeline.run()


def demo_contracts():
    """Demonstrate contract parsing and validation without running training."""
    print("\n" + "=" * 70)
    print("CONTRACT SYSTEM DEMO")
    print("=" * 70)

    print("\n[1] Parsing individual contracts:")
    examples = [
        "tokens:2:i64 -> logits:3:f",
        "logits:3:f, hidden:3:f -> logits:3:f, hidden:3:f",
        "logits:3:f | ctx.target:2:i64, ctx.mask:2:f -> loss:0:f",
    ]
    for ex in examples:
        c = parse_contract(ex)
        print(f"\n  Input: '{ex}'")
        print(f"  Parsed: {c}")
        print(f"    Inputs: {[str(i) for i in c.inputs]}")
        print(f"    Outputs: {[str(o) for o in c.outputs]}")
        print(f"    Ctx deps: {[str(d) for d in c.ctx_deps]}")

    print("\n\n[2] Validating a complete pipeline:")

    pre = [
        parse_contract("batch:2:i64 -> batch:2:i64"),
    ]
    model = parse_contract("batch:2:i64 -> logits:3:f, hidden:3:f")
    post = [
        parse_contract("logits:3:f, hidden:3:f -> logits:3:f, hidden:3:f"),
    ]
    loss = parse_contract("logits:3:f | ctx.target:2:i64 -> loss:0:f")

    errors = validate_pipeline_chain(pre, model, post, loss)
    if errors:
        print("  Validation errors:")
        for e in errors:
            print(f"    - {e}")
    else:
        print("  Pipeline contracts are valid!")

    print("\n\n[3] Dtype shorthand reference:")
    print("  f    - any float (f16, bf16, f32, f64)")
    print("  f16  - float16")
    print("  f32  - float32")
    print("  bf16 - bfloat16")
    print("  i    - any int (i8, i16, i32, i64)")
    print("  i64  - int64")
    print("  i32  - int32")
    print("  b    - bool")
    print("  u8   - uint8")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo_contracts()
    run_example()
