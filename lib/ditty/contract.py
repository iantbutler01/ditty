"""
Pipeline contract system for declarative tensor specifications.

Contracts use a terse syntax to declare tensor shapes and types:
    "name:rank:dtype, name:rank:dtype -> output:rank:dtype"

Examples:
    "tokens:3:i64 -> logits:4:f, z0_pred:4:f, z0_true:4:f, mask:3:b"
    "logits:4:f, z0_pred:4:f | ctx.input_ids:3:i64 -> loss:0:f"

Dtype shorthand (rust-style):
    f    - any float (f16, bf16, f32, f64)
    f16  - float16
    f32  - float32
    bf16 - bfloat16
    i    - any int (i8, i16, i32, i64)
    i64  - int64
    i32  - int32
    b    - bool
    u8   - uint8

Context dependencies use | separator:
    "input:3:f | ctx.target:4:f, ctx.mask:3:b -> output:4:f"
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Set
import re
import torch


DTYPE_MAP = {
    "f": (torch.float16, torch.float32, torch.bfloat16, torch.float64),
    "f16": (torch.float16,),
    "f32": (torch.float32,),
    "f64": (torch.float64,),
    "bf16": (torch.bfloat16,),
    "i": (torch.int8, torch.int16, torch.int32, torch.int64),
    "i8": (torch.int8,),
    "i16": (torch.int16,),
    "i32": (torch.int32,),
    "i64": (torch.int64,),
    "b": (torch.bool,),
    "u8": (torch.uint8,),
}


class ContractViolation(Exception):
    """Raised when a tensor doesn't match its declared contract."""
    pass


class ContractParseError(Exception):
    """Raised when a contract string is malformed."""
    pass


@dataclass
class TensorSpec:
    """Specification for a single tensor."""
    name: str
    rank: int
    dtype: Optional[str] = None  # None means any dtype
    is_ctx: bool = False  # True if this is a ctx.xxx reference

    def __str__(self) -> str:
        dtype_str = f":{self.dtype}" if self.dtype else ""
        prefix = "ctx." if self.is_ctx else ""
        return f"{prefix}{self.name}:{self.rank}{dtype_str}"

    def validate(self, tensor: torch.Tensor) -> None:
        """Validate a tensor against this spec. Raises ContractViolation on mismatch."""
        if tensor.ndim != self.rank:
            raise ContractViolation(
                f"{self.name}: expected rank {self.rank}, got {tensor.ndim} "
                f"(shape: {tuple(tensor.shape)})"
            )

        if self.dtype:
            valid_dtypes = DTYPE_MAP.get(self.dtype)
            if valid_dtypes is None:
                raise ContractViolation(f"{self.name}: unknown dtype spec '{self.dtype}'")
            if tensor.dtype not in valid_dtypes:
                expected = self.dtype if len(valid_dtypes) > 1 else valid_dtypes[0]
                raise ContractViolation(
                    f"{self.name}: expected dtype {expected}, got {tensor.dtype}"
                )


@dataclass
class Contract:
    """Parsed contract with inputs, outputs, and context dependencies."""
    inputs: List[TensorSpec]
    outputs: List[TensorSpec]
    ctx_deps: List[TensorSpec]  # ctx.xxx dependencies
    raw: str  # Original contract string

    def __str__(self) -> str:
        inputs_str = ", ".join(str(s) for s in self.inputs)
        outputs_str = ", ".join(str(s) for s in self.outputs)
        if self.ctx_deps:
            ctx_str = ", ".join(str(s) for s in self.ctx_deps)
            return f"{inputs_str} | {ctx_str} -> {outputs_str}"
        return f"{inputs_str} -> {outputs_str}"

    @property
    def input_names(self) -> Set[str]:
        return {s.name for s in self.inputs}

    @property
    def output_names(self) -> Set[str]:
        return {s.name for s in self.outputs}

    @property
    def ctx_names(self) -> Set[str]:
        return {s.name for s in self.ctx_deps}

    def validate_inputs(self, tensors: Tuple[Any, ...], ctx: Dict[str, Any]) -> None:
        """Validate input tensors and context against contract."""
        if len(tensors) != len(self.inputs):
            raise ContractViolation(
                f"Expected {len(self.inputs)} inputs, got {len(tensors)}"
            )

        for spec, tensor in zip(self.inputs, tensors):
            if not isinstance(tensor, torch.Tensor):
                raise ContractViolation(
                    f"{spec.name}: expected Tensor, got {type(tensor).__name__}"
                )
            spec.validate(tensor)

        for spec in self.ctx_deps:
            if spec.name not in ctx:
                raise ContractViolation(f"Missing ctx.{spec.name}")
            val = ctx[spec.name]
            if isinstance(val, torch.Tensor):
                spec.validate(val)

    def validate_outputs(self, tensors: Tuple[Any, ...]) -> None:
        """Validate output tensors against contract."""
        if len(tensors) != len(self.outputs):
            raise ContractViolation(
                f"Expected {len(self.outputs)} outputs, got {len(tensors)}"
            )

        for spec, tensor in zip(self.outputs, tensors):
            if not isinstance(tensor, torch.Tensor):
                raise ContractViolation(
                    f"{spec.name}: expected Tensor, got {type(tensor).__name__}"
                )
            spec.validate(tensor)


def parse_tensor_spec(spec_str: str) -> TensorSpec:
    """
    Parse a single tensor spec like "name:rank:dtype" or "ctx.name:rank:dtype".

    Examples:
        "logits:4:f" -> TensorSpec(name="logits", rank=4, dtype="f")
        "mask:3:b" -> TensorSpec(name="mask", rank=3, dtype="b")
        "tokens:3" -> TensorSpec(name="tokens", rank=3, dtype=None)
        "ctx.target:4:f" -> TensorSpec(name="target", rank=4, dtype="f", is_ctx=True)
    """
    spec_str = spec_str.strip()
    if not spec_str:
        raise ContractParseError("Empty tensor spec")

    is_ctx = spec_str.startswith("ctx.")
    if is_ctx:
        spec_str = spec_str[4:]  # Remove "ctx." prefix

    parts = spec_str.split(":")
    if len(parts) < 2:
        raise ContractParseError(
            f"Invalid tensor spec '{spec_str}': expected 'name:rank' or 'name:rank:dtype'"
        )

    name = parts[0].strip()
    if not name:
        raise ContractParseError(f"Empty name in tensor spec '{spec_str}'")

    try:
        rank = int(parts[1].strip())
    except ValueError:
        raise ContractParseError(
            f"Invalid rank '{parts[1]}' in tensor spec '{spec_str}': expected integer"
        )

    dtype = None
    if len(parts) >= 3:
        dtype = parts[2].strip()
        if dtype and dtype not in DTYPE_MAP:
            raise ContractParseError(
                f"Unknown dtype '{dtype}' in tensor spec '{spec_str}'. "
                f"Valid dtypes: {', '.join(DTYPE_MAP.keys())}"
            )

    return TensorSpec(name=name, rank=rank, dtype=dtype or None, is_ctx=is_ctx)


def parse_tensor_list(specs_str: str) -> Tuple[List[TensorSpec], List[TensorSpec]]:
    """
    Parse a comma-separated list of tensor specs, separating regular and ctx specs.

    Returns:
        (regular_specs, ctx_specs)
    """
    if not specs_str.strip():
        return [], []

    regular = []
    ctx = []

    for spec_str in specs_str.split(","):
        spec_str = spec_str.strip()
        if not spec_str:
            continue
        spec = parse_tensor_spec(spec_str)
        if spec.is_ctx:
            ctx.append(spec)
        else:
            regular.append(spec)

    return regular, ctx


def parse_contract(contract_str: str) -> Contract:
    """
    Parse a full contract string.

    Format: "inputs | ctx_deps -> outputs"

    Examples:
        "tokens:3:i64 -> logits:4:f"
        "logits:4:f, z0_pred:4:f | ctx.input_ids:3:i64 -> loss:0:f"
    """
    if not contract_str or not contract_str.strip():
        raise ContractParseError("Empty contract string")

    contract_str = contract_str.strip()

    # Split on ->
    if "->" not in contract_str:
        raise ContractParseError(
            f"Invalid contract '{contract_str}': missing '->' separator"
        )

    input_side, output_side = contract_str.split("->", 1)
    input_side = input_side.strip()
    output_side = output_side.strip()

    # Parse output side (no ctx deps allowed)
    outputs, output_ctx = parse_tensor_list(output_side)
    if output_ctx:
        raise ContractParseError(
            f"Invalid contract: ctx references not allowed in outputs"
        )

    # Split input side on | for ctx deps
    ctx_deps = []
    if "|" in input_side:
        main_inputs, ctx_side = input_side.split("|", 1)
        main_inputs = main_inputs.strip()
        ctx_side = ctx_side.strip()

        # Parse ctx deps - everything after | is a ctx dep
        regular_specs, ctx_specs = parse_tensor_list(ctx_side)
        # Mark all specs from the ctx side as ctx deps
        for spec in regular_specs:
            spec.is_ctx = True
        for spec in ctx_specs:
            spec.is_ctx = True
        ctx_deps = regular_specs + ctx_specs
    else:
        main_inputs = input_side

    # Parse main inputs
    inputs, input_ctx = parse_tensor_list(main_inputs)
    ctx_deps.extend(input_ctx)

    return Contract(
        inputs=inputs,
        outputs=outputs,
        ctx_deps=ctx_deps,
        raw=contract_str,
    )


def validate_pipeline_chain(
    preprocessor_contracts: List[Contract],
    model_contract: Contract,
    postprocessor_contracts: List[Contract],
    loss_contract: Contract,
) -> List[str]:
    """
    Validate that a pipeline's contracts chain together correctly.

    Returns list of validation errors (empty if valid).
    """
    errors = []

    # Track what's available in ctx
    ctx_available: Set[str] = set()

    # Track current tensor outputs (name -> TensorSpec)
    current_outputs: Dict[str, TensorSpec] = {}

    # Process preprocessors
    for i, contract in enumerate(preprocessor_contracts):
        # Check ctx deps are satisfied
        for ctx_dep in contract.ctx_deps:
            if ctx_dep.name not in ctx_available:
                errors.append(
                    f"Preprocessor {i}: requires ctx.{ctx_dep.name} but not available"
                )

        # Add outputs to ctx (preprocessors output to ctx)
        for out in contract.outputs:
            if out.is_ctx:
                ctx_available.add(out.name)
            else:
                current_outputs[out.name] = out

    # Check model inputs
    for inp in model_contract.inputs:
        # Model inputs come from preprocessor outputs or ctx
        if inp.name not in current_outputs and inp.name not in ctx_available:
            errors.append(
                f"Model: requires input '{inp.name}' but not provided by preprocessors"
            )
        elif inp.name in current_outputs:
            provided = current_outputs[inp.name]
            if provided.rank != inp.rank:
                errors.append(
                    f"Model input '{inp.name}': expected rank {inp.rank}, "
                    f"preprocessor provides rank {provided.rank}"
                )

    for ctx_dep in model_contract.ctx_deps:
        if ctx_dep.name not in ctx_available:
            errors.append(f"Model: requires ctx.{ctx_dep.name} but not available")

    # Model outputs become current outputs
    current_outputs = {out.name: out for out in model_contract.outputs}

    # Process postprocessors
    for i, contract in enumerate(postprocessor_contracts):
        for inp in contract.inputs:
            if inp.name not in current_outputs:
                errors.append(
                    f"Postprocessor {i}: requires input '{inp.name}' but not available"
                )
            else:
                provided = current_outputs[inp.name]
                if provided.rank != inp.rank:
                    errors.append(
                        f"Postprocessor {i} input '{inp.name}': expected rank {inp.rank}, "
                        f"got rank {provided.rank}"
                    )

        for ctx_dep in contract.ctx_deps:
            if ctx_dep.name not in ctx_available:
                errors.append(
                    f"Postprocessor {i}: requires ctx.{ctx_dep.name} but not available"
                )

        # Update current outputs
        current_outputs = {out.name: out for out in contract.outputs}
        # Postprocessors can also add to ctx
        for out in contract.outputs:
            if out.is_ctx:
                ctx_available.add(out.name)

    # Check loss calculator inputs
    for inp in loss_contract.inputs:
        if inp.name not in current_outputs:
            errors.append(
                f"LossCalculator: requires input '{inp.name}' but not available"
            )
        else:
            provided = current_outputs[inp.name]
            if provided.rank != inp.rank:
                errors.append(
                    f"LossCalculator input '{inp.name}': expected rank {inp.rank}, "
                    f"got rank {provided.rank}"
                )

    for ctx_dep in loss_contract.ctx_deps:
        if ctx_dep.name not in ctx_available:
            errors.append(
                f"LossCalculator: requires ctx.{ctx_dep.name} but not available"
            )

    return errors


def format_pipeline_contracts(
    preprocessor_contracts: List[Tuple[str, Contract]],
    model_contract: Tuple[str, Contract],
    postprocessor_contracts: List[Tuple[str, Contract]],
    loss_contract: Tuple[str, Contract],
) -> str:
    """Format pipeline contracts for display/debugging."""
    lines = ["Pipeline Contracts:", "=" * 60]

    if preprocessor_contracts:
        lines.append("\nPreprocessors:")
        for name, contract in preprocessor_contracts:
            lines.append(f"  {name}:")
            lines.append(f"    {contract}")

    lines.append(f"\nModel ({model_contract[0]}):")
    lines.append(f"  {model_contract[1]}")

    if postprocessor_contracts:
        lines.append("\nPostprocessors:")
        for name, contract in postprocessor_contracts:
            lines.append(f"  {name}:")
            lines.append(f"    {contract}")

    lines.append(f"\nLossCalculator ({loss_contract[0]}):")
    lines.append(f"  {loss_contract[1]}")

    lines.append("=" * 60)
    return "\n".join(lines)
