"""
PropOpWrapper: Wraps any nn.Module with PropOp local learning.
"""
from typing import List, Optional, Dict, Type, Callable
from enum import Enum, auto

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck

from .config import PropOpConfig
from .state import LayerState


class LayerKind(Enum):
    """Categories of layers for PropOp processing."""
    WEIGHT = auto()      # Has learnable weights (Linear, Conv2d)
    ACTIVATION = auto()  # Nonlinearity (ReLU, GELU, etc.)
    POOL = auto()        # Spatial pooling
    NORM = auto()        # Normalization layers
    RESHAPE = auto()     # Shape manipulation (Flatten)
    RESIDUAL = auto()    # Residual block with skip connection
    UNKNOWN = auto()     # Unrecognized - pass through


# Registry mapping layer types to their kind
LAYER_REGISTRY: Dict[Type[nn.Module], LayerKind] = {
    # Weight layers
    nn.Linear: LayerKind.WEIGHT,
    nn.Conv2d: LayerKind.WEIGHT,
    nn.Conv1d: LayerKind.WEIGHT,
    # Activations
    nn.ReLU: LayerKind.ACTIVATION,
    nn.LeakyReLU: LayerKind.ACTIVATION,
    nn.GELU: LayerKind.ACTIVATION,
    nn.SiLU: LayerKind.ACTIVATION,
    nn.Mish: LayerKind.ACTIVATION,
    nn.Sigmoid: LayerKind.ACTIVATION,
    nn.Tanh: LayerKind.ACTIVATION,
    nn.Softplus: LayerKind.ACTIVATION,
    nn.ELU: LayerKind.ACTIVATION,
    # Pooling
    nn.MaxPool2d: LayerKind.POOL,
    nn.MaxPool1d: LayerKind.POOL,
    nn.AvgPool2d: LayerKind.POOL,
    nn.AvgPool1d: LayerKind.POOL,
    nn.AdaptiveAvgPool2d: LayerKind.POOL,
    nn.AdaptiveMaxPool2d: LayerKind.POOL,
    # Normalization
    nn.BatchNorm1d: LayerKind.NORM,
    nn.BatchNorm2d: LayerKind.NORM,
    nn.LayerNorm: LayerKind.NORM,
    nn.GroupNorm: LayerKind.NORM,
    nn.InstanceNorm2d: LayerKind.NORM,
    # Reshape
    nn.Flatten: LayerKind.RESHAPE,
    # Residual blocks
    BasicBlock: LayerKind.RESIDUAL,
    Bottleneck: LayerKind.RESIDUAL,
}


def get_layer_kind(module: nn.Module) -> LayerKind:
    """Look up layer kind from registry."""
    return LAYER_REGISTRY.get(type(module), LayerKind.UNKNOWN)


# =============================================================================
# Credit backward functions for each layer type
# Signature: (module, state, downstream_credit) -> upstream_credit
# =============================================================================

def _credit_linear(module: nn.Linear, state: LayerState, credit: torch.Tensor) -> torch.Tensor:
    return credit @ module.weight


def _credit_conv2d(module: nn.Conv2d, state: LayerState, credit: torch.Tensor) -> torch.Tensor:
    # Handle output_padding for strided convs to get correct input size
    output_padding = 0
    if module.stride[0] > 1 and state.input_cache is not None:
        input_size = state.input_cache.shape[2]
        output_size = credit.shape[2]
        expected_input = (output_size - 1) * module.stride[0] - 2 * module.padding[0] + module.dilation[0] * (module.kernel_size[0] - 1) + 1
        output_padding = input_size - expected_input

    return F.conv_transpose2d(
        credit,
        module.weight,
        stride=module.stride,
        padding=module.padding,
        output_padding=output_padding,
        groups=module.groups,
        dilation=module.dilation,
    )


def _credit_conv1d(module: nn.Conv1d, state: LayerState, credit: torch.Tensor) -> torch.Tensor:
    # Handle output_padding for strided convs to get correct input size
    output_padding = 0
    if module.stride[0] > 1 and state.input_cache is not None:
        input_size = state.input_cache.shape[2]
        output_size = credit.shape[2]
        expected_input = (output_size - 1) * module.stride[0] - 2 * module.padding[0] + module.dilation[0] * (module.kernel_size[0] - 1) + 1
        output_padding = input_size - expected_input

    return F.conv_transpose1d(
        credit,
        module.weight,
        stride=module.stride,
        padding=module.padding,
        output_padding=output_padding,
        groups=module.groups,
        dilation=module.dilation,
    )


def _credit_relu(module: nn.Module, state: LayerState, credit: torch.Tensor) -> torch.Tensor:
    # Gate by output > 0 (matches main.py H_active = H_batch > 0)
    return credit * (state.output_cache > 0).float()


def _credit_leaky_relu(module: nn.LeakyReLU, state: LayerState, credit: torch.Tensor) -> torch.Tensor:
    mask = (state.input_cache > 0).float()
    return credit * (mask + (1 - mask) * module.negative_slope)


def _credit_sigmoid(module: nn.Module, state: LayerState, credit: torch.Tensor) -> torch.Tensor:
    sig = torch.sigmoid(state.input_cache)
    return credit * sig * (1 - sig)


def _credit_tanh(module: nn.Module, state: LayerState, credit: torch.Tensor) -> torch.Tensor:
    tanh = torch.tanh(state.input_cache)
    return credit * (1 - tanh ** 2)


def _credit_gelu(module: nn.Module, state: LayerState, credit: torch.Tensor) -> torch.Tensor:
    # Approximate: derivative is ~1 where output > 0
    return credit * (state.output_cache > 0).float()


def _credit_silu(module: nn.Module, state: LayerState, credit: torch.Tensor) -> torch.Tensor:
    # SiLU(x) = x * sigmoid(x), derivative = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    sig = torch.sigmoid(state.input_cache)
    return credit * (sig + state.input_cache * sig * (1 - sig))


def _credit_softplus(module: nn.Module, state: LayerState, credit: torch.Tensor) -> torch.Tensor:
    return credit * torch.sigmoid(state.input_cache)


def _credit_elu(module: nn.ELU, state: LayerState, credit: torch.Tensor) -> torch.Tensor:
    mask = (state.input_cache > 0).float()
    return credit * (mask + (1 - mask) * (state.output_cache + module.alpha))


def _credit_flatten(module: nn.Module, state: LayerState, credit: torch.Tensor) -> torch.Tensor:
    return credit.view(state.input_cache.shape)


def _credit_maxpool2d(module: nn.MaxPool2d, state: LayerState, credit: torch.Tensor) -> torch.Tensor:
    # Route credit only to max positions using stored indices
    if state.pool_indices is not None:
        input_size = state.input_cache.shape
        return F.max_unpool2d(
            credit,
            state.pool_indices,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            output_size=input_size,
        )
    # Fallback to uniform distribution if indices not available
    input_size = state.input_cache.shape[2:]
    scale = (input_size[0] / credit.shape[2]) * (input_size[1] / credit.shape[3])
    return F.interpolate(credit, size=input_size, mode='nearest') * scale


def _credit_maxpool1d(module: nn.MaxPool1d, state: LayerState, credit: torch.Tensor) -> torch.Tensor:
    # Route credit only to max positions using stored indices
    if state.pool_indices is not None:
        input_size = state.input_cache.shape
        return F.max_unpool1d(
            credit,
            state.pool_indices,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            output_size=input_size,
        )
    # Fallback to uniform distribution if indices not available
    input_size = state.input_cache.shape[2:]
    scale = input_size[0] / credit.shape[2]
    return F.interpolate(credit, size=input_size, mode='nearest') * scale


def _credit_avgpool2d(module: nn.Module, state: LayerState, credit: torch.Tensor) -> torch.Tensor:
    # AvgPool distributes credit uniformly
    input_size = state.input_cache.shape[2:]

    # Handle case where credit is 2D (from Linear after Flatten)
    # Reshape to 4D with spatial dims of 1
    if credit.dim() == 2:
        credit = credit.unsqueeze(-1).unsqueeze(-1)

    output_size = credit.shape[2:]
    scale = (input_size[0] / output_size[0]) * (input_size[1] / output_size[1])
    return F.interpolate(credit, size=input_size, mode='nearest') / scale


def _credit_avgpool1d(module: nn.Module, state: LayerState, credit: torch.Tensor) -> torch.Tensor:
    # AvgPool distributes credit uniformly (correct behavior)
    input_size = state.input_cache.shape[2:]
    output_size = credit.shape[2:]
    scale = input_size[0] / output_size[0]
    return F.interpolate(credit, size=input_size, mode='nearest') / scale


def _credit_passthrough(module: nn.Module, state: LayerState, credit: torch.Tensor) -> torch.Tensor:
    return credit


def _credit_batchnorm1d(module: nn.BatchNorm1d, state: LayerState, credit: torch.Tensor) -> torch.Tensor:
    # Full credit propagation through BatchNorm1d including off-diagonal coupling
    x = state.input_cache
    gamma = module.weight

    if module.training:
        # Training mode: use batch statistics (with off-diagonal coupling)
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, keepdim=True, unbiased=False)
        std = torch.sqrt(var + module.eps)
        x_hat = (x - mean) / std

        # Credit through gamma scaling
        credit_xhat = credit * gamma.view(1, -1)

        # Credit through normalization (includes off-diagonal coupling)
        credit_x = (credit_xhat - credit_xhat.mean(dim=0, keepdim=True)
                    - x_hat * (credit_xhat * x_hat).mean(dim=0, keepdim=True)) / std
    else:
        # Eval mode: use running stats (diagonal only, no coupling)
        std = torch.sqrt(module.running_var + module.eps)
        # Simple diagonal scaling: gamma / std
        credit_x = credit * (gamma / std).view(1, -1)

    return credit_x


def _credit_batchnorm2d(module: nn.BatchNorm2d, state: LayerState, credit: torch.Tensor) -> torch.Tensor:
    # Full credit propagation through BatchNorm2d including off-diagonal coupling
    x = state.input_cache
    gamma = module.weight

    if module.training:
        # Training mode: use batch statistics (with off-diagonal coupling)
        mean = x.mean(dim=[0, 2, 3], keepdim=True)
        var = x.var(dim=[0, 2, 3], keepdim=True, unbiased=False)
        std = torch.sqrt(var + module.eps)
        x_hat = (x - mean) / std

        # Credit through gamma scaling
        credit_xhat = credit * gamma.view(1, -1, 1, 1)

        # Credit through normalization (includes off-diagonal coupling)
        credit_x = (credit_xhat - credit_xhat.mean(dim=[0, 2, 3], keepdim=True)
                    - x_hat * (credit_xhat * x_hat).mean(dim=[0, 2, 3], keepdim=True)) / std
    else:
        # Eval mode: use running stats (diagonal only, no coupling)
        std = torch.sqrt(module.running_var + module.eps)
        # Simple diagonal scaling: gamma / std
        credit_x = credit * (gamma / std).view(1, -1, 1, 1)

    return credit_x


# Registry mapping layer types to their credit backward function
CREDIT_BACKWARD_REGISTRY: Dict[Type[nn.Module], Callable] = {
    # Weight layers
    nn.Linear: _credit_linear,
    nn.Conv2d: _credit_conv2d,
    nn.Conv1d: _credit_conv1d,
    # Activations
    nn.ReLU: _credit_relu,
    nn.LeakyReLU: _credit_leaky_relu,
    nn.GELU: _credit_gelu,
    nn.SiLU: _credit_silu,
    nn.Mish: _credit_gelu,  # Approximate same as GELU
    nn.Sigmoid: _credit_sigmoid,
    nn.Tanh: _credit_tanh,
    nn.Softplus: _credit_softplus,
    nn.ELU: _credit_elu,
    # Pooling
    nn.MaxPool2d: _credit_maxpool2d,
    nn.MaxPool1d: _credit_maxpool1d,
    nn.AvgPool2d: _credit_avgpool2d,
    nn.AvgPool1d: _credit_avgpool1d,
    nn.AdaptiveAvgPool2d: _credit_avgpool2d,
    nn.AdaptiveMaxPool2d: _credit_avgpool2d,  # Adaptive doesn't have indices
    # Normalization (full credit propagation with off-diagonal coupling)
    nn.BatchNorm1d: _credit_batchnorm1d,
    nn.BatchNorm2d: _credit_batchnorm2d,
    nn.LayerNorm: _credit_passthrough,  # TODO: proper LayerNorm credit
    nn.GroupNorm: _credit_passthrough,  # TODO: proper GroupNorm credit
    nn.InstanceNorm2d: _credit_passthrough,
    # Reshape
    nn.Flatten: _credit_flatten,
}


class PropOpWrapper(nn.Module):
    """
    Wraps any nn.Module with PropOp local learning.

    Uses forward hooks to cache activations during forward pass,
    then applies local credit assignment and weight updates in learn().
    Does NOT reimplement the forward pass - delegates to wrapped model.
    """

    def __init__(self, model: nn.Module, config: PropOpConfig, debug_file=None):
        super().__init__()
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        self.debug_file = debug_file
        self.sample_counter = 0
        self.log_interval = 128  # Log every N samples
        self._last_batch_loss = 0.0

        # Discover and wrap layers
        self.layer_states: List[LayerState] = []
        # Depth + offset tracking for nested residual blocks
        # _depth_offsets[d] = starting offset in layer_states at depth d
        self._depth = 0
        self._depth_offsets: List[int] = [0]  # offset at depth 0 is 0
        self._current_block_state: Optional[LayerState] = None  # Block we're currently inside
        self._wire_up(model)

    def _wire_up(self, model: nn.Module):
        """Build DAG from FX trace if possible, fallback to tree walk."""
        import torch.fx as fx

        try:
            traced = fx.symbolic_trace(model)
            self._wire_up_fx(traced)
            print(f"FX wiring: {len(self.layer_states)} states, {len(getattr(self, 'merge_points', {}))} merge points")
        except Exception as e:
            print(f"FX trace failed ({e}), using tree walk")
            self._wire_up_tree(model)

    def _wire_up_fx(self, traced):
        """Build LayerState DAG from FX traced graph.

        Iterates FX nodes in topological order, creating states and linking predecessors.
        Merge points (operator.add) are tracked so credit branches properly.
        """
        import operator
        import torch.fx as fx

        self.merge_points = {}  # add_node_name -> [input_node_names]
        node_to_state: Dict[str, LayerState] = {}  # fx node name -> LayerState
        self._traced = traced  # Keep reference for credit backward

        last_weight_state = None

        for node in traced.graph.nodes:
            if node.op == 'placeholder':
                # Input node - no state, but track for predecessor lookup
                pass

            elif node.op == 'call_function' and node.target == operator.add:
                # Merge point: record inputs, next module will get both as predecessors
                inputs = [arg.name for arg in node.args if isinstance(arg, fx.Node)]
                self.merge_points[node.name] = inputs
                # Create a virtual state to track the merge output
                # (needed so next node can reference it)
                node_to_state[node.name] = None  # Placeholder - credit will branch here

            elif node.op == 'call_function' or node.op == 'call_method':
                # Non-add function calls (flatten, view, etc.) - pass through predecessor
                # Find the first input node that has a state and use that
                inputs = [arg.name for arg in node.args if isinstance(arg, fx.Node)]
                for inp in inputs:
                    if inp in node_to_state and node_to_state[inp] is not None:
                        node_to_state[node.name] = node_to_state[inp]
                        break
                    elif inp in self.merge_points:
                        # Point to one of the merge inputs (arbitrary - credit flows both ways)
                        for merge_input in self.merge_points[inp]:
                            if merge_input in node_to_state and node_to_state[merge_input] is not None:
                                node_to_state[node.name] = node_to_state[merge_input]
                                break
                        if node.name in node_to_state:
                            break

            elif node.op == 'call_module':
                module = traced.get_submodule(node.target)
                kind = get_layer_kind(module)

                # Get predecessor states from args
                pred_names = [arg.name for arg in node.args if isinstance(arg, fx.Node)]
                predecessors = []
                for pname in pred_names:
                    if pname in self.merge_points:
                        # This node follows a merge - get both merge inputs as predecessors
                        for merge_input in self.merge_points[pname]:
                            if merge_input in node_to_state and node_to_state[merge_input] is not None:
                                predecessors.append(node_to_state[merge_input])
                    elif pname in node_to_state and node_to_state[pname] is not None:
                        predecessors.append(node_to_state[pname])

                # Create appropriate state
                if kind == LayerKind.WEIGHT:
                    state = self._create_weight_layer_state(module)
                    last_weight_state = state
                else:
                    state = LayerState(module=module, device=self.device)

                # Link predecessors
                state.predecessors = predecessors if predecessors else None

                # Enable return_indices for MaxPool layers
                if isinstance(module, (nn.MaxPool2d, nn.MaxPool1d)):
                    module.return_indices = True

                # Register hooks
                module.register_forward_hook(self._make_cache_hook(state))

                # Link weight layer to following activation for post_activation_state
                if kind == LayerKind.ACTIVATION and last_weight_state is not None:
                    last_weight_state.post_activation_state = state
                    last_weight_state = None

                # Track state
                node_to_state[node.name] = state
                self.layer_states.append(state)

            elif node.op == 'output':
                # Mark the output state (last state before output)
                output_args = [arg.name for arg in node.args if isinstance(arg, fx.Node)]
                if output_args and output_args[0] in node_to_state:
                    self._output_state = node_to_state[output_args[0]]

        # Store node mapping for credit backward
        self._node_to_state = node_to_state

    def _wire_up_tree(self, module: nn.Module):
        """Fallback: Walk model tree, register hooks, create LayerState for each layer.

        Uses depth + offset tracking for residual blocks:
        - When entering a block: increment depth, store offset, recurse
        - Child states get collected into block's child_states list
        - When exiting: restore depth, children are accessible via block.child_states
        """
        last_weight_state = None
        prev_module = None
        prev_state = None

        for child in module.children():
            kind = get_layer_kind(child)

            if kind == LayerKind.UNKNOWN:
                # Recurse into containers (Sequential, ModuleList, etc.)
                self._wire_up_tree(child)
            elif kind == LayerKind.RESIDUAL:
                # Residual block: cache input/output for skip credit, then recurse
                state = LayerState(module=child, device=self.device, child_states=[])
                self.layer_states.append(state)
                child.register_forward_hook(self._make_cache_hook(state))

                # Enter block: increment depth, track offset
                self._depth += 1
                self._depth_offsets.append(len(self.layer_states))  # Children start here
                parent_block = self._current_block_state
                self._current_block_state = state

                # Recurse to wire up internal layers (conv, bn, relu, etc.)
                self._wire_up_tree(child)

                # Exit block: restore depth, children are now in state.child_states
                self._depth -= 1
                self._depth_offsets.pop()
                self._current_block_state = parent_block

            elif kind == LayerKind.WEIGHT:
                state = self._create_weight_layer_state(child)
                self._add_state(state)
                child.register_forward_hook(self._make_cache_hook(state))
                last_weight_state = state
                prev_module = child
                prev_state = state
            elif kind == LayerKind.ACTIVATION:
                # Register theta hook on the layer immediately before activation
                # This handles: Linear -> BatchNorm -> (theta) -> ReLU
                # as well as: Linear -> (theta) -> ReLU
                if prev_module is not None and last_weight_state is not None and last_weight_state.theta is not None:
                    prev_module.register_forward_hook(self._make_theta_hook(last_weight_state))

                # Cache activations
                state = LayerState(module=child, device=self.device)
                self._add_state(state)
                child.register_forward_hook(self._make_cache_hook(state))

                # Link preceding weight layer to this activation
                if last_weight_state is not None:
                    last_weight_state.post_activation_state = state
                    last_weight_state = None  # Don't link multiple activations to same weight

                prev_module = child
                prev_state = state
            else:
                # All other tracked layers (Norm, Pool, etc.): just cache, no learning state
                # prev_module is updated so theta hook goes on THIS layer (e.g., BatchNorm)
                # when we see the following activation
                state = LayerState(module=child, device=self.device)
                self._add_state(state)

                # Enable return_indices for MaxPool layers to capture max positions
                if isinstance(child, (nn.MaxPool2d, nn.MaxPool1d)):
                    child.return_indices = True

                child.register_forward_hook(self._make_cache_hook(state))
                prev_module = child
                prev_state = state

    def _add_state(self, state: LayerState):
        """Add state to appropriate container based on nesting depth.

        When inside a residual block (depth > 0), add to block's child_states only.
        At top level (depth == 0), add to flat layer_states.
        """
        if self._current_block_state is not None:
            # Inside a block: add to block's children only
            self._current_block_state.child_states.append(state)
        else:
            # Top level: add to flat list
            self.layer_states.append(state)

    def _iter_all_states(self, states: List[LayerState] = None):
        """Iterate over all states including nested children (depth-first)."""
        if states is None:
            states = self.layer_states
        for state in states:
            yield state
            if state.child_states:
                yield from self._iter_all_states(state.child_states)

    def _get_all_weight_states(self) -> List[LayerState]:
        """Get all weight-bearing layer states (including nested)."""
        return [s for s in self._iter_all_states() if s.eligibility is not None]

    def _create_weight_layer_state(self, module: nn.Module) -> LayerState:
        """Initialize learning state for weight-bearing layers."""
        if isinstance(module, nn.Linear):
            out_dim = module.out_features
        else:  # Conv2d
            out_dim = module.out_channels

        # Lateral strength scales with 1/out_dim (matches main.py)
        lateral_strength = 1.0 / out_dim

        return LayerState(
            module=module,
            device=self.device,
            cofire=torch.zeros((out_dim, out_dim), device=self.device),
            lateral=torch.zeros((out_dim, out_dim), device=self.device),
            theta=torch.zeros(out_dim, device=self.device),
            firing_rate=torch.ones(out_dim, device=self.device) * self.config.target_fire,
            eligibility=torch.zeros_like(module.weight.data),
            lateral_strength=lateral_strength,
            # Per-layer learnable echo params (start from global config)
            echo_tau=self.config.echo_tau_init,
            echo_strength=self.config.echo_strength_init,
            credit_variance_ema=0.0,
        )

    def _make_cache_hook(self, state: LayerState):
        """Hook that caches input/output during forward pass."""
        def hook(module, inputs, output):
            state.input_cache = inputs[0].detach()
            # Handle MaxPool with return_indices=True (returns tuple)
            if isinstance(output, tuple):
                state.output_cache = output[0].detach()
                state.pool_indices = output[1].detach()
            else:
                state.output_cache = output.detach()
                state.pool_indices = None
        return hook

    def _make_theta_hook(self, weight_state: LayerState):
        """Hook that subtracts theta and applies cofire modulation (before activation)."""
        def hook(module, inputs, output):
            # Apply theta (homeostatic threshold)
            if self.config.use_theta and weight_state.theta is not None:
                if output.dim() == 4:
                    # Conv: theta is (C,), broadcast to (1, C, 1, 1)
                    output = output - weight_state.theta.view(1, -1, 1, 1)
                else:
                    output = output - weight_state.theta

            # Apply cofire forward modulation (coalition excitation)
            if self.config.use_cofire and self.config.cofire_forward and weight_state.cofire is not None:
                gamma = 20.0
                # h_active based on pre-activation > 0 (will fire after ReLU)
                if output.dim() == 4:
                    # Conv: pool spatially for cofire lookup, then broadcast back
                    h_pooled = output.mean(dim=(2, 3))  # (N, C)
                    h_active = (h_pooled > 0).float()
                    h_drive = gamma * (h_active @ weight_state.cofire.T)  # (N, C)
                    output = output + h_drive.unsqueeze(-1).unsqueeze(-1)
                else:
                    h_active = (output > 0).float()
                    h_drive = gamma * (h_active @ weight_state.cofire.T)
                    output = output + h_drive

            return output
        return hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through wrapped model. Hooks cache activations."""
        with torch.no_grad():
            return self.model(x)

    def begin_batch(self):
        """Snapshot frozen state for batch-consistent lateral computation."""
        for state in self._iter_all_states():
            if state.cofire is not None:
                state.cofire_pre = state.cofire.clone()
                state.lateral_pre = state.lateral.clone()

    @torch.no_grad()
    def learn(self, output: torch.Tensor, y: torch.Tensor, losses: torch.Tensor = None):
        """Compute credit assignment and apply weight updates.

        Args:
            output: Model logits (N, num_classes)
            y: Target labels (N,)
            losses: Per-sample losses for error-gain (optional)
        """
        n = len(y)
        self.sample_counter += n
        if losses is not None:
            self._last_batch_loss = losses.mean().item()

        # Output error (cross-entropy gradient)
        probs = F.softmax(output, dim=-1)
        delta = -probs.clone()
        delta[torch.arange(n, device=self.device), y] += 1.0

        # Find weight layers (have eligibility) - includes nested
        weight_states = self._get_all_weight_states()
        if not weight_states:
            return

        # Identify output layer (last weight layer) and hidden layers
        output_state = weight_states[-1]
        hidden_states = weight_states[:-1]

        # Error gain for hidden layers (per-sample learning rate boost)
        if losses is not None:
            batch_mean_loss = losses.mean()
            loss_diff = losses - batch_mean_loss
            error_gains = loss_diff.clamp(0.0, 2.0)
            hidden_lrs = self.config.lr * 0.25 * (1.0 + error_gains)  # hidden_lr_scale = 0.25
        else:
            hidden_lrs = torch.full((n,), self.config.lr * 0.25, device=self.device)

        # === PHASE 1: Update output layer (simple, no activity gating/lateral) ===
        input_to_output = output_state.input_cache  # H2 in 3-layer

        # Normalize input
        input_norms = input_to_output.norm(dim=1, keepdim=True) + 1e-8
        input_normed = input_to_output / input_norms

        # W3 update: lr_head * (H2_normed.T @ Delta)
        # PyTorch Linear weight is (out, in), so delta is (out, in) = einsum('bo,bi->oi', delta, input)
        output_delta = torch.einsum('bi,bo->oi', input_normed, delta)
        output_state.module.weight.data += self.config.lr * output_delta

        # === PHASE 2: Backward credit assignment ===
        # Credit flows backward through all layers with activation gating:
        #   Credit2 = (Delta @ W3.T) * H2_active
        #   Credit1 = (Credit2 @ W2.T) * H1_active
        # For residual blocks: credit branches at output, merges at input
        credit = delta
        credit = self._backward_credit_recursive(self.layer_states, credit)

        # === PHASE 3: Update hidden layers (output to input) ===
        # Credit was computed once in Phase 2, no recompute needed
        for state in reversed(hidden_states):
            self._apply_hidden_weight_update(state, hidden_lrs)

    @torch.no_grad()
    def end_batch(self):
        """EMA updates for cofire, lateral, theta, firing_rate."""
        # Store weight norms before updates for debug
        weight_norms_before = {}
        weight_states = self._get_all_weight_states()
        for idx, state in enumerate(weight_states):
            weight_norms_before[f'W{idx+1}'] = state.module.weight.data.norm().item()

        for state in self._iter_all_states():
            if state.cofire is not None:
                # Find post-activation output for this weight layer
                post_activation = self._get_post_activation(state)
                self._update_cofire(state, post_activation)
                if self.config.use_lateral:
                    self._update_lateral(state, post_activation)
                if self.config.use_theta:
                    self._update_theta(state)
                self._update_firing_rate(state, post_activation)

                # Adaptive echo: adjust per-layer echo_tau based on credit variance
                if self.config.use_echo and state.credit_variance_ema is not None:
                    self._adapt_echo_params(state)

        # Debug logging
        if self.debug_file is not None and self.sample_counter % self.log_interval == 0:
            self._write_debug_log(weight_norms_before, weight_states)

    def _get_post_activation(self, state: LayerState) -> torch.Tensor:
        """Get the post-activation output for a weight layer using linked reference."""
        if state.post_activation_state is not None:
            return state.post_activation_state.output_cache
        # Fallback to weight layer's output (shouldn't happen in normal networks)
        return state.output_cache

    def _backward_credit_recursive(self, states: List[LayerState], credit: torch.Tensor) -> torch.Tensor:
        """Backward credit through a list of states, handling nested residual blocks.

        For FX-wired graphs: uses predecessors field for DAG traversal.
        For tree-wired graphs: uses child_states for residual block handling.

        For residual blocks: credit BRANCHES at block output, MERGES at block input.
        skip_credit = credit (at block output)
        deep_credit = credit through children
        credit_to_input = deep_credit + skip_credit
        """
        # Check if we have FX-based predecessors (DAG mode)
        if hasattr(self, '_node_to_state') and self._node_to_state:
            return self._backward_credit_dag(credit)

        # Fallback: tree-based traversal with child_states
        for state in reversed(states):
            # Check if this is a residual block with children
            if state.child_states:
                # Save skip credit (credit at block output, before going through children)
                skip_credit = credit.clone()

                # Process children recursively (deep path)
                credit = self._backward_credit_recursive(state.child_states, credit)

                # Merge: add skip credit to credit after children processed
                credit = credit + skip_credit
                credit = credit.clamp(-self.config.credit_clip, self.config.credit_clip)

                # Store on block state
                state.credit = skip_credit  # Block's credit is at its output
                continue

            # Apply echo for weight layers (adds persistent baseline to credit)
            if self.config.use_echo and state.eligibility is not None:
                credit = self._apply_echo(state, credit)

            state.credit = credit
            credit = self._backward_credit(state, credit)
            # Clip credit to prevent explosion through deep networks
            credit = credit.clamp(-self.config.credit_clip, self.config.credit_clip)

        return credit

    def _backward_credit_dag(self, credit: torch.Tensor) -> torch.Tensor:
        """Backward credit through DAG using predecessors (FX-wired graphs).

        Credit propagates backward from output state through predecessors.
        At merge points (state with multiple predecessors), credit BRANCHES:
        the same credit is sent to ALL predecessors.
        """
        # Start from output state
        if not hasattr(self, '_output_state') or self._output_state is None:
            # Fallback: use last state
            if self.layer_states:
                self._output_state = self.layer_states[-1]
            else:
                return credit

        # BFS backward through predecessors
        # Use dict to track credit at each state (handles merge points)
        state_credit: Dict[int, torch.Tensor] = {}
        visited = set()

        # Initialize with output state
        state_credit[id(self._output_state)] = credit

        # Process states in reverse topological order (reverse of layer_states)
        for state in reversed(self.layer_states):
            state_id = id(state)

            # Get credit for this state (may have been set by successor)
            if state_id not in state_credit:
                continue

            current_credit = state_credit[state_id]

            # Apply echo for weight layers
            if self.config.use_echo and state.eligibility is not None:
                current_credit = self._apply_echo(state, current_credit)

            # Store credit on state
            state.credit = current_credit

            # Backward through this layer
            upstream_credit = self._backward_credit(state, current_credit)
            upstream_credit = upstream_credit.clamp(-self.config.credit_clip, self.config.credit_clip)

            # Propagate to predecessors (credit BRANCHES at merge points)
            if state.predecessors:
                for pred in state.predecessors:
                    pred_id = id(pred)
                    if pred_id in state_credit:
                        # Merge: add credit from multiple successors
                        state_credit[pred_id] = state_credit[pred_id] + upstream_credit
                        state_credit[pred_id] = state_credit[pred_id].clamp(
                            -self.config.credit_clip, self.config.credit_clip
                        )
                    else:
                        state_credit[pred_id] = upstream_credit

        return credit

    def _apply_echo(self, state: LayerState, credit: torch.Tensor) -> torch.Tensor:
        """Apply credit echo (dopamine-like persistence) to a weight layer's credit."""
        # Use per-layer echo params (fall back to config if not set)
        layer_echo_tau = state.echo_tau if state.echo_tau is not None else self.config.echo_tau_init
        layer_echo_strength = state.echo_strength if state.echo_strength is not None else self.config.echo_strength_init

        # Batch-average credit, preserving spatial dims for convs
        # Linear: (batch, features) -> (features,)
        # Conv: (batch, C, H, W) -> (C, H, W)
        credit_mean = credit.mean(dim=0)

        # Track credit variance for adaptive echo (before echo boost)
        credit_var = credit.var(dim=0).mean().item()
        var_ema_tau = 0.9
        if state.credit_variance_ema is None:
            state.credit_variance_ema = credit_var
        else:
            state.credit_variance_ema = var_ema_tau * state.credit_variance_ema + (1 - var_ema_tau) * credit_var

        # Lazy init echo on first use (shape matches credit_mean)
        if state.credit_echo is None:
            state.credit_echo = torch.zeros_like(credit_mean)

        # STORE credit BEFORE echo boost for A/B comparison
        state._credit_without_echo = credit.clone()

        # Add PREVIOUS echo to credit BEFORE updating (ease-in: first batch gets no boost)
        credit = credit + layer_echo_strength * state.credit_echo.unsqueeze(0)

        # STORE credit AFTER echo boost for A/B comparison
        state._credit_with_echo = credit.clone()

        # Track the echo contribution magnitude
        state._echo_contribution = (layer_echo_strength * state.credit_echo).norm().item()
        state._echo_contribution_mean = (layer_echo_strength * state.credit_echo).mean().item()

        # Sign-flip reset: accumulate if same sign, reset if sign flips
        # But if echo is near-zero (fresh), always accumulate to bootstrap
        echo_is_fresh = state.credit_echo.abs() < 1e-8
        sign_match = (credit_mean * state.credit_echo) > 0
        should_accumulate = sign_match | echo_is_fresh
        state._echo_reset_frac = (~should_accumulate).float().mean().item()

        state.credit_echo = torch.where(
            should_accumulate,
            layer_echo_tau * state.credit_echo + (1 - layer_echo_tau) * credit_mean,
            torch.zeros_like(state.credit_echo)
        )

        return credit

    def _backward_credit(self, state: LayerState, downstream_credit: torch.Tensor) -> torch.Tensor:
        """Route credit backwards through this layer using dispatch registry."""
        module = state.module
        module_type = type(module)

        # Look up backward function from registry
        backward_fn = CREDIT_BACKWARD_REGISTRY.get(module_type)
        if backward_fn is not None:
            return backward_fn(module, state, downstream_credit)

        # Fallback: pass through unchanged
        return downstream_credit

    def _apply_hidden_weight_update(self, state: LayerState, hidden_lrs: torch.Tensor):
        """Compute and apply hidden layer weight update with per-sample learning rates."""
        module = state.module
        credit = state.credit
        input_data = state.input_cache

        # For activity gating, we need POST-activation values (after ReLU), not pre-activation
        # The reference uses H1/H2 which are post-ReLU: H1 = np.maximum(0.0, H1_pre)
        # Use linked post_activation_state if available, else fallback to pre-activation
        if state.post_activation_state is not None:
            post_activation = state.post_activation_state.output_cache
        else:
            post_activation = state.output_cache  # fallback to pre-activation

        if isinstance(module, nn.Linear):
            delta = self._compute_linear_update(state, input_data, post_activation, credit, hidden_lrs)
        elif isinstance(module, nn.Conv2d):
            delta = self._compute_conv_update(state, input_data, post_activation, credit, hidden_lrs)
        else:
            return

        # Apply update directly (no eligibility trace for now - main.py doesn't use it)
        module.weight.data += delta

        # Apply norm banding if configured
        if self.config.norm_band is not None:
            self._apply_norm_band(module, state)

    def _compute_linear_update(
        self,
        state: LayerState,
        input_data: torch.Tensor,
        output: torch.Tensor,
        credit: torch.Tensor,
        hidden_lrs: torch.Tensor,
    ) -> torch.Tensor:
        """Local learning rule for Linear layer with per-sample learning rates.

        Matches main.py end_batch() W1/W2 update logic:
        - Normalize input
        - Activity gating: update_scalars = hidden_lr * activity * credit
        - Lateral push subtraction
        """
        # Normalize input (optional)
        if self.config.normalize_updates:
            input_norms = input_data.norm(dim=1, keepdim=True) + 1e-8
            input_data = input_data / input_norms

        # Activity gating: update_scalars = hidden_lr * activity * credit
        if self.config.activity_gate:
            activity = output / (output.max(dim=1, keepdim=True).values + 1e-8)
            update_scalars = hidden_lrs.unsqueeze(1) * activity * credit
        else:
            update_scalars = hidden_lrs.unsqueeze(1) * credit

        # Weight update: outer product summed over batch
        # W is (out, in), so we need einsum to produce (out, in)
        delta = torch.einsum('bi,bo->oi', input_data, update_scalars)

        # Apply lateral push if enabled
        if self.config.use_cofire and self.config.use_lateral and state.lateral_pre is not None:
            delta = self._apply_lateral_push(state, delta, output, hidden_lrs.mean())

        return delta

    def _compute_conv_update(
        self,
        state: LayerState,
        input_data: torch.Tensor,
        output: torch.Tensor,
        credit: torch.Tensor,
        hidden_lrs: torch.Tensor,
    ) -> torch.Tensor:
        """Local learning rule for Conv2d layer with per-sample learning rates."""
        conv = state.module

        # Unfold input into patches: (N, C_in*kH*kW, H'*W')
        input_unfolded = F.unfold(
            input_data,
            kernel_size=conv.kernel_size,
            padding=conv.padding,
            stride=conv.stride,
        )

        # Normalize across spatial positions (dim=2)
        if self.config.normalize_updates:
            input_unfolded = F.normalize(input_unfolded, dim=2)

        # Flatten credit spatially: (N, C_out, H'*W')
        credit_flat = credit.view(credit.shape[0], credit.shape[1], -1)

        # Activity gating for conv (optional)
        if self.config.activity_gate:
            output_flat = output.view(output.shape[0], output.shape[1], -1)
            activity = output_flat / (output_flat.max(dim=2, keepdim=True).values + 1e-8)
            update_scalars = hidden_lrs.view(-1, 1, 1) * activity * credit_flat
        else:
            update_scalars = hidden_lrs.view(-1, 1, 1) * credit_flat

        # Weight update: einsum over batch and spatial
        # delta[c_out, c_in*kH*kW] = sum over n,s of update_scalars[n,c_out,s] * input[n,c_in*kH*kW,s]
        delta = torch.einsum('nos,nis->oi', update_scalars, input_unfolded)

        # Normalize by spatial positions (conv sums over more terms than linear)
        num_spatial = input_unfolded.shape[2]
        delta = delta / num_spatial

        # Reshape to conv weight shape: (C_out, C_in, kH, kW)
        delta = delta.view(conv.weight.shape)
        return delta

    def _apply_lateral_push(
        self,
        state: LayerState,
        delta: torch.Tensor,
        output: torch.Tensor,
        mean_hidden_lr: float = None,
    ) -> torch.Tensor:
        """Apply lateral inhibition push to weight updates.

        Reference (numpy layout, W is (in, out)):
            W_col_u = W / col_norms  # unit columns
            weighted_lateral = mean_h[:, None] * lateral  # (out, out)
            lateral_push = W_col_u @ weighted_lateral  # (in, out) @ (out, out) = (in, out)

        PyTorch layout (W is (out, in)):
            W_u = W / row_norms  # unit rows = unit columns in transposed view
            result_pytorch = (W_u.T @ weighted_lateral).T = weighted_lateral.T @ W_u
        """
        module = state.module
        lateral_pre = state.lateral_pre

        if lateral_pre is None:
            return delta

        W = module.weight.data  # (out, in) in PyTorch

        if isinstance(module, nn.Linear):
            row_norms = W.norm(dim=1, keepdim=True) + 1e-8
            W_u = W / row_norms
        else:
            W_2d = W.view(W.shape[0], -1)
            row_norms = W_2d.norm(dim=1, keepdim=True) + 1e-8
            W_u = W_2d / row_norms

        if output.dim() == 4:
            mean_h = output.mean(dim=(0, 2, 3))
        else:
            mean_h = output.mean(dim=0)

        # weighted_lateral = mean_h[:, None] * lateral in reference
        # Need weighted_lateral.T @ W_u for correct PyTorch equivalent
        weighted_lateral = mean_h.unsqueeze(1) * lateral_pre  # (out, out)
        lateral_push = weighted_lateral.T @ W_u  # (out, out) @ (out, in) = (out, in)

        if isinstance(module, nn.Conv2d):
            lateral_push = lateral_push.view(delta.shape)

        n = output.shape[0]
        if mean_hidden_lr is None:
            mean_hidden_lr = self.config.lr * 0.25
        # Use per-layer lateral_strength (1/out_dim) instead of global config
        delta = delta - n * mean_hidden_lr * self.config.lam * state.lateral_strength * lateral_push

        return delta

    def _update_cofire(self, state: LayerState, post_activation: torch.Tensor):
        """Update cofire matrix from credit signals using post-activation values."""
        if not self.config.use_cofire:
            return

        credit = state.credit
        output = post_activation  # Use post-activation (after ReLU), not pre-activation

        # Pool if conv layer
        if credit.dim() == 4:
            credit = credit.mean(dim=(2, 3))
        if output.dim() == 4:
            output = output.mean(dim=(2, 3))

        # Activity normalization (A1 = H1 / H1_max in reference)
        output_max = output.max(dim=1, keepdim=True).values + 1e-8
        activity = output / output_max

        # Cofire term: activity * sign(credit) * |credit|
        cofire_term = activity * torch.sign(credit) * torch.abs(credit)
        batch_cofire = cofire_term.T @ cofire_term
        batch_cofire.fill_diagonal_(0.0)

        # EMA update (tau adjusted for batch size)
        n = credit.shape[0]
        tau = self.config.cofire_tau ** n
        state.cofire = tau * state.cofire + (1 - tau) * (batch_cofire / n)
        state.cofire.clamp_(-1.0, 1.0)
        state.cofire.fill_diagonal_(0.0)  # Clear diagonal after EMA (matches reference line 750)

    def _update_lateral(self, state: LayerState, post_activation: torch.Tensor):
        """Update lateral inhibition from credit sign disagreement using post-activation values."""
        if not self.config.use_lateral or state.cofire_pre is None:
            return

        credit = state.credit
        output = post_activation  # Use post-activation (after ReLU), not pre-activation

        if credit.dim() == 4:
            credit = credit.mean(dim=(2, 3))
        if output.dim() == 4:
            output = output.mean(dim=(2, 3))

        n = credit.shape[0]
        credit_sign = torch.sign(credit)

        # Sign agreement matrix
        sign_prod = credit_sign.T @ credit_sign
        disagree_frac = (n - sign_prod) / (2.0 * n + 1e-8)
        disagree_frac.clamp_(0, 1)

        # Activity correlation (using post-activation values)
        activity = output / (output.max(dim=1, keepdim=True).values + 1e-8)
        both_active = activity.T @ activity

        # Lateral grows where: cofire positive AND neurons disagree
        cofire_pos = torch.maximum(state.cofire_pre, torch.zeros_like(state.cofire_pre))
        batch_lateral = (both_active / n) * disagree_frac * cofire_pos * n
        batch_lateral.fill_diagonal_(0.0)  # Clear diagonal before EMA (matches reference line 592)

        # EMA update with decay
        tau = self.config.lateral_tau ** n
        decay = self.config.lateral_decay ** n
        state.lateral = tau * state.lateral + (1 - tau) * (batch_lateral / n)
        state.lateral *= decay
        state.lateral.fill_diagonal_(0.0)

    def _update_theta(self, state: LayerState):
        """Update homeostatic thresholds based on post-activation firing."""
        if not self.config.use_theta or state.theta is None:
            return

        # Use linked post_activation_state, fallback to weight layer's output
        post_activation = self._get_post_activation(state)

        if post_activation.dim() == 4:
            post_activation = post_activation.mean(dim=(2, 3))

        active = (post_activation > 0).float()
        n = post_activation.shape[0]

        # Delta proportional to (actual - target) firing rate
        theta_delta = self.config.theta_lr * (active.sum(dim=0) - n * self.config.target_fire) / n
        state.theta += theta_delta
        state.theta.clamp_(-self.config.theta_clip, self.config.theta_clip)

    def _update_firing_rate(self, state: LayerState, post_activation: torch.Tensor):
        """Update firing rate EMA for homeostasis tracking."""
        if state.firing_rate is None:
            return

        output = post_activation
        if output.dim() == 4:
            output = output.mean(dim=(2, 3))

        n = output.shape[0]
        active = (output > 0).float()
        avg_active = active.mean(dim=0)

        # EMA update with tau adjusted for batch size
        tau = self.config.firing_rate_tau ** n
        state.firing_rate = tau * state.firing_rate + (1 - tau) * avg_active

    def _adapt_echo_params(self, state: LayerState):
        """Adapt per-layer echo params based on credit variance.

        High variance → increase tau (more smoothing) and strength
        Low variance → decrease tau (more responsive) and strength
        """
        if state.echo_tau is None or state.credit_variance_ema is None:
            return

        # Target variance - if actual variance is higher, increase smoothing
        target_var = 0.1
        adapt_lr = 0.01  # Slow adaptation

        var_ratio = state.credit_variance_ema / (target_var + 1e-8)

        # Adapt tau: high variance → higher tau (more smoothing)
        # var_ratio > 1 means variance is high → nudge tau up
        # var_ratio < 1 means variance is low → nudge tau down
        tau_delta = adapt_lr * (var_ratio - 1.0)
        tau_delta = max(-0.01, min(0.01, tau_delta))  # Clamp adjustment
        state.echo_tau = max(0.1, min(0.99, state.echo_tau + tau_delta))

        # Adapt strength: high variance → stronger echo contribution
        strength_delta = adapt_lr * 0.5 * (var_ratio - 1.0)
        strength_delta = max(-0.005, min(0.005, strength_delta))
        state.echo_strength = max(0.01, min(0.3, state.echo_strength + strength_delta))

    def _write_debug_log(self, weight_norms_before: dict, weight_states: list):
        """Write debug log entry in format compatible with visualize_debug.py."""
        import json

        # Get first hidden layer's cofire for stats
        hidden_states = weight_states[:-1]  # Exclude output layer
        if not hidden_states:
            return

        state1 = hidden_states[0]
        cofire = state1.cofire

        # Get post-activation for sparsity
        post_act = self._get_post_activation(state1)
        if post_act.dim() == 4:
            post_act = post_act.mean(dim=(2, 3))
        active1 = (post_act > 0).float()
        avg_active1 = active1.mean(dim=0)
        sparsity = 1.0 - avg_active1.mean().item()
        n_active = (avg_active1 > 0.5).sum().item()

        # Weight norms after
        weight_norms_after = {}
        for idx, state in enumerate(weight_states):
            weight_norms_after[f'W{idx+1}'] = state.module.weight.data.norm().item()

        debug_entry = {
            "sample": self.sample_counter,
            "loss": self._last_batch_loss,
            "signal": 0.0,
            "sparsity": sparsity,
            "n_active": int(n_active),
            "n_strong_aligned": 0,
            "n_weak_aligned": 0,
            "cofire_min": cofire.min().item() if cofire is not None else 0,
            "cofire_max": cofire.max().item() if cofire is not None else 0,
            "cofire_mean": cofire.mean().item() if cofire is not None else 0,
            "cofire_std": cofire.std().item() if cofire is not None else 0,
            "n_clusters": 1,
            "cluster_sizes": {},
            "clusters": {},
            "total_positive_update": 0.0,
            "total_negative_update": 0.0,
            "net_update": 0.0,
            "W1_norm_before": weight_norms_before.get('W1', 0),
            "W1_norm_after": weight_norms_after.get('W1', 0),
            "W2_norm_before": weight_norms_before.get('W2', 0),
            "W2_norm_after": weight_norms_after.get('W2', 0),
        }

        # Add layer 2 stats if 3-layer
        if len(hidden_states) > 1:
            state2 = hidden_states[1]
            cofire2 = state2.cofire
            post_act2 = self._get_post_activation(state2)
            if post_act2.dim() == 4:
                post_act2 = post_act2.mean(dim=(2, 3))
            active2 = (post_act2 > 0).float()
            avg_active2 = active2.mean(dim=0)

            debug_entry["is_3_layer"] = True
            debug_entry["W3_norm"] = weight_norms_after.get('W3', 0)
            debug_entry["hidden2_n_active"] = int((avg_active2 > 0.5).sum().item())
            debug_entry["hidden2_sparsity"] = 1.0 - avg_active2.mean().item()
            debug_entry["n_clusters2"] = 1
            debug_entry["cluster_sizes2"] = {}
            debug_entry["clusters2"] = {}

            # Add lateral stats for debugging
            if state1.lateral is not None:
                debug_entry["lateral1_min"] = state1.lateral.min().item()
                debug_entry["lateral1_max"] = state1.lateral.max().item()
                debug_entry["lateral1_mean"] = state1.lateral.mean().item()
            if state2.lateral is not None:
                debug_entry["lateral2_min"] = state2.lateral.min().item()
                debug_entry["lateral2_max"] = state2.lateral.max().item()
                debug_entry["lateral2_mean"] = state2.lateral.mean().item()

        # Echo A/B comparison stats (if echo enabled)
        if self.config.use_echo:
            for idx, state in enumerate(weight_states):
                key = f"echo{idx+1}"

                # Echo state
                if state.credit_echo is not None:
                    debug_entry[f"{key}_norm"] = state.credit_echo.norm().item()
                    debug_entry[f"{key}_mean"] = state.credit_echo.mean().item()
                    debug_entry[f"{key}_min"] = state.credit_echo.min().item()
                    debug_entry[f"{key}_max"] = state.credit_echo.max().item()

                # Echo contribution (what echo added to credit)
                if hasattr(state, '_echo_contribution'):
                    debug_entry[f"{key}_contrib_norm"] = state._echo_contribution
                    debug_entry[f"{key}_contrib_mean"] = state._echo_contribution_mean

                # Credit comparison
                if hasattr(state, '_credit_without_echo') and hasattr(state, '_credit_with_echo'):
                    debug_entry[f"{key}_credit_without_norm"] = state._credit_without_echo.norm().item()
                    debug_entry[f"{key}_credit_with_norm"] = state._credit_with_echo.norm().item()
                    debug_entry[f"{key}_credit_boost_ratio"] = (
                        state._credit_with_echo.norm().item() /
                        (state._credit_without_echo.norm().item() + 1e-8)
                    )

                # Reset fraction
                if hasattr(state, '_echo_reset_frac'):
                    debug_entry[f"{key}_reset_frac"] = state._echo_reset_frac

                # Per-layer adaptive echo params
                if state.echo_tau is not None:
                    debug_entry[f"{key}_tau"] = state.echo_tau
                if state.echo_strength is not None:
                    debug_entry[f"{key}_strength"] = state.echo_strength
                if state.credit_variance_ema is not None:
                    debug_entry[f"{key}_credit_var"] = state.credit_variance_ema

        self.debug_file.write(json.dumps(debug_entry) + "\n")
        self.debug_file.flush()

    def _apply_norm_band(self, module: nn.Module, state: LayerState):
        """Keep column norms within band around target via gentle multiplicative rescaling."""
        band = self.config.norm_band
        if band is None:
            return

        W = module.weight.data
        norm_lr = self.config.norm_lr  # Gentle rescaling factor (default 0.01)

        if isinstance(module, nn.Linear):
            # For Linear: norms along input dimension (columns)
            col_norms = W.norm(dim=1) + 1e-8
            target = col_norms.mean()
        else:
            # For Conv2d: norms per output channel
            W_2d = W.view(W.shape[0], -1)
            col_norms = W_2d.norm(dim=1) + 1e-8
            target = col_norms.mean()

        low = target * (1.0 - band)
        high = target * (1.0 + band)

        # Gentle multiplicative rescaling (matches main.py)
        scale = torch.ones_like(col_norms)
        too_low = col_norms < low
        too_high = col_norms > high
        # Use power of norm_lr for gradual adjustment instead of hard clamping
        scale[too_low] = (low / col_norms[too_low]) ** norm_lr
        scale[too_high] = (high / col_norms[too_high]) ** norm_lr

        if isinstance(module, nn.Linear):
            module.weight.data *= scale.unsqueeze(1)
        else:
            module.weight.data *= scale.view(-1, 1, 1, 1)

    def state_dict(self):
        """Return full state including model weights and PropOp learning state."""
        state = {
            "model": self.model.state_dict(),
            "config": {
                "lr": self.config.lr,
                "beta": self.config.beta,
                "use_cofire": self.config.use_cofire,
                "use_lateral": self.config.use_lateral,
                "use_theta": self.config.use_theta,
                "cofire_forward": self.config.cofire_forward,
                "activity_gate": self.config.activity_gate,
                "normalize_updates": self.config.normalize_updates,
                "norm_band": self.config.norm_band,
                "cofire_tau": self.config.cofire_tau,
                "lateral_tau": self.config.lateral_tau,
                "lateral_decay": self.config.lateral_decay,
                "lateral_strength": self.config.lateral_strength,
                "lam": self.config.lam,
                "theta_lr": self.config.theta_lr,
                "theta_clip": self.config.theta_clip,
                "target_fire": self.config.target_fire,
                "firing_rate_tau": self.config.firing_rate_tau,
                "use_echo": self.config.use_echo,
                "echo_tau_init": self.config.echo_tau_init,
                "echo_strength_init": self.config.echo_strength_init,
            },
            "layer_states": [],
        }

        # Save learning state for each weight layer (including nested)
        for i, layer_state in enumerate(self._iter_all_states()):
            if layer_state.eligibility is not None:
                state["layer_states"].append({
                    "index": i,
                    "eligibility": layer_state.eligibility.cpu(),
                    "cofire": layer_state.cofire.cpu() if layer_state.cofire is not None else None,
                    "lateral": layer_state.lateral.cpu() if layer_state.lateral is not None else None,
                    "theta": layer_state.theta.cpu() if layer_state.theta is not None else None,
                    "firing_rate": layer_state.firing_rate.cpu() if layer_state.firing_rate is not None else None,
                    "credit_echo": layer_state.credit_echo.cpu() if layer_state.credit_echo is not None else None,
                })

        return state

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load model weights and PropOp learning state."""
        # Load model weights
        self.model.load_state_dict(state_dict["model"], strict=strict)

        # Build index -> state mapping for all states (including nested)
        all_states = list(self._iter_all_states())

        # Load learning state for weight layers
        for saved_state in state_dict.get("layer_states", []):
            idx = saved_state["index"]
            if idx >= len(all_states):
                continue  # Skip if index out of bounds
            layer_state = all_states[idx]

            if saved_state["eligibility"] is not None:
                layer_state.eligibility = saved_state["eligibility"].to(self.device)
            if saved_state["cofire"] is not None:
                layer_state.cofire = saved_state["cofire"].to(self.device)
            if saved_state["lateral"] is not None:
                layer_state.lateral = saved_state["lateral"].to(self.device)
            if saved_state["theta"] is not None:
                layer_state.theta = saved_state["theta"].to(self.device)
            if saved_state["firing_rate"] is not None:
                layer_state.firing_rate = saved_state["firing_rate"].to(self.device)
            if saved_state.get("credit_echo") is not None:
                layer_state.credit_echo = saved_state["credit_echo"].to(self.device)
