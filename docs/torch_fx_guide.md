# torch.fx Guide for Credit Assignment

## Overview

`torch.fx` is PyTorch's symbolic tracing toolkit that captures the computation graph of an `nn.Module`. Unlike the module tree (which is hierarchical), the FX graph represents the actual dataflow DAG - including skip connections.

## Why FX for PropOp?

The module tree doesn't capture skip connections:
```python
# Module tree sees:
BasicBlock
├── conv1, bn1, relu, conv2, bn2
└── downsample (Sequential)

# But forward() does:
out = conv1 -> bn1 -> relu -> conv2 -> bn2
identity = downsample(x)  # or just x
out = out + identity  # <-- This addition isn't a module!
```

FX tracing captures the `+` as a `call_function` node, revealing the actual DAG.

## Basic Usage

```python
import torch.fx as fx
from torchvision.models import resnet18

model = resnet18()
traced = fx.symbolic_trace(model)

# traced is a GraphModule (subclass of nn.Module)
# It has a .graph attribute with the computation graph
```

## Graph Structure

The graph is a list of `Node` objects. Each node has:

| Attribute | Description |
|-----------|-------------|
| `op` | Operation type: `placeholder`, `call_module`, `call_function`, `call_method`, `get_attr`, `output` |
| `name` | Unique identifier for this node |
| `target` | What to call (module path, function, method name) |
| `args` | Positional arguments (other Nodes or constants) |
| `kwargs` | Keyword arguments |
| `users` | Dict of nodes that use this node's output |

### Op Types

| Op | Meaning | Example |
|----|---------|---------|
| `placeholder` | Input to the graph | `x` |
| `call_module` | Call to a submodule | `self.conv1(x)` |
| `call_function` | Call to a function | `operator.add`, `torch.relu` |
| `call_method` | Call to a method | `x.view(...)` |
| `get_attr` | Access a module attribute | `self.weight` |
| `output` | Return value | `return x` |

## Example: ResNet Block

```python
# For a BasicBlock with downsample:
def forward(self, x):
    identity = self.downsample(x)
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = out + identity  # Skip connection
    out = self.relu(out)
    return out
```

FX produces this graph:
```
placeholder     x
call_module     downsample_0         inputs=[x]
call_module     downsample_1         inputs=[downsample_0]
call_module     conv1                inputs=[x]
call_module     bn1                  inputs=[conv1]
call_module     relu                 inputs=[bn1]
call_module     conv2                inputs=[relu]
call_module     bn2                  inputs=[conv2]
call_function   add                  inputs=[bn2, downsample_1]  # <-- MERGE POINT
call_module     relu_1               inputs=[add]
output          output               inputs=[relu_1]
```

The `add` node with `target=operator.add` is where the skip connection merges.

## Building Backward Credit Routing

For credit assignment, we need to route credit backward through the graph:
1. Start at `output` node
2. For each node, send credit to its predecessors (nodes in `args`)
3. For `add` nodes: send the **same credit** to **both inputs** (credit branches)

```python
import operator

def build_predecessor_map(graph):
    """Map each node to its input nodes."""
    predecessors = {}
    for node in graph.nodes:
        preds = [arg for arg in node.args if isinstance(arg, fx.Node)]
        predecessors[node] = preds
    return predecessors

def find_add_nodes(graph):
    """Find all addition nodes (skip connection merge points)."""
    return [n for n in graph.nodes
            if n.op == 'call_function' and n.target == operator.add]

# Backward credit traversal
def backward_credit(traced, output_credit):
    predecessors = build_predecessor_map(traced.graph)
    credit = {}  # node.name -> credit tensor

    # Start from output
    for node in reversed(list(traced.graph.nodes)):
        if node.op == 'output':
            # Get credit from the node that feeds into output
            credit[node.args[0].name] = output_credit

        elif node.op == 'call_function' and node.target == operator.add:
            # ADD: both inputs get the same credit (branching)
            node_credit = credit.get(node.name)
            for pred in predecessors[node]:
                credit[pred.name] = node_credit  # Same credit to both!

        elif node.op == 'call_module':
            # Apply layer-specific backward (credit @ W.T, etc.)
            node_credit = credit.get(node.name)
            module = traced.get_submodule(node.target)
            upstream_credit = backward_through_module(module, node_credit)
            for pred in predecessors[node]:
                credit[pred.name] = upstream_credit
```

## Accessing Modules

```python
traced = fx.symbolic_trace(model)

# For a call_module node with target='layer1.0.conv1'
node_target = 'layer1.0.conv1'
module = traced.get_submodule(node_target)  # Returns the Conv2d

# Check module type
isinstance(module, nn.Conv2d)  # True

# Hooks still work
module.register_forward_hook(my_hook)
```

## Forward Hooks on Traced Model

The `GraphModule` is still an `nn.Module`, so hooks work normally:

```python
traced = fx.symbolic_trace(model)

activations = {}
def make_hook(name):
    def hook(module, input, output):
        activations[name] = output
    return hook

# Register on submodules
for node in traced.graph.nodes:
    if node.op == 'call_module':
        module = traced.get_submodule(node.target)
        module.register_forward_hook(make_hook(node.name))

# Run forward - hooks fire as normal
output = traced(input_tensor)
```

## Limitations

1. **Data-dependent control flow**: FX cannot trace `if x.sum() > 0` because it depends on tensor values
2. **Dynamic shapes in control flow**: `for i in range(x.size(0))` fails
3. **Non-torch operations**: Some custom Python code may not trace

For models with these issues, alternatives:
- `torch.compile` with `fullgraph=False` (falls back to eager for untraceable parts)
- Manual annotation with `@torch.fx.wrap` for leaf functions

## ResNet-18 Graph Stats

For `torchvision.models.resnet18`:
- 60 `call_module` nodes
- 8 `call_function` nodes (all are `operator.add` for skip connections + 1 flatten)
- 1 `placeholder` (input)
- 1 `output`

## Key Insight for PropOp

When credit flows backward and hits an `add` node:
```
      credit
         ↓
       [add]
      ↙     ↘
  [bn2]     [downsample_1]
```

Both `bn2` and `downsample_1` receive the **full credit** (it branches, not splits).
This matches the gradient behavior: ∂L/∂(a+b) contributes equally to ∂L/∂a and ∂L/∂b.

## Sources

- [PyTorch FX Documentation](https://docs.pytorch.org/docs/stable/fx.html)
- [torch.fx README](https://github.com/pytorch/pytorch/blob/main/torch/fx/README.md)
