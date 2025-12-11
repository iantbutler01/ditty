"""
Hebbian IGOD: Inertial Gradient Optimization via Decomposition with Coherence Neighborhoods

An optimizer that:
1. Keeps inertia (belief) per parameter
2. Groups parameters into blocks (by index)
3. Learns which blocks "fire together" using gradient similarity (Hebbian rule)
4. Computes coherence at the neighborhood level for IGOD decomposition
5. Applies decomposed, coherence-scaled updates per parameter
"""

import math
import torch
from torch.optim import Optimizer
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field

try:
    from torchao.optim.subclass_8bit import OptimState8bit
    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False


@dataclass
class BlockState:
    """State for a single block."""
    g_ema: torch.Tensor  # Can be OptimState8bit or regular tensor
    neighbors: List[int] = field(default_factory=list)
    weights: Dict[int, float] = field(default_factory=dict)

    def get_g_ema_float(self) -> torch.Tensor:
        if hasattr(self.g_ema, 'dequantize'):
            return self.g_ema.dequantize()
        return self.g_ema.float()


class HebbianIGOD(Optimizer):
    """
    Hebbian IGOD: Inertial Gradient Optimization via Decomposition
    with Coherence Neighborhoods.

    Args:
        params: Model parameters
        lr: Learning rate (η)
        gamma: Inertia update rate (γ)
        alpha: Boost for confirming corrections (α ≥ 0)
        beta: Suppression for contradicting corrections (β ≥ 0)
        delta: Boost for orthogonal/new directions (δ ≥ 0)
        eps: Numerical stability constant (ε)
        rho: EMA rate for g_ema (ρ)
        lambda_w: EMA rate for Hebbian weights (λ_W)
        tau: Threshold for Hebbian weight to count as neighbor (τ)
        k: Max neighbors per block
        t_hebb: Frequency (in steps) to update Hebbian neighborhoods (T_hebb)
        block_size: Size of parameter blocks (B)
        use_8bit: Use 8-bit quantization for inertia state
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        gamma: float = 0.1,
        alpha: float = 0.0,
        beta: float = 1.0,
        delta: float = 0.0,
        eps: float = 1e-8,
        rho: float = 0.01,
        lambda_w: float = 0.01,
        tau: float = 0.5,
        k: int = 8,
        t_hebb: int = 100,
        block_size: int = 256,
        use_8bit: bool = False,
    ):
        if use_8bit and not TORCHAO_AVAILABLE:
            raise ImportError("torchao required for 8-bit mode: pip install torchao")

        defaults = dict(
            lr=lr,
            gamma=gamma,
            alpha=alpha,
            beta=beta,
            delta=delta,
            eps=eps,
            rho=rho,
            lambda_w=lambda_w,
            tau=tau,
            k=k,
            t_hebb=t_hebb,
            block_size=block_size,
            use_8bit=use_8bit,
        )
        super().__init__(params, defaults)

        self._block_registry: Dict[int, Tuple[int, int]] = {}
        self._block_states: Dict[int, BlockState] = {}
        self._param_blocks: Dict[int, List[int]] = {}
        self._param_names: Dict[int, str] = {}
        self._global_step = 0
        self._next_block_id = 0

        # Logging stats (collected during step, reset after get_stats)
        self._step_stats: Dict[str, list] = {
            'inertia_norms': [],
            'c_G_values': [],
            'neighbor_counts': [],
        }

    def _get_block_slice(self, block_idx: int, num_elements: int, block_size: int) -> Tuple[int, int]:
        start = block_idx * block_size
        end = min((block_idx + 1) * block_size, num_elements)
        return start, end

    def _init_param_state(self, p: torch.Tensor, param_id: int, group: dict):
        state = self.state[p]
        block_size = group['block_size']
        use_8bit = group['use_8bit']

        if use_8bit and p.numel() >= 4096:
            state['inertia'] = OptimState8bit.zeros(
                p.shape, signed=True, block_size=block_size, device=p.device
            )
        else:
            state['inertia'] = torch.zeros_like(p, memory_format=torch.preserve_format)

        num_elements = p.numel()
        num_blocks = math.ceil(num_elements / block_size)

        block_ids = []
        for local_idx in range(num_blocks):
            global_id = self._next_block_id
            self._next_block_id += 1

            self._block_registry[global_id] = (param_id, local_idx)
            block_ids.append(global_id)

            start, end = self._get_block_slice(local_idx, num_elements, block_size)
            block_len = end - start

            if use_8bit and block_len >= 256:
                g_ema = OptimState8bit.zeros((block_len,), signed=True, block_size=min(block_size, block_len), device=p.device)
            else:
                g_ema = torch.zeros(block_len, device=p.device, dtype=torch.float32)

            self._block_states[global_id] = BlockState(g_ema=g_ema)

        self._param_blocks[param_id] = block_ids
        state['param_id'] = param_id
        state['initialized'] = True

    def _update_hebbian_neighborhoods(self, group: dict):
        lambda_w = group['lambda_w']
        tau = group['tau']
        k = group['k']
        eps = group['eps']

        all_block_ids = list(self._block_states.keys())
        if len(all_block_ids) < 2:
            return

        for b in all_block_ids:
            b_state = self._block_states[b]
            g_b = b_state.get_g_ema_float()

            g_b_norm = torch.norm(g_b)
            if g_b_norm < eps:
                continue

            candidates = [j for j in all_block_ids if j != b]
            if len(candidates) > k * 4:
                indices = torch.randperm(len(candidates))[:k * 4].tolist()
                candidates = [candidates[i] for i in indices]

            for j in candidates:
                j_state = self._block_states[j]
                g_j = j_state.get_g_ema_float()

                g_j_norm = torch.norm(g_j)
                if g_j_norm < eps:
                    continue

                sim = torch.dot(g_b, g_j) / (g_b_norm * g_j_norm + eps)
                sim = sim.item()

                old_w = b_state.weights.get(j, 0.0)
                new_w = (1 - lambda_w) * old_w + lambda_w * sim
                b_state.weights[j] = new_w

            valid_neighbors = [(j, w) for j, w in b_state.weights.items() if w > tau]
            valid_neighbors.sort(key=lambda x: x[1], reverse=True)
            top_k = valid_neighbors[:k]

            b_state.neighbors = [j for j, w in top_k]
            b_state.weights = {j: w for j, w in top_k}

    def _compute_igod_update(
        self,
        g_flat: torch.Tensor,
        I_flat: torch.Tensor,
        block_ids: List[int],
        group: dict,
    ) -> torch.Tensor:
        lr = group['lr']
        alpha = group['alpha']
        beta = group['beta']
        delta = group['delta']
        eps = group['eps']
        block_size = group['block_size']

        num_elements = g_flat.numel()
        update = torch.zeros_like(g_flat)
        updated_mask = torch.zeros(num_elements, dtype=torch.bool, device=g_flat.device)

        for local_idx, global_id in enumerate(block_ids):
            start, end = self._get_block_slice(local_idx, num_elements, block_size)

            b_state = self._block_states[global_id]
            group_ids = [global_id] + b_state.neighbors

            g_parts = []
            I_parts = []

            for gid in group_ids:
                param_id, blk_idx = self._block_registry[gid]

                for p in group['params']:
                    p_state = self.state.get(p, {})
                    if p_state.get('param_id') == param_id:
                        p_g = p.grad.view(-1).float()
                        if hasattr(p_state['inertia'], 'dequantize'):
                            p_I = p_state['inertia'].dequantize().view(-1)
                        else:
                            p_I = p_state['inertia'].view(-1).float()

                        blk_start, blk_end = self._get_block_slice(
                            blk_idx, p_g.numel(), block_size
                        )
                        g_parts.append(p_g[blk_start:blk_end])
                        I_parts.append(p_I[blk_start:blk_end])
                        break

            if not g_parts:
                continue

            g_G = torch.cat(g_parts)
            I_G = torch.cat(I_parts)

            B_G = torch.norm(I_G).item()

            if B_G < eps:
                update[start:end] = -lr * g_flat[start:end]
                updated_mask[start:end] = True
                continue

            u_G = I_G / (B_G + eps)

            c_G = torch.dot(g_G, u_G).item()
            g_parallel = c_G * u_G
            g_orthogonal = g_G - g_parallel

            # Log c_G and neighbor count
            self._step_stats['c_G_values'].append(c_G)
            self._step_stats['neighbor_counts'].append(len(b_state.neighbors))

            if c_G > 0:
                s_parallel = 1.0 + alpha * B_G
            elif c_G < 0:
                s_parallel = 1.0 / (1.0 + beta * B_G)
            else:
                s_parallel = 1.0

            s_orthogonal = 1.0 + delta * B_G

            delta_G = -lr * (s_parallel * g_parallel + s_orthogonal * g_orthogonal)

            block_len = end - start
            block_update = delta_G[:block_len]

            update[start:end] = block_update
            updated_mask[start:end] = True

        if not updated_mask.all():
            update[~updated_mask] = -lr * g_flat[~updated_mask]

        return update

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._global_step += 1
        param_id_counter = 0

        for group in self.param_groups:
            gamma = group['gamma']
            rho = group['rho']
            block_size = group['block_size']
            t_hebb = group['t_hebb']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                if 'initialized' not in state:
                    self._init_param_state(p, param_id_counter, group)
                    param_id_counter += 1

                param_id = state['param_id']
                grad = p.grad.float()
                g_flat = grad.view(-1)

                I = state['inertia']
                if hasattr(I, 'dequantize'):
                    I_flat = I.dequantize().view(-1)
                else:
                    I_flat = I.view(-1).float()

                block_ids = self._param_blocks[param_id]
                num_elements = g_flat.numel()

                for local_idx, global_id in enumerate(block_ids):
                    start, end = self._get_block_slice(local_idx, num_elements, block_size)
                    g_block = g_flat[start:end]

                    b_state = self._block_states[global_id]
                    g_ema_f32 = b_state.get_g_ema_float()
                    new_g_ema = (1 - rho) * g_ema_f32 + rho * g_block
                    b_state.g_ema.copy_(new_g_ema)

                new_inertia = I_flat + gamma * g_flat
                I.copy_(new_inertia.view(I.shape))

                # Log inertia norm
                inertia_norm = torch.norm(new_inertia).item()
                param_name = self._param_names.get(param_id, f"param_{param_id}")
                self._step_stats['inertia_norms'].append((param_name, inertia_norm))

            if self._global_step % t_hebb == 0:
                self._update_hebbian_neighborhoods(group)

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                param_id = state['param_id']
                block_ids = self._param_blocks[param_id]

                grad = p.grad.float()
                g_flat = grad.view(-1)

                I = state['inertia']
                if hasattr(I, 'dequantize'):
                    I_flat = I.dequantize().view(-1)
                else:
                    I_flat = I.view(-1).float()

                update = self._compute_igod_update(g_flat, I_flat, block_ids, group)

                p.add_(update.view(p.shape).to(p.dtype))

        return loss

    def get_stats(self, reset: bool = True) -> Dict[str, any]:
        """
        Get logging statistics from the last step(s).

        Returns dict with:
            - inertia_norms: list of (param_name, norm) tuples
            - c_G_values: list of c_G scalars (confirmation vs contradiction)
            - c_G_mean: mean c_G (positive = confirming, negative = contradicting)
            - c_G_pos_ratio: fraction of blocks with c_G > 0
            - neighbor_counts: list of neighbor counts per block
            - avg_neighbors: average neighbors per block
            - total_blocks: total number of blocks
            - step: current global step
        """
        stats = {}

        # Inertia norms
        stats['inertia_norms'] = list(self._step_stats['inertia_norms'])

        # c_G distribution
        c_vals = self._step_stats['c_G_values']
        stats['c_G_values'] = list(c_vals)
        if c_vals:
            stats['c_G_mean'] = sum(c_vals) / len(c_vals)
            stats['c_G_pos_ratio'] = sum(1 for c in c_vals if c > 0) / len(c_vals)
        else:
            stats['c_G_mean'] = 0.0
            stats['c_G_pos_ratio'] = 0.0

        # Neighbor counts
        neighbor_counts = self._step_stats['neighbor_counts']
        stats['neighbor_counts'] = list(neighbor_counts)
        if neighbor_counts:
            stats['avg_neighbors'] = sum(neighbor_counts) / len(neighbor_counts)
        else:
            stats['avg_neighbors'] = 0.0

        stats['total_blocks'] = len(self._block_states)
        stats['step'] = self._global_step

        if reset:
            self._step_stats = {
                'inertia_norms': [],
                'c_G_values': [],
                'neighbor_counts': [],
            }

        return stats

    def set_param_names(self, model: torch.nn.Module):
        """Set parameter names from model for cleaner logging."""
        param_to_name = {id(p): name for name, p in model.named_parameters()}
        for p in self.param_groups[0]['params']:
            state = self.state.get(p, {})
            if 'param_id' in state:
                name = param_to_name.get(id(p), f"param_{state['param_id']}")
                self._param_names[state['param_id']] = name

    def get_neighborhood_graph(self) -> Dict[int, List[Tuple[int, float]]]:
        """
        Get the current Hebbian neighborhood graph.

        Returns dict mapping block_id -> [(neighbor_id, weight), ...]
        """
        graph = {}
        for block_id, state in self._block_states.items():
            graph[block_id] = [(n, state.weights.get(n, 0.0)) for n in state.neighbors]
        return graph
