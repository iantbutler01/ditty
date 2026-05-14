"""vLLM WorkerExtension for split-role GRPO weight transfer."""
from __future__ import annotations

from typing import Sequence

import torch


def _coerce_dtype(dtype_name: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype_name, torch.dtype):
        return dtype_name
    name = str(dtype_name)
    if name.startswith("torch."):
        name = name.split(".", 1)[1]
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported vLLM weight-update dtype: {dtype_name!r}")
    return dtype


class StatelessWeightUpdateWorkerExtension:
    """Receive full tensors over a side NCCL group and load them into vLLM."""

    def init_weight_update_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
    ) -> None:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.parallel_state import get_world_group
        from vllm.distributed.utils import StatelessProcessGroup

        side_rank = int(get_world_group().rank) + int(rank_offset)
        group = StatelessProcessGroup.create(
            host=str(master_address),
            port=int(master_port),
            rank=side_rank,
            world_size=int(world_size),
        )
        self.model_update_group = PyNcclCommunicator(group, device=self.device)

    def update_weight(self, name: str, dtype_name: str | torch.dtype, shape: Sequence[int]) -> None:
        if not hasattr(self, "model_update_group"):
            raise RuntimeError("init_weight_update_group must run before update_weight")
        weight = torch.empty(
            tuple(int(dim) for dim in shape),
            dtype=_coerce_dtype(dtype_name),
            device=self.device,
        )
        self.model_update_group.broadcast(weight, src=0, stream=torch.cuda.current_stream())
        self.model_runner.model.load_weights(weights=[(name, weight)])
        group = getattr(self.model_update_group, "group", None)
        barrier = getattr(group, "barrier", None)
        if callable(barrier):
            barrier()
        del weight

    def check_weight_signature(self) -> float:
        for _, param in self.model_runner.model.named_parameters():
            return float(param.detach().to(torch.float32).abs().sum().item())
        return 0.0
