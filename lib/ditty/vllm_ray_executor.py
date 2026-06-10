"""Ditty vLLM Ray executor hooks.

vLLM's Ray executor computes worker ``CUDA_VISIBLE_DEVICES`` from Ray's
accelerator ids. In a split trainer/inference job the Ray resource reservation
can be correct while the physical CUDA ids still need to be pinned explicitly
for the vLLM workers. This subclass keeps vLLM's Ray scheduling intact and only
overrides the worker-visible CUDA list when Ditty asks for it.
"""
from __future__ import annotations

import os
from typing import Any

from vllm.platforms import current_platform
from vllm.v1.executor.ray_executor import RayDistributedExecutor


class DittyRayDistributedExecutor(RayDistributedExecutor):
    """Ray executor that can pin vLLM workers to an explicit CUDA device list."""

    def _init_executor(self) -> None:
        # vLLM derives ParallelConfig.use_ray from the backend selector string.
        # A custom RayDistributedExecutor subclass still needs the flag set for
        # vLLM v1's compiled DAG execution path.
        if isinstance(self.parallel_config.distributed_executor_backend, str):
            self.parallel_config.distributed_executor_backend = type(self)
        super()._init_executor()

    def _update_noset_device_env_vars(self, ray_remote_kwargs):
        ray_remote_kwargs = super()._update_noset_device_env_vars(ray_remote_kwargs)
        visible_devices = os.environ.get("DITTY_VLLM_CUDA_VISIBLE_DEVICES")
        if visible_devices:
            runtime_env = ray_remote_kwargs.setdefault("runtime_env", {})
            env_vars = runtime_env.setdefault("env_vars", {})
            env_vars[current_platform.device_control_env_var] = visible_devices
        return ray_remote_kwargs

    def _get_env_vars_to_be_updated(self) -> list[dict[str, Any]]:
        env_updates = super()._get_env_vars_to_be_updated()
        visible_devices = os.environ.get("DITTY_VLLM_CUDA_VISIBLE_DEVICES")
        if visible_devices:
            device_env = current_platform.device_control_env_var
            for worker_env in env_updates:
                worker_env[device_env] = visible_devices
        return env_updates
