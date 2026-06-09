"""Ray-hosted vLLM rollout engine for split trainer/inference GRPO.

This module implements the separated-role shape used by the vLLM/OpenRLHF/TRL
references: FSDP trainer ranks stay on trainer GPUs, rank 0 owns the control
plane, and vLLM runs in Ray workers on disjoint inference GPUs. Weight transfer
uses a side NCCL communicator created with vLLM's StatelessProcessGroup, so it
does not mutate the trainer FSDP process group or vLLM's TP group.
"""
from __future__ import annotations

import os
import gc
import socket
import time
import traceback
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.distributed as dist


@dataclass
class VllmCompletionOutput:
    token_ids: list[int]


@dataclass
class VllmRequestOutput:
    outputs: list[VllmCompletionOutput]


def _rank_context() -> tuple[int, int, int]:
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("LOCAL_RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )


def _is_rank0() -> bool:
    rank, _, _ = _rank_context()
    return rank == 0


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _default_host() -> str:
    configured = os.environ.get("DITTY_RAY_VLLM_WEIGHT_SYNC_HOST")
    if configured:
        return configured
    for module_name in ("vllm.utils.network_utils", "vllm.utils"):
        try:
            module = __import__(module_name, fromlist=["get_ip"])
            get_ip = getattr(module, "get_ip")
            host = str(get_ip())
            if host and not host.startswith("127."):
                return host
        except Exception:
            pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            host = str(sock.getsockname()[0])
            if host and not host.startswith("127."):
                return host
    except OSError:
        pass
    return socket.gethostbyname(socket.gethostname())


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _optional_positive_int(value: int | str | None) -> int | None:
    if value is None or value == "":
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def _optional_positive_float(value: float | str | None) -> float | None:
    if value is None or value == "":
        return None
    parsed = float(value)
    return parsed if parsed > 0 else None


def _sampling_value(sampling_params: Any, name: str, default: Any) -> Any:
    if isinstance(sampling_params, dict):
        return sampling_params.get(name, default)
    return getattr(sampling_params, name, default)


def _sampling_sequence_count(sampling_params: Any) -> int:
    value = _sampling_value(sampling_params, "n", 1)
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _sampling_expected_new_tokens(sampling_params: Any) -> int:
    value = _sampling_value(sampling_params, "max_tokens", None)
    if value is None:
        value = _sampling_value(sampling_params, "max_new_tokens", 0)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


class RayVllmActor:
    """Ray actor that owns one TP-sharded vLLM engine."""

    def __init__(
        self,
        *,
        model_path: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        dtype: str = "bfloat16",
        max_model_len: int | None = None,
        trust_remote_code: bool = True,
        enable_sleep_mode: bool = True,
        enforce_eager: bool = False,
        disable_custom_all_reduce: bool = False,
        worker_extension_cls: str = "ditty.vllm_worker_extension.StatelessWeightUpdateWorkerExtension",
        distributed_executor_backend: str = "ray",
        weight_transfer_backend: str | None = "nccl",
        gdn_prefill_backend: str | None = None,
        cuda_visible_devices: str | None = None,
        unset_cuda_visible_devices: bool = True,
    ) -> None:
        from vllm import LLM

        self.llm = None
        self._is_asleep = False
        allocator_config = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments" in allocator_config:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            os.environ.pop("PYTORCH_ALLOC_CONF", None)
            print(
                "[rank 0] ray_vllm cleared expandable_segments allocator config "
                "for vLLM memory pool compatibility",
                flush=True,
            )
        uses_ray_executor = distributed_executor_backend == "ray" or distributed_executor_backend.endswith("RayDistributedExecutor")
        if uses_ray_executor:
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(str(i) for i in range(int(tensor_parallel_size)))
        else:
            os.environ.pop("VLLM_RAY_BUNDLE_INDICES", None)
        if cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
            if uses_ray_executor:
                os.environ["DITTY_VLLM_CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
                distributed_executor_backend = "ditty.vllm_ray_executor.DittyRayDistributedExecutor"
        else:
            os.environ.pop("DITTY_VLLM_CUDA_VISIBLE_DEVICES", None)
            if unset_cuda_visible_devices:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        kwargs: dict[str, Any] = {
            "model": model_path,
            "tensor_parallel_size": int(tensor_parallel_size),
            "distributed_executor_backend": distributed_executor_backend,
            "worker_extension_cls": worker_extension_cls,
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "enable_sleep_mode": bool(enable_sleep_mode),
            "enforce_eager": bool(enforce_eager),
            "disable_custom_all_reduce": bool(disable_custom_all_reduce),
        }
        self._native_weight_transfer = False
        if weight_transfer_backend:
            try:
                from vllm.config import WeightTransferConfig

                kwargs["weight_transfer_config"] = WeightTransferConfig(
                    backend=str(weight_transfer_backend)
                )
                self._native_weight_transfer = True
            except Exception:
                self._native_weight_transfer = False
        if max_model_len is not None:
            kwargs["max_model_len"] = int(max_model_len)
        if gdn_prefill_backend:
            kwargs["gdn_prefill_backend"] = str(gdn_prefill_backend)
        self.llm = LLM(**kwargs)
        self._is_asleep = False

    def __ray_shutdown__(self) -> None:
        timeout_s = _optional_positive_float(os.environ.get("DITTY_RAY_VLLM_SHUTDOWN_TIMEOUT_S", "20"))
        self.shutdown(timeout_s=timeout_s)

    def init_weight_update_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
    ) -> None:
        if self._native_weight_transfer:
            self.llm.collective_rpc(
                "init_weight_transfer_engine",
                args=(
                    {
                        "master_address": str(master_address),
                        "master_port": int(master_port),
                        "rank_offset": int(rank_offset),
                        "world_size": int(world_size),
                    },
                ),
            )
        else:
            self.llm.collective_rpc(
                "init_weight_update_group",
                args=(master_address, int(master_port), int(rank_offset), int(world_size)),
            )

    def update_weight(self, name: str, dtype_name: str, shape: Sequence[int]) -> None:
        if self._native_weight_transfer:
            update_info = {
                "names": [name],
                "dtype_names": [dtype_name],
                "shapes": [[int(dim) for dim in shape]],
                "packed": False,
            }
            self.llm.start_weight_update(is_checkpoint_format=True)
            try:
                self.llm.update_weights({"update_info": update_info})
            finally:
                self.llm.finish_weight_update()
        else:
            self.llm.collective_rpc(
                "update_weight",
                args=(name, dtype_name, tuple(int(dim) for dim in shape)),
            )

    def update_weights(
        self,
        names: Sequence[str],
        dtype_names: Sequence[str],
        shapes: Sequence[Sequence[int]],
    ) -> None:
        if self._native_weight_transfer:
            update_info = {
                "names": [str(name) for name in names],
                "dtype_names": [str(dtype_name) for dtype_name in dtype_names],
                "shapes": [[int(dim) for dim in shape] for shape in shapes],
                "packed": False,
            }
            self.llm.start_weight_update(is_checkpoint_format=True)
            try:
                self.llm.update_weights({"update_info": update_info})
            finally:
                self.llm.finish_weight_update()
            return
        for name, dtype_name, shape in zip(names, dtype_names, shapes):
            self.update_weight(str(name), str(dtype_name), shape)

    def sleep(self, level: int = 1) -> None:
        if not self._is_asleep:
            self.llm.sleep(level=int(level))
            self._is_asleep = True

    def wake_up(self, tags: list[str] | None = None) -> None:
        if self._is_asleep:
            self.llm.wake_up(tags=tags)
            if tags is None or "kv_cache" in tags:
                self._is_asleep = False

    def generate(self, prompts: Sequence[str], sampling_params: Any, *, use_tqdm: bool = False) -> list[list[list[int]]]:
        if self._is_asleep:
            self.llm.wake_up(tags=["kv_cache"])
            self._is_asleep = False
        outputs = self.llm.generate(list(prompts), sampling_params=sampling_params, use_tqdm=use_tqdm)
        return [
            [list(completion.token_ids) for completion in request_output.outputs]
            for request_output in outputs
        ]

    def shutdown(self, timeout_s: float | None = 10.0) -> None:
        llm = getattr(self, "llm", None)
        if llm is None:
            return
        try:
            llm_engine = getattr(llm, "llm_engine", None)
            engine_core = getattr(llm_engine, "engine_core", None)
            if engine_core is not None and hasattr(engine_core, "shutdown"):
                engine_core.shutdown(timeout=float(timeout_s) if timeout_s is not None else None)
            elif llm_engine is not None and hasattr(llm_engine, "shutdown"):
                llm_engine.shutdown()
        finally:
            self.llm = None
            self._is_asleep = False
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class RayVllmRolloutEngine:
    """Trainer-side client for a Ray-hosted TP-sharded vLLM engine.

    Every FSDP rank owns an instance of this class. Only rank 0 creates Ray
    actors and the side NCCL communicator. All ranks must call
    ``update_weights_from_fsdp_model`` so FSDP ``full_tensor()`` collectives stay
    aligned.
    """

    def __init__(
        self,
        *,
        model_path: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        dtype: str = "bfloat16",
        max_model_len: int | None = None,
        trust_remote_code: bool = True,
        enable_sleep_mode: bool = True,
        enforce_eager: bool = False,
        disable_custom_all_reduce: bool = False,
        distributed_executor_backend: str = "ray",
        weight_transfer_backend: str | None = "nccl",
        gdn_prefill_backend: str | None = None,
        placement_group_ready_timeout_s: float | None = 300.0,
        ray_address: str | None = None,
        ray_runtime_env: dict[str, Any] | None = None,
        worker_extension_cls: str = "ditty.vllm_worker_extension.StatelessWeightUpdateWorkerExtension",
        cuda_visible_devices: str | None = None,
        unset_cuda_visible_devices: bool = True,
        max_prompts_per_request: int | None = None,
        max_sequences_per_request: int | None = None,
        max_expected_new_tokens_per_request: int | None = None,
        generate_timeout_s: float | None = None,
        num_replicas: int | None = None,
    ) -> None:
        self.model_path = model_path
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.num_replicas = max(
            1,
            int(
                num_replicas
                if num_replicas is not None
                else os.environ.get("DITTY_RAY_VLLM_NUM_REPLICAS", "1")
            ),
        )
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.trust_remote_code = bool(trust_remote_code)
        self.enable_sleep_mode = bool(enable_sleep_mode)
        self.enforce_eager = bool(enforce_eager)
        self.disable_custom_all_reduce = bool(disable_custom_all_reduce)
        self.distributed_executor_backend = distributed_executor_backend
        self.weight_transfer_backend = weight_transfer_backend
        self.gdn_prefill_backend = (
            gdn_prefill_backend
            if gdn_prefill_backend is not None
            else os.environ.get("DITTY_RAY_VLLM_GDN_PREFILL_BACKEND")
        )
        self.placement_group_ready_timeout_s = placement_group_ready_timeout_s
        self.ray_address = ray_address
        self.ray_runtime_env = dict(ray_runtime_env or {})
        self.worker_extension_cls = worker_extension_cls
        self.cuda_visible_devices = cuda_visible_devices
        self.unset_cuda_visible_devices = bool(unset_cuda_visible_devices)
        self.max_prompts_per_request = _optional_positive_int(
            max_prompts_per_request
            if max_prompts_per_request is not None
            else os.environ.get("DITTY_RAY_VLLM_MAX_PROMPTS_PER_REQUEST")
        )
        self.max_sequences_per_request = _optional_positive_int(
            max_sequences_per_request
            if max_sequences_per_request is not None
            else os.environ.get("DITTY_RAY_VLLM_MAX_SEQUENCES_PER_REQUEST", "216")
        )
        self.max_expected_new_tokens_per_request = _optional_positive_int(
            max_expected_new_tokens_per_request
            if max_expected_new_tokens_per_request is not None
            else os.environ.get("DITTY_RAY_VLLM_MAX_EXPECTED_NEW_TOKENS_PER_REQUEST", "65536")
        )
        self.generate_timeout_s = _optional_positive_float(
            generate_timeout_s
            if generate_timeout_s is not None
            else os.environ.get("DITTY_RAY_VLLM_GENERATE_TIMEOUT_S", "540")
        )
        self.actor = None
        self.actors: list[Any] = []
        self.placement_groups: list[Any] = []
        self.placement_group = None
        self.ray = None
        self.model_update_group = None
        self.model_update_groups: list[Any] = []
        self._initialized = False
        self._last_sync_step = -1

    def ensure_started(self) -> None:
        if self._initialized:
            return
        if _is_rank0():
            import ray
            from ray.util.placement_group import placement_group
            from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

            self.ray = ray
            if not ray.is_initialized():
                init_kwargs: dict[str, Any] = {}
                if self.ray_address:
                    init_kwargs["address"] = self.ray_address
                if self.ray_runtime_env:
                    init_kwargs["runtime_env"] = self.ray_runtime_env
                ray.init(**init_kwargs)

            placement_groups = []
            if self.distributed_executor_backend == "ray":
                for replica_index in range(self.num_replicas):
                    placement_groups.append(
                        placement_group(
                            [{"GPU": 1, "CPU": 0} for _ in range(self.tensor_parallel_size)],
                            strategy="PACK",
                        )
                    )
                self.placement_groups = placement_groups
                self.placement_group = placement_groups[0] if placement_groups else None
            try:
                for pg in placement_groups:
                    ray.get(pg.ready(), timeout=self.placement_group_ready_timeout_s)
                actor_cls = ray.remote(num_cpus=0, num_gpus=0)(RayVllmActor)
                actors = []
                for replica_index in range(self.num_replicas):
                    actor_options: dict[str, Any] = {}
                    if placement_groups:
                        scheduling = PlacementGroupSchedulingStrategy(
                            placement_group=placement_groups[replica_index],
                            placement_group_capture_child_tasks=True,
                            placement_group_bundle_index=0,
                        )
                        actor_options["scheduling_strategy"] = scheduling
                    actors.append(
                        actor_cls.options(**actor_options).remote(
                            model_path=self.model_path,
                            tensor_parallel_size=self.tensor_parallel_size,
                            gpu_memory_utilization=self.gpu_memory_utilization,
                            dtype=self.dtype,
                            max_model_len=self.max_model_len,
                            trust_remote_code=self.trust_remote_code,
                            enable_sleep_mode=self.enable_sleep_mode,
                            enforce_eager=self.enforce_eager,
                            disable_custom_all_reduce=self.disable_custom_all_reduce,
                            worker_extension_cls=self.worker_extension_cls,
                            distributed_executor_backend=self.distributed_executor_backend,
                            weight_transfer_backend=self.weight_transfer_backend,
                            gdn_prefill_backend=self.gdn_prefill_backend,
                            cuda_visible_devices=self.cuda_visible_devices,
                            unset_cuda_visible_devices=self.unset_cuda_visible_devices,
                        )
                    )
                self.actors = actors
                self.actor = actors[0] if actors else None
                self._init_weight_update_group()
            except BaseException:
                self._cleanup_rank0()
                raise
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()
        self._initialized = True

    def _cleanup_rank0(self) -> None:
        if not _is_rank0() or self.ray is None:
            return
        actors = list(self.actors or ([] if self.actor is None else [self.actor]))
        for actor in actors:
            timeout_s = _optional_positive_float(os.environ.get("DITTY_RAY_VLLM_SHUTDOWN_TIMEOUT_S", "20"))
            if timeout_s is not None:
                try:
                    self.ray.get(actor.shutdown.remote(timeout_s), timeout=timeout_s + 5.0)
                except BaseException:
                    pass
            try:
                self.ray.kill(actor, no_restart=True)
            except BaseException:
                pass
        self.actors = []
        self.actor = None
        for pg in list(self.placement_groups or ([] if self.placement_group is None else [self.placement_group])):
            try:
                from ray.util import remove_placement_group

                remove_placement_group(pg)
            except BaseException:
                pass
        self.placement_groups = []
        self.placement_group = None
        self.model_update_groups = []
        self.model_update_group = None
        self._initialized = False

    def _combined_request_chunks(self, sampling_params: Sequence[Any]) -> list[tuple[int, int]]:
        chunks: list[tuple[int, int]] = []
        start = 0
        prompt_count = 0
        sequence_count = 0
        expected_new_tokens = 0
        for index, params in enumerate(sampling_params):
            prompt_cost = 1
            sequence_cost = _sampling_sequence_count(params)
            token_cost = sequence_cost * _sampling_expected_new_tokens(params)
            would_exceed = (
                prompt_count > 0
                and (
                    (
                        self.max_prompts_per_request is not None
                        and prompt_count + prompt_cost > self.max_prompts_per_request
                    )
                    or (
                        self.max_sequences_per_request is not None
                        and sequence_count + sequence_cost > self.max_sequences_per_request
                    )
                    or (
                        self.max_expected_new_tokens_per_request is not None
                        and expected_new_tokens + token_cost > self.max_expected_new_tokens_per_request
                    )
                )
            )
            if would_exceed:
                chunks.append((start, index))
                start = index
                prompt_count = 0
                sequence_count = 0
                expected_new_tokens = 0
            prompt_count += prompt_cost
            sequence_count += sequence_cost
            expected_new_tokens += token_cost
        if start < len(sampling_params):
            chunks.append((start, len(sampling_params)))
        return chunks

    def _init_weight_update_group(self) -> None:
        assert self.ray is not None
        assert self.actors
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        master_address = _default_host()
        store_timeout_s = int(os.environ.get("DITTY_RAY_VLLM_WEIGHT_SYNC_STORE_TIMEOUT_S", "1800"))
        side_world_size = self.tensor_parallel_size + 1
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        groups = []
        handles = []
        configured_port = os.environ.get("DITTY_RAY_VLLM_WEIGHT_SYNC_PORT")
        for replica_index, actor in enumerate(self.actors):
            master_port = int(configured_port) + replica_index if configured_port else _find_free_port()
            print(
                f"[rank 0] ray_vllm weight sync group init "
                f"replica={replica_index} host={master_address} port={master_port} "
                f"world_size={side_world_size} store_timeout_s={store_timeout_s}",
                flush=True,
            )
            handles.append(
                actor.init_weight_update_group.remote(
                    master_address,
                    master_port,
                    1,
                    side_world_size,
                )
            )
            pg = StatelessProcessGroup.create(
                host=master_address,
                port=master_port,
                rank=0,
                world_size=side_world_size,
                store_timeout=store_timeout_s,
            )
            groups.append(PyNcclCommunicator(pg, device=device))
        self.model_update_groups = groups
        self.model_update_group = groups[0] if groups else None
        self.ray.get(handles)
        print(
            f"[rank 0] ray_vllm weight sync group ready replicas={len(groups)}",
            flush=True,
        )

    def update_weights_from_fsdp_model(self, model: torch.nn.Module) -> int:
        self.ensure_started()
        rank, _, _ = _rank_context()
        count = 0
        start = time.time()
        log_every = max(1, int(os.environ.get("DITTY_RAY_VLLM_SYNC_LOG_EVERY", "25")))
        trace_first = max(0, int(os.environ.get("DITTY_RAY_VLLM_SYNC_TRACE_FIRST", "0")))
        named_params = list(model.named_parameters())
        update_handle = None
        if rank == 0:
            assert self.ray is not None
            assert self.actors
            names = [name for name, _param in named_params]
            dtype_names = ["bfloat16" for _name, _param in named_params]
            shapes = [
                [int(dim) for dim in getattr(param, "shape", ())]
                for _name, param in named_params
            ]
            print(
                f"[rank 0] ray_vllm weight sync metadata start "
                f"replicas={len(self.actors)} params={len(names)}",
                flush=True,
            )
            update_handle = [
                actor.update_weights.remote(names, dtype_names, shapes)
                for actor in self.actors
            ]
        try:
            for name, param in named_params:
                param_start = time.time()
                if count < trace_first:
                    shape = tuple(int(dim) for dim in getattr(param, "shape", ()))
                    print(
                        f"[rank {rank}] ray_vllm weight sync param begin "
                        f"index={count + 1} name={name} shape={shape} "
                        f"has_full_tensor={hasattr(param, 'full_tensor')}",
                        flush=True,
                    )
                if hasattr(param, "full_tensor"):
                    full = param.full_tensor()
                else:
                    full = param.detach()
                if count < trace_first:
                    print(
                        f"[rank {rank}] ray_vllm weight sync full_tensor done "
                        f"index={count + 1} name={name} elapsed={time.time() - param_start:.1f}s",
                        flush=True,
                    )
                if rank == 0:
                    assert self.ray is not None
                    assert self.actors
                    assert self.model_update_groups
                    tensor = full.detach().to(torch.bfloat16).contiguous()
                    if count < trace_first:
                        print(
                            f"[rank 0] ray_vllm weight sync broadcast start "
                            f"index={count + 1} name={name} shape={tuple(tensor.shape)}",
                            flush=True,
                        )
                    for replica_index, group in enumerate(self.model_update_groups):
                        group.broadcast(tensor, src=0, stream=torch.cuda.current_stream())
                        if count < trace_first:
                            print(
                                f"[rank 0] ray_vllm weight sync broadcast replica done "
                                f"replica={replica_index} index={count + 1} name={name}",
                                flush=True,
                            )
                    if count < trace_first:
                        print(
                            f"[rank 0] ray_vllm weight sync broadcast done "
                            f"index={count + 1} name={name} elapsed={time.time() - param_start:.1f}s",
                            flush=True,
                        )
                    del tensor
                del full
                count += 1
                if rank == 0 and (count == 1 or count % log_every == 0):
                    print(
                        f"[rank 0] ray_vllm weight sync progress "
                        f"params={count} last={name} elapsed={time.time() - start:.1f}s",
                        flush=True,
                    )
                if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                    dist.barrier()
            if rank == 0 and update_handle is not None:
                assert self.ray is not None
                self.ray.get(update_handle)
                print(
                    f"[rank 0] ray_vllm weight sync actor load done "
                    f"replicas={len(self.actors)} params={count} "
                    f"elapsed={time.time() - start:.1f}s",
                    flush=True,
                )
            if rank == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._last_sync_step += 1
            if rank == 0:
                print(
                    f"[rank 0] ray_vllm weight sync done params={count} elapsed={time.time() - start:.1f}s",
                    flush=True,
                )
            return count
        except BaseException:
            self._cleanup_rank0()
            raise

    def generate(self, prompts: Sequence[str], sampling_params: Any, *, use_tqdm: bool = False) -> list[VllmRequestOutput]:
        self.ensure_started()
        rank, _, world_size = _rank_context()
        local_payload = {
            "rank": rank,
            "prompts": list(prompts),
            "sampling_params": sampling_params,
            "use_tqdm": use_tqdm,
        }
        if dist.is_available() and dist.is_initialized() and world_size > 1:
            gathered: list[Any] | None = [None for _ in range(world_size)] if rank == 0 else None
            dist.gather_object(local_payload, gathered, dst=0)
            outputs_by_rank: list[Any] | None = None
            if rank == 0:
                assert self.ray is not None
                assert self.actors
                try:
                    outputs_by_rank = [None for _ in range(world_size)]
                    combined_prompts: list[str] = []
                    combined_sampling_params: list[Any] = []
                    combined_items: list[tuple[int, int]] = []
                    use_tqdm_any = False
                    for payload in gathered or []:
                        if not isinstance(payload, dict):
                            continue
                        payload_rank = int(payload["rank"])
                        payload_prompts = list(payload.get("prompts") or [])
                        outputs_by_rank[payload_rank] = [None for _ in payload_prompts]
                        payload_sampling_params = payload.get("sampling_params")
                        if isinstance(payload_sampling_params, Sequence):
                            per_prompt_sampling_params = list(payload_sampling_params)
                            if len(per_prompt_sampling_params) != len(payload_prompts):
                                raise ValueError(
                                    "per-prompt sampling params length mismatch: "
                                    f"rank={payload_rank} prompts={len(payload_prompts)} "
                                    f"sampling_params={len(per_prompt_sampling_params)}"
                                )
                        else:
                            per_prompt_sampling_params = [
                                payload_sampling_params for _ in payload_prompts
                            ]
                        use_tqdm_any = use_tqdm_any or bool(payload.get("use_tqdm", False))
                        for prompt_index, prompt in enumerate(payload_prompts):
                            combined_items.append((payload_rank, prompt_index))
                            combined_prompts.append(str(prompt))
                            combined_sampling_params.append(per_prompt_sampling_params[prompt_index])
                    if combined_prompts:
                        chunks = self._combined_request_chunks(combined_sampling_params)
                        total_sequences = sum(
                            _sampling_sequence_count(params)
                            for params in combined_sampling_params
                        )
                        if len(chunks) > 1:
                            print(
                                "[rank 0] ray_vllm generate chunking "
                                f"prompts={len(combined_prompts)} sequences={total_sequences} "
                                f"chunks={len(chunks)} "
                                f"max_sequences_per_request={self.max_sequences_per_request} "
                                "max_expected_new_tokens_per_request="
                                f"{self.max_expected_new_tokens_per_request}",
                                flush=True,
                            )
                        combined_token_ids = [None for _ in combined_items]
                        chunk_handles: list[tuple[int, int, Any]] = []
                        for chunk_index, (start, end) in enumerate(chunks):
                            if len(chunks) > 1:
                                chunk_sequences = sum(
                                    _sampling_sequence_count(params)
                                    for params in combined_sampling_params[start:end]
                                )
                                chunk_tokens = sum(
                                    _sampling_sequence_count(params)
                                    * _sampling_expected_new_tokens(params)
                                    for params in combined_sampling_params[start:end]
                                )
                                print(
                                    "[rank 0] ray_vllm generate chunk "
                                    f"{chunk_index + 1}/{len(chunks)} "
                                    f"prompts={end - start} sequences={chunk_sequences} "
                                    f"expected_new_tokens={chunk_tokens}",
                                    flush=True,
                                )
                            actor = self.actors[chunk_index % len(self.actors)]
                            handle = actor.generate.remote(
                                combined_prompts[start:end],
                                combined_sampling_params[start:end],
                                use_tqdm=use_tqdm_any,
                            )
                            chunk_handles.append((start, end, handle))
                        handles = [handle for _start, _end, handle in chunk_handles]
                        if self.generate_timeout_s is None:
                            chunk_results = self.ray.get(handles)
                        else:
                            chunk_results = self.ray.get(handles, timeout=self.generate_timeout_s)
                        for (start, end, _handle), chunk_token_ids in zip(chunk_handles, chunk_results):
                            if len(chunk_token_ids) != end - start:
                                raise RuntimeError(
                                    "vLLM returned an unexpected number of chunk request outputs: "
                                    f"expected={end - start} got={len(chunk_token_ids)}"
                                )
                            combined_token_ids[start:end] = chunk_token_ids
                        if any(value is None for value in combined_token_ids):
                            raise RuntimeError("vLLM generation left empty slots in the combined output")
                        if len(combined_token_ids) != len(combined_items):
                            raise RuntimeError(
                                "vLLM returned an unexpected number of request outputs: "
                                f"expected={len(combined_items)} got={len(combined_token_ids)}"
                            )
                        for (payload_rank, prompt_index), token_ids in zip(combined_items, combined_token_ids):
                            assert outputs_by_rank[payload_rank] is not None
                            outputs_by_rank[payload_rank][prompt_index] = token_ids
                except Exception as error:
                    self._cleanup_rank0()
                    message = (
                        "rank 0 ray_vllm generate failed: "
                        f"{error.__class__.__name__}: {error}\n{traceback.format_exc()}"
                    )
                    print(message, flush=True)
                    outputs_by_rank = [
                        {"__ditty_ray_vllm_error__": message}
                        for _ in range(world_size)
                    ]
            rank_token_ids: Any = [None]
            dist.scatter_object_list(rank_token_ids, outputs_by_rank, src=0)
            token_ids_payload = rank_token_ids[0]
            if isinstance(token_ids_payload, dict) and "__ditty_ray_vllm_error__" in token_ids_payload:
                raise RuntimeError(str(token_ids_payload["__ditty_ray_vllm_error__"]))
        else:
            assert self.ray is not None
            assert self.actors
            token_ids_payload = self.ray.get(
                self.actors[0].generate.remote(list(prompts), sampling_params, use_tqdm=use_tqdm)
            )
        return [
            VllmRequestOutput(
                outputs=[VllmCompletionOutput(token_ids=list(token_ids)) for token_ids in request_outputs]
            )
            for request_outputs in (token_ids_payload or [])
        ]

    def sleep(self, level: int = 1) -> None:
        if _is_rank0() and self.actors and self.ray is not None:
            self.ray.get([actor.sleep.remote(int(level)) for actor in self.actors])

    def shutdown(self) -> None:
        if _is_rank0() and self.actors and self.ray is not None:
            timeout_s = float(os.environ.get("DITTY_RAY_VLLM_SHUTDOWN_TIMEOUT_S", "20"))
            for actor in list(self.actors):
                try:
                    self.ray.get(actor.shutdown.remote(timeout_s), timeout=timeout_s + 5.0)
                except BaseException:
                    pass
        self._cleanup_rank0()
