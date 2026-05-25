"""Ray-managed distributed module launching for Ditty training jobs.

This module keeps the placement/orchestration primitive in Ditty while letting
projects keep their own training entrypoints. Ray Train is the preferred mode
for fault-tolerant fixed-world-size runs; the lower-level actor launcher remains
available for local experiments that need only placement control.
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Sequence


@dataclass
class RayModuleLaunchConfig:
    module: str
    module_args: list[str] = field(default_factory=list)
    num_workers: int = 1
    num_cpus_per_worker: float = 1.0
    num_gpus_per_worker: float = 1.0
    placement_strategy: str = "PACK"
    distributed_backend: str = "nccl"
    distributed_timeout_s: int = 1800
    master_addr: str | None = None
    master_port: int | None = None
    ray_address: str | None = None
    runtime_env: dict[str, Any] | None = None
    env: dict[str, str] = field(default_factory=dict)
    local_world_size: int | None = None
    node_rank: int = 0


@dataclass
class RayTrainModuleLaunchConfig:
    module: str
    module_args: list[str] = field(default_factory=list)
    num_workers: int = 1
    num_cpus_per_worker: float = 1.0
    num_gpus_per_worker: float = 1.0
    placement_strategy: str = "PACK"
    distributed_backend: str = "nccl"
    distributed_timeout_s: int = 1800
    max_failures: int = 0
    num_checkpoints_to_keep: int | None = None
    storage_path: str | None = None
    run_name: str | None = None
    ray_address: str | None = None
    runtime_env: dict[str, Any] | None = None
    env: dict[str, str] = field(default_factory=dict)
    restore_checkpoint_to: str | None = None


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _default_master_addr() -> str:
    try:
        import ray

        return str(ray.util.get_node_ip_address())
    except Exception:
        return "127.0.0.1"


def _normal_module_args(args: Sequence[str]) -> list[str]:
    values = list(args)
    if values and values[0] == "--":
        values = values[1:]
    return values


def _enable_ray_child_process_cleanup_env() -> None:
    # vLLM's mp executor starts CUDA child processes inside Ray workers. Ray's
    # opt-in process-group cleanup prevents leaked child GPU memory across
    # worker crashes and Ray Train retries.
    os.environ.setdefault("RAY_process_group_cleanup_enabled", "true")
    os.environ.setdefault(
        "RAY_kill_child_processes_on_worker_exit_with_raylet_subreaper",
        "true",
    )


def _uri_join(root: str, *parts: str) -> str:
    value = root.rstrip("/")
    for part in parts:
        stripped = str(part).strip("/")
        if stripped:
            value = f"{value}/{stripped}"
    return value


def _parse_env(values: Sequence[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected --env KEY=VALUE, got {value!r}")
        key, env_value = value.split("=", 1)
        if not key:
            raise ValueError(f"Expected non-empty env key in {value!r}")
        env[key] = env_value
    return env


def _parse_env_inherit(values: Sequence[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for key in values:
        key = str(key)
        if not key:
            raise ValueError("Expected non-empty env key for --env-inherit")
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def _runtime_env_with_env_vars(
    runtime_env: dict[str, Any] | None,
    env: dict[str, str],
) -> dict[str, Any] | None:
    merged = dict(runtime_env or {})
    if env:
        env_vars = dict(merged.get("env_vars") or {})
        env_vars.update({str(key): str(value) for key, value in env.items()})
        merged["env_vars"] = env_vars
    return merged or None


def _apply_module_env(env: dict[str, str]) -> None:
    os.environ.update(env)
    for path in reversed(os.environ.get("PYTHONPATH", "").split(os.pathsep)):
        if path and path not in sys.path:
            sys.path.insert(0, path)


def _run_module(module: str, module_args: Sequence[str]) -> None:
    old_argv = sys.argv
    sys.argv = [module, *[str(value) for value in module_args]]
    try:
        runpy.run_module(module, run_name="__main__")
    finally:
        sys.argv = old_argv


def _read_ditty_checkpoint_num(checkpoint_dir: str | os.PathLike[str]) -> int | None:
    from ditty.checkpoint import CheckpointManager

    return CheckpointManager.read_ray_train_checkpoint_num(str(checkpoint_dir))


def _gcs_uri_from_checkpoint(checkpoint: Any) -> str | None:
    path = str(getattr(checkpoint, "path", "") or "")
    if not path:
        return None
    if path.startswith("gs://"):
        return path

    filesystem = getattr(checkpoint, "filesystem", None)
    filesystem_type = str(getattr(filesystem, "type_name", "") or "").lower()
    if filesystem_type == "gcs":
        return f"gs://{path.lstrip('/')}"
    return None


def _sync_gcs_checkpoint(uri: str, target: Path) -> None:
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["gcloud", "storage", "rsync", "-r", uri, str(target)],
        check=True,
    )


def _restore_ditty_ray_checkpoint_dir(source: Path, target_root: Path) -> int | None:
    from ditty.checkpoint import CheckpointManager

    pointer = CheckpointManager.read_ray_checkpoint_pointer(str(source))
    if pointer is not None:
        durable_uri = str(pointer.get("durable_uri") or "")
        checkpoint_num = CheckpointManager.restore_durable_checkpoint(durable_uri, target_root)
        if checkpoint_num is None:
            raise RuntimeError(
                "Ray Train supplied a Ditty checkpoint pointer that could not be restored: "
                f"durable_uri={durable_uri!r}"
            )
        return checkpoint_num

    checkpoint_num = _read_ditty_checkpoint_num(source)
    if checkpoint_num is None:
        return None
    target = target_root / "checkpoints" / f"checkpoint_{checkpoint_num}"
    _replace_checkpoint_dir(source, target)
    return checkpoint_num


def _replace_checkpoint_dir(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.parent / f".{target.name}.restore_tmp"
    shutil.rmtree(tmp_target, ignore_errors=True)
    shutil.copytree(source, tmp_target)
    shutil.rmtree(target, ignore_errors=True)
    tmp_target.rename(target)


def _restore_gcs_ray_train_checkpoint(checkpoint: Any, target_root: Path) -> int | None:
    uri = _gcs_uri_from_checkpoint(checkpoint)
    if uri is None:
        return None

    tmp_target = target_root / "checkpoints" / ".ray_train_gcs_restore_tmp"
    print(
        f"[ditty.ray_training] restoring GCS Ray Train checkpoint via gcloud storage rsync: {uri}",
        flush=True,
    )
    _sync_gcs_checkpoint(uri, tmp_target)
    try:
        return _restore_ditty_ray_checkpoint_dir(tmp_target, target_root)
    finally:
        shutil.rmtree(tmp_target, ignore_errors=True)


def _restore_local_ray_train_checkpoint(checkpoint: Any, target_root: Path) -> int | None:
    with checkpoint.as_directory() as checkpoint_dir:
        return _restore_ditty_ray_checkpoint_dir(Path(checkpoint_dir), target_root)


def _restore_ray_train_checkpoint(restore_checkpoint_to: str | None) -> str | None:
    if not restore_checkpoint_to:
        return None

    try:
        import ray.train as ray_train
    except Exception:
        return None

    checkpoint = ray_train.get_checkpoint()
    if checkpoint is None:
        return None

    rank = int(os.environ.get("RANK", "0"))
    try:
        rank = int(ray_train.get_context().get_world_rank())
    except Exception:
        pass

    target_root = Path(restore_checkpoint_to)
    checkpoint_num: int | None = None
    if rank == 0:
        checkpoint_num = _restore_gcs_ray_train_checkpoint(checkpoint, target_root)
        if checkpoint_num is None:
            checkpoint_num = _restore_local_ray_train_checkpoint(checkpoint, target_root)
        if checkpoint_num is None:
            raise RuntimeError(
                "Ray Train supplied a checkpoint without Ditty metadata; "
                "cannot restore it into the Ditty checkpoint tree."
            )

    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            objects = [checkpoint_num]
            dist.broadcast_object_list(objects, src=0)
            checkpoint_num = int(objects[0]) if objects[0] is not None else None
            dist.barrier()
    except Exception:
        pass

    if checkpoint_num is None:
        raise RuntimeError(
            "Ray Train supplied a checkpoint without Ditty metadata; "
            "cannot restore it into the Ditty checkpoint tree."
        )
    target = target_root / "checkpoints" / f"checkpoint_{checkpoint_num}"
    if not target.exists():
        raise RuntimeError(f"Restored checkpoint path does not exist after rank-0 copy: {target}")
    return str(target)


class _RayModuleTrainWorker:
    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        local_world_size: int,
        node_rank: int,
        master_addr: str,
        master_port: int,
    ) -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.local_world_size = int(local_world_size)
        self.node_rank = int(node_rank)
        self.master_addr = str(master_addr)
        self.master_port = int(master_port)

    def ping(self) -> int:
        return self.rank

    def run(self, payload: dict[str, Any]) -> None:
        module = str(payload["module"])
        module_args = [str(value) for value in payload.get("module_args", [])]
        env = {str(key): str(value) for key, value in dict(payload.get("env") or {}).items()}
        backend = str(payload.get("distributed_backend") or "nccl")
        timeout_s = int(payload.get("distributed_timeout_s") or 1800)

        _apply_module_env(env)
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["LOCAL_RANK"] = "0"
        os.environ["LOCAL_WORLD_SIZE"] = str(self.local_world_size)
        os.environ["NODE_RANK"] = str(self.node_rank)
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["DITTY_RAY_MANAGED_TRAINER"] = "1"

        import torch
        import torch.distributed as dist

        if torch.cuda.is_available():
            torch.cuda.set_device(0)

        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                init_method=f"tcp://{self.master_addr}:{self.master_port}",
                rank=self.rank,
                world_size=self.world_size,
                timeout=timedelta(seconds=timeout_s),
            )

        try:
            _run_module(module, module_args)
        finally:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()


def launch_ray_module(config: RayModuleLaunchConfig):
    """Launch ``config.module`` in a Ray-managed distributed actor group."""

    if config.num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    if config.num_gpus_per_worker <= 0:
        raise ValueError("num_gpus_per_worker must be > 0")

    import ray
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    _enable_ray_child_process_cleanup_env()
    runtime_env = _runtime_env_with_env_vars(config.runtime_env, config.env)

    if not ray.is_initialized():
        init_kwargs: dict[str, Any] = {}
        if config.ray_address:
            init_kwargs["address"] = config.ray_address
        if runtime_env:
            init_kwargs["runtime_env"] = runtime_env
        ray.init(**init_kwargs)

    master_addr = config.master_addr or _default_master_addr()
    master_port = int(config.master_port or _find_free_port())
    local_world_size = int(config.local_world_size or config.num_workers)

    bundles = [
        {"CPU": float(config.num_cpus_per_worker), "GPU": float(config.num_gpus_per_worker)}
        for _ in range(config.num_workers)
    ]
    pg = placement_group(bundles, strategy=config.placement_strategy)
    ray.get(pg.ready())

    actor_cls = ray.remote(
        num_cpus=float(config.num_cpus_per_worker),
        num_gpus=float(config.num_gpus_per_worker),
    )(_RayModuleTrainWorker)
    workers = []
    for rank in range(config.num_workers):
        scheduling = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=rank,
            placement_group_capture_child_tasks=False,
        )
        workers.append(
            actor_cls.options(scheduling_strategy=scheduling).remote(
                rank=rank,
                world_size=config.num_workers,
                local_world_size=local_world_size,
                node_rank=config.node_rank,
                master_addr=master_addr,
                master_port=master_port,
            )
        )

    ready_ranks = ray.get([worker.ping.remote() for worker in workers])
    if sorted(int(rank) for rank in ready_ranks) != list(range(config.num_workers)):
        raise RuntimeError(f"Ray trainer actor startup returned unexpected ranks: {ready_ranks}")

    payload = {
        "module": config.module,
        "module_args": list(config.module_args),
        "env": dict(config.env),
        "distributed_backend": config.distributed_backend,
        "distributed_timeout_s": int(config.distributed_timeout_s),
    }
    return ray.get([worker.run.remote(payload) for worker in workers])


def _ray_train_module_worker(config: dict[str, Any]) -> None:
    module = str(config["module"])
    module_args = [str(value) for value in config.get("module_args", [])]
    env = {str(key): str(value) for key, value in dict(config.get("env") or {}).items()}
    restore_checkpoint_to = config.get("restore_checkpoint_to")

    _apply_module_env(env)
    os.environ["DITTY_RAY_TRAIN_WORKER"] = "1"
    restored = _restore_ray_train_checkpoint(str(restore_checkpoint_to) if restore_checkpoint_to else None)
    if restored and int(os.environ.get("RANK", "0")) == 0:
        print(f"[ditty.ray_training] restored Ray Train checkpoint to {restored}", flush=True)
    _run_module(module, module_args)


def launch_ray_train_module(config: RayTrainModuleLaunchConfig):
    """Launch ``config.module`` under Ray Train with fixed-world-size retries."""

    if config.num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    if config.num_gpus_per_worker <= 0:
        raise ValueError("num_gpus_per_worker must be > 0")

    import ray
    import ray.train as ray_train
    from ray.train.torch import TorchConfig, TorchTrainer

    _enable_ray_child_process_cleanup_env()
    train_env = dict(config.env)
    if config.storage_path:
        train_env.setdefault("DITTY_RAY_TRAIN_STORAGE_PATH", str(config.storage_path))
    if config.run_name:
        train_env.setdefault("DITTY_RAY_TRAIN_RUN_NAME", str(config.run_name))
    if config.storage_path and config.run_name:
        train_env.setdefault(
            "DITTY_RAY_TRAIN_DURABLE_ROOT",
            _uri_join(str(config.storage_path), str(config.run_name), "ditty_checkpoints"),
        )
        train_env.setdefault("DITTY_RAY_TRAIN_CHECKPOINT_MODE", "auto")

    runtime_env = _runtime_env_with_env_vars(config.runtime_env, train_env)

    if not ray.is_initialized():
        init_kwargs: dict[str, Any] = {}
        if config.ray_address:
            init_kwargs["address"] = config.ray_address
        if runtime_env:
            init_kwargs["runtime_env"] = runtime_env
        ray.init(**init_kwargs)

    resources_per_worker = {
        "CPU": float(config.num_cpus_per_worker),
        "GPU": float(config.num_gpus_per_worker),
    }
    scaling_config = ray_train.ScalingConfig(
        num_workers=int(config.num_workers),
        use_gpu=True,
        resources_per_worker=resources_per_worker,
        placement_strategy=config.placement_strategy,
    )
    checkpoint_config = None
    if config.num_checkpoints_to_keep is not None and config.num_checkpoints_to_keep > 0:
        checkpoint_config = ray_train.CheckpointConfig(
            num_to_keep=int(config.num_checkpoints_to_keep)
        )
    run_config = ray_train.RunConfig(
        name=config.run_name,
        storage_path=config.storage_path,
        failure_config=ray_train.FailureConfig(max_failures=int(config.max_failures)),
        checkpoint_config=checkpoint_config,
        worker_runtime_env=runtime_env,
    )
    trainer = TorchTrainer(
        _ray_train_module_worker,
        train_loop_config={
            "module": config.module,
            "module_args": list(config.module_args),
            "env": train_env,
            "restore_checkpoint_to": config.restore_checkpoint_to,
        },
        torch_config=TorchConfig(
            backend=config.distributed_backend,
            timeout_s=int(config.distributed_timeout_s),
        ),
        scaling_config=scaling_config,
        run_config=run_config,
    )
    return trainer.fit()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch a Python module in Ray-managed Ditty trainer actors.")
    parser.add_argument("--launcher", default="actors", choices=["actors", "ray-train"])
    parser.add_argument("--module", required=True, help="Python module to execute, for example training.grpo_ditty_pipeline")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-cpus-per-worker", type=float, default=1.0)
    parser.add_argument("--num-gpus-per-worker", type=float, default=1.0)
    parser.add_argument("--placement-strategy", default="PACK", choices=["PACK", "SPREAD", "STRICT_PACK", "STRICT_SPREAD"])
    parser.add_argument("--distributed-backend", default="nccl")
    parser.add_argument("--distributed-timeout-s", type=int, default=1800)
    parser.add_argument("--master-addr", default=None)
    parser.add_argument("--master-port", type=int, default=None)
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--runtime-env-json", default=None)
    parser.add_argument("--working-dir", default=None)
    parser.add_argument("--env", action="append", default=[], help="Environment override as KEY=VALUE; may be repeated")
    parser.add_argument("--env-inherit", action="append", default=[], help="Inherit KEY from the parent environment if set")
    parser.add_argument("--ray-train-storage-path", default=None)
    parser.add_argument("--ray-train-run-name", default=None)
    parser.add_argument("--ray-train-max-failures", type=int, default=0)
    parser.add_argument("--ray-train-num-checkpoints-to-keep", type=int, default=None)
    parser.add_argument("--restore-checkpoint-to", default=None)
    parser.add_argument("module_args", nargs=argparse.REMAINDER)
    return parser


def config_from_args(args: argparse.Namespace) -> RayModuleLaunchConfig:
    runtime_env: dict[str, Any] | None = None
    if args.runtime_env_json:
        runtime_env = json.loads(args.runtime_env_json)
    if args.working_dir:
        runtime_env = dict(runtime_env or {})
        runtime_env["working_dir"] = args.working_dir
    return RayModuleLaunchConfig(
        module=args.module,
        module_args=_normal_module_args(args.module_args),
        num_workers=args.num_workers,
        num_cpus_per_worker=args.num_cpus_per_worker,
        num_gpus_per_worker=args.num_gpus_per_worker,
        placement_strategy=args.placement_strategy,
        distributed_backend=args.distributed_backend,
        distributed_timeout_s=args.distributed_timeout_s,
        master_addr=args.master_addr,
        master_port=args.master_port,
        ray_address=args.ray_address,
        runtime_env=runtime_env,
        env={**_parse_env(args.env), **_parse_env_inherit(args.env_inherit)},
    )


def ray_train_config_from_args(args: argparse.Namespace) -> RayTrainModuleLaunchConfig:
    runtime_env: dict[str, Any] | None = None
    if args.runtime_env_json:
        runtime_env = json.loads(args.runtime_env_json)
    if args.working_dir:
        runtime_env = dict(runtime_env or {})
        runtime_env["working_dir"] = args.working_dir
    return RayTrainModuleLaunchConfig(
        module=args.module,
        module_args=_normal_module_args(args.module_args),
        num_workers=args.num_workers,
        num_cpus_per_worker=args.num_cpus_per_worker,
        num_gpus_per_worker=args.num_gpus_per_worker,
        placement_strategy=args.placement_strategy,
        distributed_backend=args.distributed_backend,
        distributed_timeout_s=args.distributed_timeout_s,
        max_failures=args.ray_train_max_failures,
        num_checkpoints_to_keep=args.ray_train_num_checkpoints_to_keep,
        storage_path=args.ray_train_storage_path,
        run_name=args.ray_train_run_name,
        ray_address=args.ray_address,
        runtime_env=runtime_env,
        env={**_parse_env(args.env), **_parse_env_inherit(args.env_inherit)},
        restore_checkpoint_to=args.restore_checkpoint_to,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.launcher == "ray-train":
        launch_ray_train_module(ray_train_config_from_args(args))
    else:
        launch_ray_module(config_from_args(args))


if __name__ == "__main__":
    main()
