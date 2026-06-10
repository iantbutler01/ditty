import os
import pickle
import random
import re
import io
import json
import shutil
import subprocess
import tempfile
import threading
import uuid
from concurrent.futures import Future as ConcurrentFuture
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn

logger = getLogger("ditty_checkpoint")

_DIRECT_DISTRIBUTED_CHECKPOINT_POINTER = "ditty_distributed_checkpoint.json"


def _dist_debug_enabled() -> bool:
    value = os.environ.get("DITTY_DEBUG_DIST", "")
    return value.lower() in {"1", "true", "yes", "on"}


def _dist_debug(message: str) -> None:
    if not _dist_debug_enabled():
        return
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    logger.info(f"[dist-debug rank={rank} local_rank={local_rank}] {message}")


def _should_log_rank_zero_only() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _summarize_checkpoint_error(error: BaseException) -> str:
    message = str(error).strip()
    if not message:
        return type(error).__name__

    patterns = (
        r"Missing key in checkpoint state_dict: ([^\s]+)",
        r"Unexpected key in checkpoint state_dict: ([^\s]+)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, message)
        for value in reversed(matches):
            if "{" not in value and "}" not in value:
                prefix = pattern.split(":")[0].replace("\\", "")
                return f"{prefix}: {value}"

    for line in message.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return type(error).__name__


def _uri_join(root: str, *parts: str) -> str:
    value = root.rstrip("/")
    for part in parts:
        stripped = str(part).strip("/")
        if stripped:
            value = f"{value}/{stripped}"
    return value


def _is_gcs_uri(value: str | None) -> bool:
    return bool(value and str(value).startswith("gs://"))


def _gcs_path_parts(uri: str) -> tuple[str, str]:
    if not _is_gcs_uri(uri):
        raise ValueError(f"Expected gs:// URI, got {uri!r}")
    stripped = uri[len("gs://") :]
    bucket, _, prefix = stripped.partition("/")
    if not bucket:
        raise ValueError(f"Expected non-empty GCS bucket in {uri!r}")
    return bucket, prefix.strip("/")


def _run_gcloud_storage(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["gcloud", "--quiet", "storage", *args],
        check=True,
        text=True,
        capture_output=True,
    )


def _gcs_copy_file(source: str | os.PathLike[str], destination_uri: str) -> None:
    _run_gcloud_storage(["cp", str(source), destination_uri])


def _gcs_download_file(source_uri: str, target: str | os.PathLike[str]) -> None:
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    _run_gcloud_storage(["cp", source_uri, str(target_path)])


def _gcs_rsync(source_uri: str, target: str | os.PathLike[str]) -> None:
    shutil.rmtree(target, ignore_errors=True)
    Path(target).mkdir(parents=True, exist_ok=True)
    _run_gcloud_storage(["rsync", "-r", source_uri, str(target)])


def _gcs_object_size(uri: str) -> int | None:
    result = _run_gcloud_storage(["ls", "-l", uri])
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("TOTAL:"):
            continue
        first = stripped.split(None, 1)[0]
        try:
            return int(first)
        except ValueError:
            continue
    return None


def _gcs_list_objects(prefix_uri: str) -> list[dict[str, Any]]:
    bucket_name, prefix = _gcs_path_parts(prefix_uri)
    normalized_prefix = f"{prefix.rstrip('/')}/" if prefix else ""
    try:
        from google.cloud import storage

        client = storage.Client(project=_gcp_project_name())
        bucket = client.bucket(bucket_name)
        return [
            {"uri": f"gs://{bucket_name}/{blob.name}", "bytes": int(blob.size or 0)}
            for blob in client.list_blobs(bucket, prefix=normalized_prefix)
            if not blob.name.endswith("/")
        ]
    except Exception:
        result = _run_gcloud_storage(["ls", "-l", f"{prefix_uri.rstrip('/')}/**"])
        objects: list[dict[str, Any]] = []
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("TOTAL:"):
                continue
            parts = stripped.split()
            if len(parts) < 2 or not parts[-1].startswith("gs://"):
                continue
            try:
                size = int(parts[0])
            except ValueError:
                continue
            objects.append({"uri": parts[-1], "bytes": size})
        return objects


def _gcp_project_name() -> str:
    for name in (
        "DITTY_GCS_PROJECT",
        "GOOGLE_CLOUD_PROJECT",
        "GCLOUD_PROJECT",
        "GCP_PROJECT",
    ):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    try:
        import google.auth

        _, project = google.auth.default()
        if project:
            return str(project)
    except Exception:
        pass
    raise RuntimeError(
        "GCS checkpointing needs a GCP project. Set DITTY_GCS_PROJECT or "
        "configure Application Default Credentials with a project."
    )


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {parsed}")
    return parsed


@dataclass(frozen=True)
class _GcsComponent:
    name: str
    bytes: int
    generation: int | None = None


def _gcs_upload_component_bytes(
    *,
    project: str,
    bucket_name: str,
    object_name: str,
    data: bytes,
    timeout: int,
) -> _GcsComponent:
    from google.cloud import storage

    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_file(
        io.BytesIO(data),
        rewind=True,
        size=len(data),
        content_type="application/octet-stream",
        if_generation_match=0,
        timeout=timeout,
    )
    return _GcsComponent(
        name=object_name,
        bytes=len(data),
        generation=int(blob.generation) if blob.generation is not None else None,
    )


def _gcs_delete_components(
    *,
    project: str,
    bucket_name: str,
    components: list[_GcsComponent],
    timeout: int,
) -> None:
    if not components:
        return
    from google.cloud import storage

    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    for component in components:
        try:
            bucket.blob(component.name, generation=component.generation).delete(
                timeout=timeout,
            )
        except Exception:
            logger.warning(
                "Failed to delete temporary GCS checkpoint component gs://%s/%s",
                bucket_name,
                component.name,
                exc_info=True,
            )


def _gcs_compose_components(
    *,
    project: str,
    bucket_name: str,
    destination_name: str,
    components: list[_GcsComponent],
    timeout: int,
) -> _GcsComponent:
    from google.cloud import storage

    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    destination = bucket.blob(destination_name)
    sources = [
        bucket.blob(component.name, generation=component.generation)
        for component in components
    ]
    generations = [component.generation for component in components]
    destination.compose(
        sources,
        if_generation_match=0,
        if_source_generation_match=generations,
        timeout=timeout,
    )
    return _GcsComponent(
        name=destination_name,
        bytes=sum(component.bytes for component in components),
        generation=int(destination.generation) if destination.generation is not None else None,
    )


def _gcs_copy_component(
    *,
    project: str,
    bucket_name: str,
    destination_name: str,
    component: _GcsComponent,
    timeout: int,
) -> _GcsComponent:
    from google.cloud import storage

    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    source = bucket.blob(component.name, generation=component.generation)
    copied = bucket.copy_blob(
        source,
        bucket,
        new_name=destination_name,
        source_generation=component.generation,
        if_generation_match=0,
        if_source_generation_match=component.generation,
        timeout=timeout,
    )
    return _GcsComponent(
        name=destination_name,
        bytes=component.bytes,
        generation=int(copied.generation) if copied.generation is not None else None,
    )


class _GcsParallelCompositeUploadStream(io.RawIOBase):
    def __init__(
        self,
        *,
        project: str,
        bucket_name: str,
        object_name: str,
        component_size: int,
        max_workers: int,
        max_pending_bytes: int,
        timeout: int,
    ) -> None:
        super().__init__()
        self._project = project
        self._bucket_name = bucket_name
        self._object_name = object_name
        self._component_size = component_size
        self._max_workers = max_workers
        self._max_pending_bytes = max_pending_bytes
        self._timeout = timeout
        self._buffer = bytearray()
        self._pos = 0
        self._part_index = 0
        self._pending_bytes = 0
        self._futures: list[ConcurrentFuture[_GcsComponent]] = []
        self._components: dict[int, _GcsComponent] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="ditty-gcs-pcu",
        )
        self._closed_for_write = False
        self._temp_prefix = (
            f"{object_name.rstrip('/')}.ditty_tmp/"
            f"{uuid.uuid4().hex}"
        )
        self._all_temp_components: list[_GcsComponent] = []
        self._lock = threading.Lock()

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._pos

    def write(self, b) -> int:
        if self._closed_for_write:
            raise ValueError("I/O operation on closed GCS upload stream")
        view = memoryview(b)
        total = len(view)
        offset = 0
        while offset < total:
            remaining = self._component_size - len(self._buffer)
            take = min(remaining, total - offset)
            self._buffer.extend(view[offset : offset + take])
            offset += take
            if len(self._buffer) >= self._component_size:
                self._submit_component(bytes(self._buffer))
                self._buffer.clear()
                self._drain_completed(block=self._pending_bytes > self._max_pending_bytes)
        self._pos += total
        return total

    def flush(self) -> None:
        self._drain_completed(block=False)

    def close(self) -> None:
        if self.closed:
            return
        error: BaseException | None = None
        try:
            if not self._closed_for_write:
                self._closed_for_write = True
                if self._buffer:
                    self._submit_component(bytes(self._buffer))
                    self._buffer.clear()
                self._drain_all()
                components = [
                    self._components[index]
                    for index in sorted(self._components)
                ]
                self._compose_final_object(components)
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._executor.shutdown(wait=True, cancel_futures=error is not None)
            try:
                _gcs_delete_components(
                    project=self._project,
                    bucket_name=self._bucket_name,
                    components=self._all_temp_components,
                    timeout=self._timeout,
                )
            finally:
                super().close()

    def _submit_component(self, data: bytes) -> None:
        index = self._part_index
        self._part_index += 1
        object_name = f"{self._temp_prefix}/part-{index:08d}"
        self._pending_bytes += len(data)
        future = self._executor.submit(
            _gcs_upload_component_bytes,
            project=self._project,
            bucket_name=self._bucket_name,
            object_name=object_name,
            data=data,
            timeout=self._timeout,
        )
        future._ditty_component_index = index  # type: ignore[attr-defined]
        self._futures.append(future)

    def _drain_completed(self, *, block: bool) -> None:
        while self._futures:
            if block:
                done, _ = wait(self._futures, return_when=FIRST_COMPLETED)
            else:
                done = {future for future in self._futures if future.done()}
            if not done:
                return
            for future in done:
                component = future.result()
                index = int(getattr(future, "_ditty_component_index"))
                self._components[index] = component
                self._all_temp_components.append(component)
                self._pending_bytes -= component.bytes
                self._futures.remove(future)
            if not block or self._pending_bytes <= self._max_pending_bytes:
                return

    def _drain_all(self) -> None:
        while self._futures:
            self._drain_completed(block=True)

    def _compose_final_object(self, components: list[_GcsComponent]) -> None:
        if not components:
            component = _gcs_upload_component_bytes(
                project=self._project,
                bucket_name=self._bucket_name,
                object_name=f"{self._temp_prefix}/part-00000000",
                data=b"",
                timeout=self._timeout,
            )
            self._all_temp_components.append(component)
            components = [component]

        current = components
        level = 0
        while len(current) > 32:
            next_level: list[_GcsComponent] = []
            for group_index, start in enumerate(range(0, len(current), 32)):
                group = current[start : start + 32]
                composed = _gcs_compose_components(
                    project=self._project,
                    bucket_name=self._bucket_name,
                    destination_name=(
                        f"{self._temp_prefix}/compose-{level:02d}-{group_index:08d}"
                    ),
                    components=group,
                    timeout=self._timeout,
                )
                next_level.append(composed)
                self._all_temp_components.append(composed)
            current = next_level
            level += 1

        if len(current) == 1:
            _gcs_copy_component(
                project=self._project,
                bucket_name=self._bucket_name,
                destination_name=self._object_name,
                component=current[0],
                timeout=self._timeout,
            )
        else:
            _gcs_compose_components(
                project=self._project,
                bucket_name=self._bucket_name,
                destination_name=self._object_name,
                components=current,
                timeout=self._timeout,
            )


class _GcsParallelCompositeFileSystem:
    def __init__(
        self,
        *,
        project: str,
        component_size: int,
        max_workers: int,
        max_pending_bytes: int,
        timeout: int,
    ) -> None:
        self.project = project
        self.component_size = component_size
        self.max_workers = max_workers
        self.max_pending_bytes = max_pending_bytes
        self.timeout = timeout

    @staticmethod
    def _blob(path: str | os.PathLike[str]):
        bucket_name, object_name = _gcs_path_parts(str(path))
        return bucket_name, object_name

    def create_stream(self, path: str | os.PathLike[str], mode: str):
        if mode != "wb":
            raise ValueError(f"GCS DCP writer only supports wb streams, got {mode!r}")
        bucket_name, object_name = self._blob(path)
        return _GcsParallelCompositeUploadStream(
            project=self.project,
            bucket_name=bucket_name,
            object_name=object_name,
            component_size=self.component_size,
            max_workers=self.max_workers,
            max_pending_bytes=self.max_pending_bytes,
            timeout=self.timeout,
        )

    def concat_path(self, path: str | os.PathLike[str], suffix: str) -> str:
        return _uri_join(str(path), suffix)

    def init_path(self, path: str | os.PathLike[str], **_: Any) -> str:
        value = str(path).rstrip("/")
        _gcs_path_parts(value)
        return value

    def mkdir(self, path: str | os.PathLike[str]) -> None:
        _gcs_path_parts(str(path))

    def exists(self, path: str | os.PathLike[str]) -> bool:
        from google.cloud import storage

        bucket_name, object_name = self._blob(path)
        client = storage.Client(project=self.project)
        return client.bucket(bucket_name).blob(object_name).exists(timeout=self.timeout)

    def rm_file(self, path: str | os.PathLike[str]) -> None:
        from google.cloud import storage

        bucket_name, object_name = self._blob(path)
        client = storage.Client(project=self.project)
        client.bucket(bucket_name).blob(object_name).delete(timeout=self.timeout)

    def rename(self, path: str | os.PathLike[str], new_path: str | os.PathLike[str]) -> None:
        from google.cloud import storage

        source_bucket_name, source_name = self._blob(path)
        target_bucket_name, target_name = self._blob(new_path)
        if source_bucket_name != target_bucket_name:
            raise ValueError("GCS DCP writer cannot rename across buckets")
        client = storage.Client(project=self.project)
        bucket = client.bucket(source_bucket_name)
        source = bucket.blob(source_name)
        copied = bucket.copy_blob(
            source,
            bucket,
            new_name=target_name,
            if_generation_match=0,
            timeout=self.timeout,
        )
        source.delete(
            if_generation_match=int(source.generation) if source.generation is not None else None,
            timeout=self.timeout,
        )
        if copied.size is None:
            copied.reload(timeout=self.timeout)

    def ls(self, path: str | os.PathLike[str]) -> list[str]:
        bucket_name, prefix = self._blob(path)
        from google.cloud import storage

        client = storage.Client(project=self.project)
        return [
            f"gs://{bucket_name}/{blob.name}"
            for blob in client.list_blobs(
                client.bucket(bucket_name),
                prefix=f"{prefix.rstrip('/')}/",
            )
        ]


class _GcsParallelCompositeDcpWriter:
    def __new__(cls, checkpoint_uri: str):
        from torch.distributed.checkpoint.filesystem import FileSystemWriter

        component_size = _env_int(
            "DITTY_GCS_PARALLEL_COMPOSITE_COMPONENT_SIZE_MB",
            256,
        ) * 1024 * 1024
        max_workers = _env_int("DITTY_GCS_PARALLEL_COMPOSITE_MAX_WORKERS", 4)
        default_pending_mb = max(512, (component_size // (1024 * 1024)) * max_workers * 2)
        max_pending_bytes = _env_int(
            "DITTY_GCS_PARALLEL_COMPOSITE_MAX_PENDING_MB",
            default_pending_mb,
        ) * 1024 * 1024
        timeout = _env_int("DITTY_GCS_PARALLEL_COMPOSITE_TIMEOUT_S", 600)
        writer = FileSystemWriter(
            checkpoint_uri,
            single_file_per_rank=True,
            sync_files=False,
            thread_count=1,
            overwrite=True,
        )
        writer.fs = _GcsParallelCompositeFileSystem(
            project=_gcp_project_name(),
            component_size=component_size,
            max_workers=max_workers,
            max_pending_bytes=max_pending_bytes,
            timeout=timeout,
        )
        writer.path = writer.fs.init_path(checkpoint_uri)
        return writer


@dataclass
class Checkpoint:
    """Container for all checkpoint data."""
    checkpoint_num: Optional[int] = None
    path: Optional[str] = None
    model_state: Optional[Dict[str, Any]] = None
    optimizer_state: Optional[Dict[str, Any]] = None
    distributed_checkpoint_path: Optional[str] = None
    scheduler_state: Optional[Dict[str, Any]] = None
    training_state: Dict[str, Any] = field(default_factory=dict)
    scaler_state: Optional[Dict[str, Any]] = None
    rng_states: Dict[str, Any] = field(default_factory=dict)
    loss_state: Optional[Dict[str, Any]] = None
    loss_optimizer_state: Optional[Dict[str, Any]] = None  # optimizer state for loss_calculator params
    preprocessor_states: Optional[List[Dict[str, Any]]] = None


class CheckpointManager:
    """
    Unified checkpoint manager for ditty training.

    Handles saving and loading of:
    - Model weights
    - Optimizer state
    - Scheduler state
    - Training state (epoch, steps, etc.)
    - Gradient scaler state
    - RNG states for reproducibility

    This replaces accelerate's save_state/load_state to give us control
    over the loading order (load before prepare() instead of after).
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.checkpoints_dir = os.path.join(output_dir, "checkpoints")

    def _get_checkpoint_path(self, checkpoint_num: int) -> str:
        return os.path.join(self.checkpoints_dir, f"checkpoint_{checkpoint_num}")

    def get_checkpoint_path(self, checkpoint_num: int) -> str:
        return self._get_checkpoint_path(checkpoint_num)

    def _checkpoint_dir_nums(self) -> list[int]:
        if not os.path.exists(self.checkpoints_dir):
            return []
        nums: list[int] = []
        for name in os.listdir(self.checkpoints_dir):
            if not name.startswith("checkpoint_"):
                continue
            try:
                nums.append(int(name.split("_", 1)[1]))
            except ValueError:
                continue
        return sorted(nums)

    def prune_old_checkpoints(self, keep_last: int | None = None) -> list[int]:
        if keep_last is None:
            value = os.environ.get("DITTY_KEEP_LAST_CHECKPOINTS", "").strip()
            if not value:
                return []
            keep_last = int(value)
        if keep_last <= 0 or not os.path.exists(self.checkpoints_dir):
            return []

        valid_nums: list[int] = []
        for num in self._checkpoint_dir_nums():
            checkpoint_path = self._get_checkpoint_path(num)
            has_training_state = (
                os.path.exists(os.path.join(checkpoint_path, "training_state.pt"))
                or os.path.exists(os.path.join(checkpoint_path, "custom_checkpoint_0.pkl"))
            )
            if has_training_state:
                valid_nums.append(num)
        if len(valid_nums) <= keep_last:
            return []

        keep = set(valid_nums[-keep_last:])
        oldest_kept = min(keep)
        removed: list[int] = []
        for num in self._checkpoint_dir_nums():
            if num in keep or num >= oldest_kept:
                continue
            checkpoint_path = self._get_checkpoint_path(num)
            shutil.rmtree(checkpoint_path, ignore_errors=True)
            removed.append(num)
        if removed and _should_log_rank_zero_only():
            logger.info(
                "Pruned old checkpoints from %s, kept last %s complete checkpoints: removed=%s",
                self.checkpoints_dir,
                keep_last,
                removed,
            )
        return removed

    @staticmethod
    def read_ray_train_checkpoint_num(checkpoint_dir: str) -> Optional[int]:
        metadata_path = os.path.join(checkpoint_dir, "ditty_checkpoint_metadata.json")
        if not os.path.exists(metadata_path):
            return None
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        value = metadata.get("checkpoint_num")
        return int(value) if value is not None else None

    def _get_latest_checkpoint_num(self) -> Optional[int]:
        if not os.path.exists(self.checkpoints_dir):
            return None

        checkpoint_dirs = []
        for name in os.listdir(self.checkpoints_dir):
            if name.startswith("checkpoint_"):
                try:
                    num = int(name.split("_")[1])
                    checkpoint_path = self._get_checkpoint_path(num)
                    has_training_state = (
                        os.path.exists(os.path.join(checkpoint_path, "training_state.pt"))
                        or os.path.exists(os.path.join(checkpoint_path, "custom_checkpoint_0.pkl"))
                    )
                    if not has_training_state:
                        continue
                    if self.ray_train_reporting_enabled() and not os.path.exists(
                        os.path.join(checkpoint_path, "ditty_checkpoint_metadata.json")
                    ):
                        continue
                    checkpoint_dirs.append(num)
                except (IndexError, ValueError):
                    continue

        if not checkpoint_dirs:
            return None

        return max(checkpoint_dirs)

    def save(
        self,
        checkpoint_num: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        training_state: Dict[str, Any],
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
        loss_calculator: Optional[Any] = None,
        preprocessors: Optional[List[Any]] = None,
        is_fsdp: bool = False,
        rank: int = 0,
        local_rank: int = 0,
    ):
        """Save a complete training checkpoint."""
        checkpoint_path = self._get_checkpoint_path(checkpoint_num)
        os.makedirs(checkpoint_path, exist_ok=True)
        _dist_debug(f"checkpoint save start path={checkpoint_path} is_fsdp={is_fsdp}")
        distributed_checkpoint_path = os.path.join(checkpoint_path, "distributed")
        skip_optimizer_checkpoint = os.environ.get(
            "DITTY_SKIP_OPTIMIZER_CHECKPOINT", ""
        ).lower() in {"1", "true", "yes", "on"}
        dcp_process_group_backend = os.environ.get("DITTY_DCP_PROCESS_GROUP", "").lower()
        dcp_process_group = None

        # Save model weights (full state dict for FSDP)
        model_path = os.path.join(self.output_dir, "dist", "model.pt")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Get the unwrapped model if compiled (avoids _orig_mod. prefix in state_dict)
        unwrapped_model = getattr(model, '_orig_mod', model)

        if is_fsdp:
            try:
                from torch.distributed.checkpoint import save as dcp_save
                from torch.distributed.checkpoint.state_dict import (
                    get_model_state_dict,
                )
                _dist_debug("capturing sharded model state dict via get_model_state_dict")
                model_state = get_model_state_dict(unwrapped_model)
                _dist_debug(
                    f"captured model state dict entries={len(model_state)}"
                )
            except ImportError:
                # Fallback for older torch versions
                _dist_debug("torch.distributed.checkpoint unavailable, falling back to state_dict()")
                model_state = {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in unwrapped_model.state_dict().items()
                }

            if "dcp_save" in locals():
                distributed_state = {"model": model_state}
            elif rank == 0 and model_state:
                _dist_debug(f"rank0 writing fallback model state to {model_path}")
                torch.save(model_state, model_path)
                _dist_debug("rank0 finished writing fallback model state")
        else:
            if rank == 0:
                _dist_debug(f"rank0 writing non-FSDP model state to {model_path}")
                torch.save(unwrapped_model.state_dict(), model_path)
                _dist_debug("rank0 finished writing non-FSDP model state")

        # Save optimizer state
        # When using FSDP with loss_calculator params in the optimizer, we need to
        # save them separately because get_optimizer_state_dict only handles model params.
        optimizer_path = os.path.join(checkpoint_path, "optimizer.bin")
        loss_optim_path = os.path.join(checkpoint_path, "loss_optimizer_state.pt")

        optim_state = None
        if skip_optimizer_checkpoint:
            _dist_debug("skipping optimizer checkpoint because DITTY_SKIP_OPTIMIZER_CHECKPOINT is enabled")
        elif is_fsdp:
            try:
                from torch.distributed.checkpoint.state_dict import (
                    get_optimizer_state_dict,
                )
                from torch.distributed.tensor import DTensor
                _dist_debug("preparing optimizer state extraction for FSDP")

                # Get loss_calculator params in order (if any)
                loss_params = []
                if loss_calculator is not None and hasattr(loss_calculator, 'parameters'):
                    loss_params = list(loss_calculator.parameters())

                if loss_params:
                    # Extract optimizer state for loss params before calling get_optimizer_state_dict
                    full_optim_state = optimizer.state_dict()
                    loss_optim_state = {"state": {}}

                    # Map param id to index in optimizer
                    param_id_to_idx = {}
                    idx = 0
                    for group in optimizer.param_groups:
                        for p in group["params"]:
                            param_id_to_idx[id(p)] = idx
                            idx += 1

                    # Extract state for loss params in order
                    loss_param_indices = []
                    for p in loss_params:
                        pid = id(p)
                        if pid in param_id_to_idx:
                            param_idx = param_id_to_idx[pid]
                            loss_param_indices.append(param_idx)
                            if param_idx in full_optim_state["state"]:
                                state_entry = full_optim_state["state"][param_idx]
                                # Convert DTensors to regular tensors
                                converted_state = {}
                                for k, v in state_entry.items():
                                    if isinstance(v, DTensor):
                                        converted_state[k] = v.full_tensor().cpu()
                                    elif isinstance(v, torch.Tensor):
                                        converted_state[k] = v.cpu()
                                    else:
                                        converted_state[k] = v
                                loss_optim_state["state"][param_idx] = converted_state

                    loss_optim_state["param_indices"] = loss_param_indices
                    loss_param_ids = {id(p) for p in loss_params}

                    if rank == 0:
                        _dist_debug(f"rank0 writing loss optimizer sidecar to {loss_optim_path}")
                        torch.save(loss_optim_state, loss_optim_path)
                        _dist_debug("rank0 finished writing loss optimizer sidecar")

                    # Temporarily remove loss params from optimizer.param_groups and optimizer.state
                    # so get_optimizer_state_dict doesn't try to map them
                    popped_params = []
                    for group in optimizer.param_groups:
                        original_params = group["params"]
                        filtered = [p for p in original_params if id(p) not in loss_param_ids]
                        removed = [p for p in original_params if id(p) in loss_param_ids]
                        popped_params.append((group, original_params, removed))
                        group["params"] = filtered

                    # Also pop from optimizer.state
                    popped_state = {}
                    for p in loss_params:
                        if p in optimizer.state:
                            popped_state[p] = optimizer.state.pop(p)

                # Now get model optimizer state
                _dist_debug("capturing sharded optimizer state dict via get_optimizer_state_dict")
                optim_state = get_optimizer_state_dict(model, optimizer)
                _dist_debug(
                    f"captured optimizer state dict keys={list(optim_state.keys()) if isinstance(optim_state, dict) else type(optim_state)}"
                )

                # Restore the popped params and state
                if loss_params:
                    for group, original_params, _ in popped_params:
                        group["params"] = original_params
                    for p, state in popped_state.items():
                        optimizer.state[p] = state
            except ImportError:
                _dist_debug("optimizer distributed checkpoint APIs unavailable, falling back to optimizer.state_dict()")
                optim_state = optimizer.state_dict()
        else:
            optim_state = optimizer.state_dict()

        if is_fsdp and "dcp_save" in locals():
            if optim_state is not None:
                distributed_state["optimizer"] = optim_state
            direct_distributed_uri: str | None = None
            direct_storage_writer = None
            try:
                if dcp_process_group_backend:
                    import torch.distributed as dist
                    _dist_debug(
                        "creating DCP process group "
                        f"backend={dcp_process_group_backend}"
                    )
                    dcp_process_group = dist.new_group(backend=dcp_process_group_backend)
                direct_distributed_uri, direct_storage_writer = self._direct_dcp_writer_for_checkpoint(
                    checkpoint_num
                )
                dcp_checkpoint_id = direct_distributed_uri or distributed_checkpoint_path
                _dist_debug(
                    "saving distributed checkpoint state_keys="
                    f"{sorted(distributed_state)} to {dcp_checkpoint_id}"
                )
                dcp_save(
                    distributed_state,
                    checkpoint_id=dcp_checkpoint_id,
                    storage_writer=direct_storage_writer,
                    process_group=dcp_process_group,
                )
                _dist_debug("finished saving distributed checkpoint")
                if direct_distributed_uri:
                    self._write_distributed_checkpoint_pointer(
                        checkpoint_path,
                        distributed_uri=direct_distributed_uri,
                    )
            finally:
                if dcp_process_group is not None:
                    import torch.distributed as dist
                    dist.destroy_process_group(dcp_process_group)
                    _dist_debug("destroyed DCP process group")
        elif rank == 0 and optim_state is not None:
            _dist_debug(f"rank0 writing optimizer state to {optimizer_path}")
            torch.save(optim_state, optimizer_path)
            _dist_debug("rank0 finished writing optimizer state")

        # Save scheduler state
        if scheduler is not None and rank == 0:
            scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
            _dist_debug(f"rank0 writing scheduler state to {scheduler_path}")
            torch.save(scheduler.state_dict(), scheduler_path)
            _dist_debug("rank0 finished writing scheduler state")

        if preprocessors is not None and rank == 0:
            preprocessor_states = []
            for index, preprocessor in enumerate(preprocessors):
                state_fn = getattr(preprocessor, "state_dict", None)
                if not callable(state_fn):
                    continue
                state = state_fn()
                if not state:
                    continue
                preprocessor_states.append(
                    {
                        "index": index,
                        "class": preprocessor.__class__.__name__,
                        "name": getattr(preprocessor, "name", preprocessor.__class__.__name__),
                        "state": state,
                    }
                )
            if preprocessor_states:
                preprocessor_path = os.path.join(checkpoint_path, "preprocessors.pt")
                _dist_debug(f"rank0 writing preprocessor state to {preprocessor_path}")
                torch.save(preprocessor_states, preprocessor_path)
                _dist_debug("rank0 finished writing preprocessor state")

        # Save training state
        if rank == 0:
            training_state_path = os.path.join(checkpoint_path, "training_state.pt")
            _dist_debug(f"rank0 writing training state to {training_state_path}")
            torch.save(training_state, training_state_path)
            _dist_debug("rank0 finished writing training state")

        # Save scaler state
        if scaler is not None and rank == 0:
            scaler_path = os.path.join(checkpoint_path, "scaler.pt")
            _dist_debug(f"rank0 writing scaler state to {scaler_path}")
            torch.save(scaler.state_dict(), scaler_path)
            _dist_debug("rank0 finished writing scaler state")

        # Save loss calculator state (if it has learnable parameters)
        # Note: full_tensor() is a collective op, so all ranks must call it together
        if loss_calculator is not None:
            loss_state = loss_calculator.state_dict()
            if loss_state:  # Only save if non-empty
                # Convert DTensors to full tensors for saving
                from torch.distributed.tensor import DTensor
                converted_state = {}
                _dist_debug(f"processing loss calculator state entries={len(loss_state)}")
                for k, v in loss_state.items():
                    if isinstance(v, DTensor):
                        # full_tensor() is collective - all ranks must call
                        _dist_debug(f"materializing DTensor loss state entry={k}")
                        converted_state[k] = v.full_tensor().cpu()
                    elif isinstance(v, torch.Tensor):
                        converted_state[k] = v.cpu()
                    else:
                        converted_state[k] = v
                # Only rank 0 saves to disk
                if rank == 0:
                    loss_path = os.path.join(checkpoint_path, "loss_state.pt")
                    _dist_debug(f"rank0 writing loss state to {loss_path}")
                    torch.save(converted_state, loss_path)
                    _dist_debug("rank0 finished writing loss state")

        # Save RNG states for this rank
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state(local_rank)

        rng_path = os.path.join(checkpoint_path, f"rng_state_{rank}.pt")
        _dist_debug(f"writing RNG state to {rng_path}")
        torch.save(rng_state, rng_path)
        _dist_debug("finished writing RNG state")

        if rank == 0:
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        _dist_debug(f"checkpoint save end path={checkpoint_path}")

    def load(self, checkpoint_num: Optional[int] = None) -> Optional[Checkpoint]:
        """
        Load a checkpoint. If checkpoint_num is None, loads the latest.
        Returns None if no checkpoint exists.
        """
        if checkpoint_num is None:
            checkpoint_num = self._get_latest_checkpoint_num()
            if checkpoint_num is None:
                return None

        checkpoint_path = self._get_checkpoint_path(checkpoint_num)
        if not os.path.exists(checkpoint_path):
            return None

        checkpoint = Checkpoint(checkpoint_num=checkpoint_num, path=checkpoint_path)
        distributed_checkpoint_path = os.path.join(checkpoint_path, "distributed")
        if os.path.exists(distributed_checkpoint_path):
            checkpoint.distributed_checkpoint_path = distributed_checkpoint_path
        else:
            direct_distributed_uri = self.read_distributed_checkpoint_pointer(checkpoint_path)
            if direct_distributed_uri:
                checkpoint.distributed_checkpoint_path = direct_distributed_uri

        # Load model weights
        model_path = os.path.join(self.output_dir, "dist", "model.pt")
        if checkpoint.distributed_checkpoint_path is None and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
            # Strip _orig_mod. prefix added by torch.compile
            checkpoint.model_state = {
                k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
            }

        # Load optimizer state
        optimizer_path = os.path.join(checkpoint_path, "optimizer.bin")
        if checkpoint.distributed_checkpoint_path is None and os.path.exists(optimizer_path):
            checkpoint.optimizer_state = torch.load(optimizer_path, map_location="cpu", weights_only=False)

        # Load scheduler state
        scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
        if os.path.exists(scheduler_path):
            checkpoint.scheduler_state = torch.load(scheduler_path, map_location="cpu", weights_only=False)

        preprocessor_path = os.path.join(checkpoint_path, "preprocessors.pt")
        if os.path.exists(preprocessor_path):
            checkpoint.preprocessor_states = torch.load(
                preprocessor_path,
                map_location="cpu",
                weights_only=False,
            )

        # Load training state (new format or legacy accelerate format)
        training_state_path = os.path.join(checkpoint_path, "training_state.pt")
        if os.path.exists(training_state_path):
            checkpoint.training_state = torch.load(training_state_path, map_location="cpu", weights_only=False)
        else:
            # Try legacy accelerate format
            legacy_path = os.path.join(checkpoint_path, "custom_checkpoint_0.pkl")
            if os.path.exists(legacy_path):
                try:
                    checkpoint.training_state = torch.load(legacy_path, map_location="cpu", weights_only=False)
                    logger.info("Loaded training state from legacy accelerate format")
                except Exception as e:
                    logger.warning(f"Failed to load legacy training state: {e}")

        # Load scaler state
        scaler_path = os.path.join(checkpoint_path, "scaler.pt")
        if os.path.exists(scaler_path):
            checkpoint.scaler_state = torch.load(scaler_path, map_location="cpu", weights_only=False)

        # Load loss calculator state
        loss_path = os.path.join(checkpoint_path, "loss_state.pt")
        if os.path.exists(loss_path):
            checkpoint.loss_state = torch.load(loss_path, map_location="cpu", weights_only=False)

        # Load loss optimizer state (for loss_calculator params)
        loss_optim_path = os.path.join(checkpoint_path, "loss_optimizer_state.pt")
        if os.path.exists(loss_optim_path):
            checkpoint.loss_optimizer_state = torch.load(loss_optim_path, map_location="cpu", weights_only=False)

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint

    def load_rng_state(self, checkpoint_num: Optional[int] = None, rank: int = 0, local_rank: int = 0):
        """Load and restore RNG states for a specific rank."""
        if checkpoint_num is None:
            checkpoint_num = self._get_latest_checkpoint_num()
            if checkpoint_num is None:
                return

        checkpoint_path = self._get_checkpoint_path(checkpoint_num)
        rng_path = os.path.join(checkpoint_path, f"rng_state_{rank}.pt")

        if not os.path.exists(rng_path):
            # Try legacy format
            rng_path = os.path.join(checkpoint_path, f"random_states_{rank}.pkl")
            if not os.path.exists(rng_path):
                return

        rng_state = torch.load(rng_path, map_location="cpu", weights_only=False)

        # Handle our new format
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"])
        if "cuda" in rng_state and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng_state["cuda"], local_rank)

        # Handle accelerate format
        if "random_state" in rng_state:
            random.setstate(rng_state["random_state"])
        if "numpy_random_seed" in rng_state:
            np.random.set_state(rng_state["numpy_random_seed"])
        if "torch_manual_seed" in rng_state:
            torch.set_rng_state(rng_state["torch_manual_seed"])
        if "torch_cuda_manual_seed" in rng_state and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng_state["torch_cuda_manual_seed"], local_rank)

    def get_latest_checkpoint_num(self) -> Optional[int]:
        return self._get_latest_checkpoint_num()

    def ray_train_reporting_enabled(self) -> bool:
        value = os.environ.get("DITTY_RAY_TRAIN_REPORT_CHECKPOINTS", "")
        return value.lower() in {"1", "true", "yes", "on"}

    def ray_train_durable_root(self) -> str | None:
        value = os.environ.get("DITTY_RAY_TRAIN_DURABLE_ROOT", "").strip()
        return value or None

    def direct_dcp_to_gcs_mode(self) -> str:
        value = os.environ.get("DITTY_RAY_TRAIN_DIRECT_DCP_TO_GCS", "").strip().lower()
        if not value:
            value = os.environ.get("DITTY_DCP_DIRECT_GCS", "").strip().lower()
        if not value or value in {"0", "false", "no", "off"}:
            return "off"
        if value in {"auto"}:
            return "auto"
        if value in {"1", "true", "yes", "on", "required"}:
            return "required"
        raise ValueError(
            "DITTY_RAY_TRAIN_DIRECT_DCP_TO_GCS must be one of off, auto, or required; "
            f"got {value!r}"
        )

    def ray_train_report_mode(self, checkpoint_num: int) -> str:
        value = os.environ.get("DITTY_RAY_TRAIN_CHECKPOINT_MODE", "auto").strip().lower()
        if value not in {"auto", "full", "pointer"}:
            raise ValueError(
                "DITTY_RAY_TRAIN_CHECKPOINT_MODE must be one of auto, full, or pointer; "
                f"got {value!r}"
            )
        if value != "auto":
            return value

        durable_root = self.ray_train_durable_root()
        checkpoint_path = self._get_checkpoint_path(checkpoint_num)
        has_distributed_shards = os.path.isdir(
            os.path.join(checkpoint_path, "distributed")
        ) or bool(self.read_distributed_checkpoint_pointer(checkpoint_path))
        if _is_gcs_uri(durable_root) and has_distributed_shards:
            return "pointer"
        return "full"

    def _direct_dcp_distributed_uri(self, checkpoint_num: int) -> str | None:
        durable_root = self.ray_train_durable_root()
        if not _is_gcs_uri(durable_root):
            return None
        return _uri_join(
            str(durable_root),
            f"ditty_checkpoint_{int(checkpoint_num)}",
            "distributed",
        )

    def _make_gcs_dcp_writer(self, checkpoint_uri: str):
        writer = os.environ.get(
            "DITTY_RAY_TRAIN_DIRECT_DCP_GCS_WRITER",
            "parallel_composite",
        ).strip().lower()
        if writer in {"parallel_composite", "pcu", "streaming_parallel_composite"}:
            _dist_debug(
                "using direct GCS DCP writer=parallel_composite "
                "component_size_mb="
                f"{os.environ.get('DITTY_GCS_PARALLEL_COMPOSITE_COMPONENT_SIZE_MB', '256')} "
                "max_workers="
                f"{os.environ.get('DITTY_GCS_PARALLEL_COMPOSITE_MAX_WORKERS', '4')}"
            )
            return _GcsParallelCompositeDcpWriter(checkpoint_uri)
        if writer not in {"fsspec", "fsspec_gcs"}:
            raise ValueError(
                "DITTY_RAY_TRAIN_DIRECT_DCP_GCS_WRITER must be one of "
                "parallel_composite or fsspec; "
                f"got {writer!r}"
            )
        _dist_debug("using direct GCS DCP writer=fsspec")
        import gcsfs  # noqa: F401 - registers the gs:// fsspec implementation.
        from torch.distributed.checkpoint import _fsspec_filesystem as fsspec_dcp

        return fsspec_dcp.FsspecWriter(checkpoint_uri, sync_files=False)

    def _make_gcs_dcp_reader(self, checkpoint_uri: str):
        import gcsfs  # noqa: F401 - registers the gs:// fsspec implementation.
        from torch.distributed.checkpoint import _fsspec_filesystem as fsspec_dcp

        return fsspec_dcp.FsspecReader(checkpoint_uri)

    def _direct_dcp_writer_for_checkpoint(self, checkpoint_num: int) -> tuple[str | None, Any | None]:
        mode = self.direct_dcp_to_gcs_mode()
        if mode == "off":
            return None, None
        checkpoint_uri = self._direct_dcp_distributed_uri(checkpoint_num)
        if checkpoint_uri is None:
            if mode == "required":
                raise RuntimeError(
                    "DITTY_RAY_TRAIN_DIRECT_DCP_TO_GCS is enabled but "
                    "DITTY_RAY_TRAIN_DURABLE_ROOT is not a gs:// URI."
                )
            return None, None
        try:
            return checkpoint_uri, self._make_gcs_dcp_writer(checkpoint_uri)
        except Exception:
            if mode == "required":
                raise
            if _should_log_rank_zero_only():
                logger.warning(
                    "Direct DCP-to-GCS checkpointing is unavailable; falling back to "
                    "local DCP checkpoint plus durable upload.",
                    exc_info=True,
                )
            return None, None

    def _write_distributed_checkpoint_pointer(
        self,
        checkpoint_path: str,
        *,
        distributed_uri: str,
    ) -> str:
        pointer_path = os.path.join(checkpoint_path, _DIRECT_DISTRIBUTED_CHECKPOINT_POINTER)
        with open(pointer_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "format": "ditty.distributed_checkpoint_pointer.v1",
                    "distributed_uri": distributed_uri,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        return pointer_path

    @staticmethod
    def read_distributed_checkpoint_pointer(checkpoint_path: str) -> str | None:
        pointer_path = os.path.join(checkpoint_path, _DIRECT_DISTRIBUTED_CHECKPOINT_POINTER)
        if not os.path.exists(pointer_path):
            return None
        with open(pointer_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        distributed_uri = str(payload.get("distributed_uri") or "")
        return distributed_uri or None

    def write_ray_train_metadata(
        self,
        *,
        checkpoint_num: int,
        training_state: Dict[str, Any],
        world_size: int,
    ) -> str:
        checkpoint_path = self._get_checkpoint_path(checkpoint_num)
        metadata = {
            "checkpoint_num": int(checkpoint_num),
            "training_state": dict(training_state),
            "world_size": int(world_size),
            "format": "ditty.checkpoint.v1",
        }
        metadata_path = os.path.join(checkpoint_path, "ditty_checkpoint_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return metadata_path

    @staticmethod
    def read_ray_checkpoint_pointer(checkpoint_dir: str) -> dict[str, Any] | None:
        pointer_path = os.path.join(checkpoint_dir, "ditty_ray_checkpoint_pointer.json")
        if not os.path.exists(pointer_path):
            return None
        with open(pointer_path, "r", encoding="utf-8") as handle:
            pointer = json.load(handle)
        return dict(pointer)

    def _checkpoint_files_for_rank(
        self,
        checkpoint_path: str,
        *,
        rank: int,
    ) -> list[Path]:
        root = Path(checkpoint_path)
        files: list[Path] = []
        rank_rng = re.compile(r"rng_state_(\d+)\.pt$")
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            name = path.name
            if rel.startswith("distributed/") and name.endswith(".distcp"):
                if name.startswith(f"__{int(rank)}_"):
                    files.append(path)
                continue
            match = rank_rng.match(name)
            if match is not None:
                if int(match.group(1)) == int(rank):
                    files.append(path)
                continue
            if rank == 0:
                files.append(path)
        return files

    def _write_ray_pointer_checkpoint(
        self,
        *,
        checkpoint_num: int,
        training_state: Dict[str, Any],
        world_size: int,
        durable_uri: str,
    ) -> str:
        pointer_dir = os.path.join(
            self.output_dir,
            ".ray_train_checkpoint_pointers",
            f"ditty_ray_checkpoint_{int(checkpoint_num)}",
        )
        shutil.rmtree(pointer_dir, ignore_errors=True)
        os.makedirs(pointer_dir, exist_ok=True)
        pointer = {
            "format": "ditty.ray_checkpoint_pointer.v1",
            "checkpoint_num": int(checkpoint_num),
            "durable_uri": durable_uri,
            "training_state": dict(training_state),
            "world_size": int(world_size),
        }
        metadata = {
            "checkpoint_num": int(checkpoint_num),
            "training_state": dict(training_state),
            "world_size": int(world_size),
            "format": "ditty.checkpoint.v1",
            "ray_checkpoint_pointer": pointer,
        }
        with open(os.path.join(pointer_dir, "ditty_ray_checkpoint_pointer.json"), "w", encoding="utf-8") as handle:
            json.dump(pointer, handle, indent=2, sort_keys=True)
            handle.write("\n")
        with open(os.path.join(pointer_dir, "ditty_checkpoint_metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return pointer_dir

    def upload_durable_checkpoint(
        self,
        *,
        checkpoint_num: int,
        training_state: Dict[str, Any],
        rank: int,
        world_size: int,
        durable_root: str | None = None,
    ) -> str:
        durable_root = durable_root or self.ray_train_durable_root()
        if not _is_gcs_uri(durable_root):
            raise ValueError(
                "Durable Ray-compatible Ditty checkpoints currently require a gs:// "
                "DITTY_RAY_TRAIN_DURABLE_ROOT."
            )
        checkpoint_path = self._get_checkpoint_path(checkpoint_num)
        target_uri = _uri_join(str(durable_root), f"ditty_checkpoint_{int(checkpoint_num)}")
        local_files = self._checkpoint_files_for_rank(checkpoint_path, rank=rank)
        uploaded_files: list[dict[str, Any]] = []
        local_error: str | None = None
        try:
            for path in local_files:
                rel = path.relative_to(checkpoint_path).as_posix()
                size = int(path.stat().st_size)
                destination_uri = _uri_join(target_uri, rel)
                _dist_debug(f"uploading durable checkpoint file {rel} bytes={size} to {destination_uri}")
                _gcs_copy_file(path, destination_uri)
                remote_size = _gcs_object_size(destination_uri)
                if remote_size != size:
                    raise RuntimeError(
                        "Durable checkpoint upload validation failed for "
                        f"{destination_uri}: expected {size} bytes, got {remote_size}"
                    )
                uploaded_files.append({"path": rel, "bytes": size, "rank": int(rank)})
        except Exception as error:
            local_error = f"rank {rank} {error.__class__.__name__}: {error}"
        local_payload = {"files": uploaded_files, "error": local_error}

        gathered: list[Any] | None = None
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized() and int(world_size) > 1:
                gathered = [None for _ in range(int(world_size))] if rank == 0 else None
                dist.gather_object(local_payload, gathered, dst=0)
                dist.barrier()
        except Exception:
            gathered = None

        if gathered is None:
            gathered = [local_payload]

        error_message: str | None = None
        if rank == 0:
            try:
                upload_errors = [
                    str(payload.get("error"))
                    for payload in gathered
                    if isinstance(payload, dict) and payload.get("error")
                ]
                if upload_errors:
                    raise RuntimeError("; ".join(upload_errors))
                files = [
                    dict(item)
                    for payload in gathered
                    for item in (payload.get("files") if isinstance(payload, dict) else payload or [])
                    if isinstance(item, dict)
                ]
                direct_distributed_uri = self.read_distributed_checkpoint_pointer(checkpoint_path)
                if direct_distributed_uri:
                    if not direct_distributed_uri.startswith(_uri_join(target_uri, "distributed")):
                        raise RuntimeError(
                            "Direct distributed checkpoint URI does not match durable target: "
                            f"{direct_distributed_uri!r} vs {target_uri!r}"
                        )
                    for item in _gcs_list_objects(direct_distributed_uri):
                        uri = str(item.get("uri") or "")
                        if not uri.startswith(f"{target_uri.rstrip('/')}/"):
                            continue
                        files.append(
                            {
                                "path": uri[len(target_uri.rstrip('/')) + 1 :],
                                "bytes": int(item.get("bytes") or 0),
                                "rank": "direct_gcs",
                            }
                        )
                files_by_path = {
                    str(item.get("path")): dict(item)
                    for item in files
                    if item.get("path")
                }
                files = list(files_by_path.values())
                rels = {str(item.get("path")) for item in files}
                has_dcp = any(path.startswith("distributed/") and path.endswith(".distcp") for path in rels)
                if has_dcp:
                    missing = [
                        rank_index
                        for rank_index in range(int(world_size))
                        if not any(path.startswith(f"distributed/__{rank_index}_") and path.endswith(".distcp") for path in rels)
                    ]
                    if missing:
                        raise RuntimeError(
                            "Durable checkpoint upload is missing distributed shard files "
                            f"for ranks {missing}"
                        )
                    if "distributed/.metadata" not in rels:
                        raise RuntimeError("Durable checkpoint upload is missing distributed/.metadata")
                manifest = {
                    "format": "ditty.durable_checkpoint_manifest.v1",
                    "checkpoint_num": int(checkpoint_num),
                    "training_state": dict(training_state),
                    "world_size": int(world_size),
                    "files": sorted(files, key=lambda item: str(item.get("path"))),
                    "complete": True,
                }
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_path = Path(tmp)
                    manifest_path = tmp_path / "ditty_durable_manifest.json"
                    complete_path = tmp_path / ".complete"
                    manifest_path.write_text(
                        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    complete_path.write_text("complete\n", encoding="utf-8")
                    _gcs_copy_file(manifest_path, _uri_join(target_uri, "ditty_durable_manifest.json"))
                    _gcs_copy_file(complete_path, _uri_join(target_uri, ".complete"))
            except Exception as error:
                error_message = f"{error.__class__.__name__}: {error}"

        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized() and int(world_size) > 1:
                errors = [error_message]
                dist.broadcast_object_list(errors, src=0)
                error_message = str(errors[0]) if errors[0] else None
                dist.barrier()
        except Exception:
            pass
        if error_message:
            raise RuntimeError(error_message)

        return target_uri

    @staticmethod
    def restore_durable_checkpoint(durable_uri: str, target_root: str | os.PathLike[str]) -> int | None:
        if not _is_gcs_uri(durable_uri):
            return None
        if os.environ.get("DITTY_DURABLE_RESTORE_DOWNLOAD_DISTRIBUTED", "").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            checkpoint_num = CheckpointManager._restore_direct_gcs_dcp_pointer_checkpoint(
                durable_uri,
                target_root,
            )
            if checkpoint_num is not None:
                return checkpoint_num
        target_root = Path(target_root)
        tmp_target = target_root / "checkpoints" / ".ditty_durable_restore_tmp"
        _gcs_rsync(durable_uri, tmp_target)
        try:
            manifest_path = tmp_target / "ditty_durable_manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for item in manifest.get("files", []):
                    rel = str(item.get("path") or "")
                    expected_size = int(item.get("bytes", -1))
                    if not rel or expected_size < 0:
                        continue
                    local_path = tmp_target / rel
                    if not local_path.exists() or int(local_path.stat().st_size) != expected_size:
                        raise RuntimeError(
                            "Restored durable checkpoint failed manifest validation: "
                            f"{rel} expected {expected_size} bytes"
                        )
            checkpoint_num = CheckpointManager.read_ray_train_checkpoint_num(str(tmp_target))
            if checkpoint_num is None and manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                value = manifest.get("checkpoint_num")
                checkpoint_num = int(value) if value is not None else None
            if checkpoint_num is None:
                return None
            target = target_root / "checkpoints" / f"checkpoint_{checkpoint_num}"
            shutil.rmtree(target, ignore_errors=True)
            tmp_target.rename(target)
            return int(checkpoint_num)
        finally:
            shutil.rmtree(tmp_target, ignore_errors=True)

    @staticmethod
    def _restore_direct_gcs_dcp_pointer_checkpoint(
        durable_uri: str,
        target_root: str | os.PathLike[str],
    ) -> int | None:
        """Restore a direct-GCS DCP checkpoint as local sidecars plus a GCS pointer.

        The distributed checkpoint shards are already durable GCS objects. Pulling
        them back to local disk makes spot recovery and eval startup spend minutes
        copying tens of GiB for no benefit, so direct-GCS checkpoints restore as a
        small local Ditty checkpoint directory whose DCP pointer still targets GCS.
        """

        target_root = Path(target_root)
        tmp_target = target_root / "checkpoints" / ".ditty_durable_restore_tmp"
        shutil.rmtree(tmp_target, ignore_errors=True)
        tmp_target.mkdir(parents=True, exist_ok=True)
        manifest_path = tmp_target / "ditty_durable_manifest.json"
        try:
            _gcs_download_file(_uri_join(durable_uri, "ditty_durable_manifest.json"), manifest_path)
        except Exception:
            shutil.rmtree(tmp_target, ignore_errors=True)
            return None

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not manifest.get("complete"):
                raise RuntimeError(f"Durable checkpoint manifest is not complete: {durable_uri}")
            files = [dict(item) for item in manifest.get("files", []) if isinstance(item, dict)]
            rels = {str(item.get("path") or "") for item in files}
            has_direct_dcp = (
                "distributed/.metadata" in rels
                and any(path.startswith("distributed/") and path.endswith(".distcp") for path in rels)
            )
            if not has_direct_dcp:
                shutil.rmtree(tmp_target, ignore_errors=True)
                return None

            for item in files:
                rel = str(item.get("path") or "")
                if not rel or rel.startswith("distributed/"):
                    continue
                local_path = tmp_target / rel
                if local_path == manifest_path:
                    continue
                _gcs_download_file(_uri_join(durable_uri, rel), local_path)
                expected_size = int(item.get("bytes", -1))
                if expected_size >= 0 and int(local_path.stat().st_size) != expected_size:
                    raise RuntimeError(
                        "Restored durable checkpoint sidecar failed manifest validation: "
                        f"{rel} expected {expected_size} bytes"
                    )

            checkpoint_num = CheckpointManager.read_ray_train_checkpoint_num(str(tmp_target))
            if checkpoint_num is None:
                value = manifest.get("checkpoint_num")
                checkpoint_num = int(value) if value is not None else None
            if checkpoint_num is None:
                raise RuntimeError(f"Durable checkpoint manifest has no checkpoint_num: {durable_uri}")

            metadata_path = tmp_target / "ditty_checkpoint_metadata.json"
            if not metadata_path.exists():
                metadata_path.write_text(
                    json.dumps(
                        {
                            "checkpoint_num": int(checkpoint_num),
                            "training_state": dict(manifest.get("training_state") or {}),
                            "world_size": int(manifest.get("world_size") or 1),
                            "format": "ditty.checkpoint.v1",
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )

            pointer_path = tmp_target / _DIRECT_DISTRIBUTED_CHECKPOINT_POINTER
            if not pointer_path.exists():
                pointer_path.write_text(
                    json.dumps(
                        {
                            "format": "ditty.distributed_checkpoint_pointer.v1",
                            "distributed_uri": _uri_join(durable_uri, "distributed"),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )

            target = target_root / "checkpoints" / f"checkpoint_{checkpoint_num}"
            shutil.rmtree(target, ignore_errors=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp_target.rename(target)
            return int(checkpoint_num)
        except Exception:
            shutil.rmtree(tmp_target, ignore_errors=True)
            raise

    def report_to_ray_train(
        self,
        *,
        checkpoint_num: int,
        training_state: Dict[str, Any],
        rank: int,
        world_size: int,
    ) -> None:
        if not self.ray_train_reporting_enabled():
            return

        import ray.train as ray_train

        metrics = {
            "ditty_checkpoint_num": int(checkpoint_num),
            "ditty_total_steps": int(training_state.get("total_steps", 0)),
            "ditty_steps": int(training_state.get("steps", 0)),
            "ditty_world_size": int(world_size),
        }
        checkpoint_path = self._get_checkpoint_path(checkpoint_num)
        delete_after_upload = os.environ.get(
            "DITTY_RAY_TRAIN_DELETE_LOCAL_AFTER_UPLOAD", ""
        ).lower() in {"1", "true", "yes", "on"}
        mode = self.ray_train_report_mode(checkpoint_num)
        checkpoint_dir_name = f"ditty_checkpoint_{int(checkpoint_num)}"
        pointer_dir: str | None = None
        if mode == "pointer":
            durable_uri = self.upload_durable_checkpoint(
                checkpoint_num=checkpoint_num,
                training_state=training_state,
                rank=rank,
                world_size=world_size,
            )
            checkpoint_dir_name = f"ditty_ray_checkpoint_{int(checkpoint_num)}"
            if rank == 0:
                pointer_dir = self._write_ray_pointer_checkpoint(
                    checkpoint_num=checkpoint_num,
                    training_state=training_state,
                    world_size=world_size,
                    durable_uri=durable_uri,
                )
                checkpoint = ray_train.Checkpoint.from_directory(pointer_dir)
            else:
                checkpoint = None
        else:
            checkpoint = ray_train.Checkpoint.from_directory(checkpoint_path) if rank == 0 else None
        ray_train.report(
            metrics,
            checkpoint=checkpoint,
            checkpoint_dir_name=checkpoint_dir_name,
            delete_local_checkpoint_after_upload=delete_after_upload if mode == "full" else True,
        )
        if pointer_dir is not None:
            shutil.rmtree(pointer_dir, ignore_errors=True)
        if mode == "pointer" and delete_after_upload:
            shutil.rmtree(checkpoint_path, ignore_errors=True)

    def apply_to_model(self, checkpoint: Checkpoint, model: nn.Module) -> bool:
        """Apply checkpoint model state to a model.

        Returns True if model weights were restored successfully, False if
        checkpoint contents were incompatible and the caller should continue
        with a freshly initialized model.
        """
        if checkpoint.distributed_checkpoint_path is not None:
            from torch.distributed.checkpoint import load as dcp_load
            from torch.distributed.checkpoint.api import CheckpointException
            from torch.distributed.checkpoint.state_dict import (
                get_model_state_dict,
                set_model_state_dict,
            )

            state_dict = {"model": get_model_state_dict(model)}
            try:
                storage_reader = (
                    self._make_gcs_dcp_reader(checkpoint.distributed_checkpoint_path)
                    if _is_gcs_uri(checkpoint.distributed_checkpoint_path)
                    else None
                )
                dcp_load(
                    state_dict,
                    checkpoint_id=checkpoint.distributed_checkpoint_path,
                    storage_reader=storage_reader,
                )
                set_model_state_dict(model, state_dict["model"])
                logger.info("Loaded sharded model weights from distributed checkpoint")
                return True
            except (CheckpointException, RuntimeError) as error:
                if _should_log_rank_zero_only():
                    logger.warning(
                        "Skipping distributed model state restore from %s due to incompatible "
                        "checkpoint contents (%s). Continuing with a freshly initialized model.",
                        checkpoint.distributed_checkpoint_path,
                        _summarize_checkpoint_error(error),
                    )
                return False
        if checkpoint.model_state is not None:
            try:
                model.load_state_dict(checkpoint.model_state)
                logger.info("Loaded model weights from checkpoint")
                return True
            except RuntimeError as error:
                if _should_log_rank_zero_only():
                    logger.warning(
                        "Skipping model state restore from checkpoint due to incompatible contents "
                        "(%s). Continuing with a freshly initialized model.",
                        _summarize_checkpoint_error(error),
                    )
                return False
        return True

    def apply_to_optimizer(
        self,
        checkpoint: Checkpoint,
        optimizer: torch.optim.Optimizer,
        model: Optional[nn.Module] = None,
    ):
        """Apply checkpoint optimizer state to an optimizer."""
        if checkpoint.distributed_checkpoint_path is not None:
            if model is None:
                raise ValueError("model is required to load distributed optimizer state")
            from torch.distributed.checkpoint import load as dcp_load
            from torch.distributed.checkpoint.api import CheckpointException
            from torch.distributed.checkpoint.state_dict import (
                get_optimizer_state_dict,
                set_optimizer_state_dict,
            )

            state_dict = {"optimizer": get_optimizer_state_dict(model, optimizer)}
            try:
                storage_reader = (
                    self._make_gcs_dcp_reader(checkpoint.distributed_checkpoint_path)
                    if _is_gcs_uri(checkpoint.distributed_checkpoint_path)
                    else None
                )
                dcp_load(
                    state_dict,
                    checkpoint_id=checkpoint.distributed_checkpoint_path,
                    storage_reader=storage_reader,
                )
                set_optimizer_state_dict(model, optimizer, state_dict["optimizer"])
                logger.info("Loaded sharded optimizer state from distributed checkpoint")
            except (CheckpointException, RuntimeError) as error:
                if _should_log_rank_zero_only():
                    logger.warning(
                        "Skipping distributed optimizer state restore from %s due to incompatible "
                        "checkpoint contents (%s). Continuing with a freshly initialized optimizer.",
                        checkpoint.distributed_checkpoint_path,
                        _summarize_checkpoint_error(error),
                    )
            return
        if checkpoint.optimizer_state is not None:
            optimizer.load_state_dict(checkpoint.optimizer_state)
            logger.info("Loaded optimizer state from checkpoint")

    def apply_to_scheduler(self, checkpoint: Checkpoint, scheduler: torch.optim.lr_scheduler.LRScheduler):
        """Apply checkpoint scheduler state to a scheduler."""
        if checkpoint.scheduler_state is not None:
            scheduler.load_state_dict(checkpoint.scheduler_state)
            logger.info("Loaded scheduler state from checkpoint")

    def apply_to_preprocessors(self, checkpoint: Checkpoint, preprocessors: List[Any]):
        """Apply checkpointed state to pipeline preprocessors."""
        if not checkpoint.preprocessor_states:
            return

        for row in checkpoint.preprocessor_states:
            if not isinstance(row, dict):
                continue
            index = int(row.get("index", -1))
            if index < 0 or index >= len(preprocessors):
                logger.warning("Skipping preprocessor state with invalid index %s", index)
                continue
            preprocessor = preprocessors[index]
            saved_class = row.get("class")
            current_class = preprocessor.__class__.__name__
            if saved_class and saved_class != current_class:
                logger.warning(
                    "Skipping preprocessor state for index %s: checkpoint class %s != current class %s",
                    index,
                    saved_class,
                    current_class,
                )
                continue
            load_fn = getattr(preprocessor, "load_state_dict", None)
            if not callable(load_fn):
                logger.warning(
                    "Skipping preprocessor state for %s because it has no load_state_dict",
                    current_class,
                )
                continue
            load_fn(row.get("state") or {})
            logger.info("Loaded preprocessor state for %s at index %s", current_class, index)

    def apply_to_scaler(self, checkpoint: Checkpoint, scaler: torch.amp.GradScaler):
        """Apply checkpoint scaler state to a gradient scaler."""
        if checkpoint.scaler_state is not None:
            scaler.load_state_dict(checkpoint.scaler_state)
            logger.info("Loaded scaler state from checkpoint")

    def apply_to_loss_calculator(self, checkpoint: Checkpoint, loss_calculator: Any):
        """Apply checkpoint loss state to a loss calculator."""
        if checkpoint.loss_state is not None:
            from torch.distributed.tensor import DTensor, Replicate

            # Check if loss_calculator params are DTensors
            state_dict = checkpoint.loss_state
            current_state = loss_calculator.state_dict()

            # Convert saved tensors to DTensors if needed
            converted_state = {}
            for k, v in state_dict.items():
                if k in current_state and isinstance(current_state[k], DTensor):
                    # Need to convert saved tensor to DTensor
                    param_dtensor = current_state[k]
                    if isinstance(v, torch.Tensor) and not isinstance(v, DTensor):
                        converted_state[k] = DTensor.from_local(
                            v.to(param_dtensor.device),
                            device_mesh=param_dtensor.device_mesh,
                            placements=[Replicate()] * param_dtensor.device_mesh.ndim,
                            run_check=False,
                        )
                    else:
                        converted_state[k] = v
                else:
                    converted_state[k] = v

            loss_calculator.load_state_dict(converted_state)
            logger.info("Loaded loss calculator state from checkpoint")

    def apply_loss_optimizer_state(
        self,
        checkpoint: Checkpoint,
        optimizer: torch.optim.Optimizer,
        loss_calculator: Any,
        is_fsdp: bool = False,
    ):
        """Apply saved loss optimizer state back into the optimizer."""
        if checkpoint.loss_optimizer_state is None:
            return

        loss_optim_state = checkpoint.loss_optimizer_state
        if not loss_optim_state.get("state"):
            return

        # Get loss_calculator params in order
        loss_params = list(loss_calculator.parameters())
        if len(loss_params) != len(loss_optim_state.get("param_indices", [])):
            logger.warning(
                f"Loss optimizer state mismatch: saved {len(loss_optim_state.get('param_indices', []))} params, "
                f"current {len(loss_params)} params. Skipping."
            )
            return

        # Inject saved state directly into optimizer.state keyed by param object
        saved_indices = loss_optim_state["param_indices"]
        for i, param in enumerate(loss_params):
            saved_idx = saved_indices[i]
            if saved_idx in loss_optim_state["state"]:
                saved_state = loss_optim_state["state"][saved_idx]
                device = param.device

                restored_state = {}
                for k, v in saved_state.items():
                    if isinstance(v, torch.Tensor):
                        from torch.distributed.tensor import DTensor, Replicate
                        if is_fsdp and isinstance(param, DTensor):
                            restored_state[k] = DTensor.from_local(
                                v.to(device),
                                device_mesh=param.device_mesh,
                                placements=[Replicate()] * param.device_mesh.ndim,
                                run_check=False,
                            )
                        else:
                            restored_state[k] = v.to(device)
                    else:
                        restored_state[k] = v

                optimizer.state[param] = restored_state

        logger.info("Loaded loss optimizer state from checkpoint")
