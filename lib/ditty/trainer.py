from dataclasses import dataclass, field
import time
from .utils import convert_seconds_to_string_time
from .loss import LossCalculator, MSELoss, LossOutput
from .processors import PreProcessor, PostProcessor, Context
from .checkpoint import CheckpointManager
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import send_to_device, set_seed
from transformers.trainer_pt_utils import get_model_param_count
import atexit
import contextlib
from logging import getLogger
from typing import Optional, Any, List, Union, Callable
import os


def default_scheduler_factory(optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


logger = getLogger("ditty_training")


def _dist_debug_enabled() -> bool:
    value = os.environ.get("DITTY_DEBUG_DIST", "")
    return value.lower() in {"1", "true", "yes", "on"}


def _dist_debug(message: str) -> None:
    if not _dist_debug_enabled():
        return
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    logger.info(f"[dist-debug rank={rank} local_rank={local_rank}] {message}")


def _format_progress_state(
    *,
    total_batches: Optional[int],
    epochs: int,
    run_start_total_steps: int,
    state_total_steps: int,
    time_elapsed: float,
) -> tuple[float, str, str]:
    run_batches_done = max(state_total_steps - run_start_total_steps, 0)
    if total_batches is not None:
        batches_per_epoch = total_batches // epochs if epochs > 0 else total_batches
        current_epoch_decimal = run_batches_done / total_batches if total_batches > 0 else 0.0
        batches_remaining = max(total_batches - run_batches_done, 0)
        estimated_time_remaining = (
            (time_elapsed / run_batches_done) * batches_remaining
            if run_batches_done > 0 else 0
        )
        estimated_time_remaining_ddhhmmss = convert_seconds_to_string_time(
            max(0, estimated_time_remaining)
        )
        percent_done = (run_batches_done / total_batches) * 100 if total_batches > 0 else 0.0
        batch_info = f"Batch {run_batches_done}/{batches_per_epoch}"
        progress_info = f"{percent_done:.2f}% done | ETA: {estimated_time_remaining_ddhhmmss}"
        return current_epoch_decimal, batch_info, progress_info

    current_epoch_decimal = run_batches_done / 1000
    batch_info = f"Batch {run_batches_done}"
    progress_info = f"elapsed: {convert_seconds_to_string_time(time_elapsed)}"
    return current_epoch_decimal, batch_info, progress_info


def _distributed_mean_metrics(metrics: dict[str, float]) -> dict[str, float]:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return metrics

    world_size = torch.distributed.get_world_size()
    if world_size <= 1:
        return metrics

    local_metrics = {
        str(key): float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }
    gathered: list[dict[str, float] | None] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, local_metrics)

    keys = sorted({key for row in gathered if isinstance(row, dict) for key in row})
    averaged: dict[str, float] = {}
    for key in keys:
        averaged[key] = sum(
            float(row.get(key, 0.0)) if isinstance(row, dict) else 0.0
            for row in gathered
        ) / world_size
    averaged["distributed_metric_world_size"] = float(world_size)
    return averaged


def _mean_numeric_metric_dicts(metric_rows: list[dict[str, Any]]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for metrics in metric_rows:
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            name = str(key)
            sums[name] = sums.get(name, 0.0) + float(value)
            counts[name] = counts.get(name, 0) + 1
    return {key: sums[key] / counts[key] for key in sorted(sums) if counts.get(key, 0) > 0}


def _slice_batch_tensor(value: Any, start: int, end: int, batch_size: int) -> Any:
    if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == batch_size:
        return value[start:end]
    return value


def _slice_batch_mapping(mapping: dict[str, Any], start: int, end: int, batch_size: int) -> dict[str, Any]:
    return {
        key: _slice_batch_tensor(value, start, end, batch_size)
        for key, value in mapping.items()
    }


def _next_checkpoint_iteration(
    *,
    local_latest_checkpoint_num: Optional[int],
    initial_checkpoint_num: Optional[int],
    has_initial_state: bool,
) -> int:
    checkpoint_iteration = int(local_latest_checkpoint_num or 0)
    if initial_checkpoint_num is not None:
        checkpoint_iteration = max(checkpoint_iteration, int(initial_checkpoint_num))
    if has_initial_state:
        checkpoint_iteration += 1
    return checkpoint_iteration


@dataclass(kw_only=True)
class TrainerState:
    epoch: int = 0
    steps: int = 0
    total_steps: int = 0
    global_loss: float = 0.0

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "steps": self.steps,
            "total_steps": self.total_steps,
            "global_loss": self.global_loss,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict.get("epoch", 0)
        self.steps = state_dict.get("steps", 0)
        self.total_steps = state_dict.get("total_steps", 0)
        self.global_loss = state_dict.get("global_loss", 0.0)


@dataclass(kw_only=True)
class Trainer:
    """
    Training loop with pipeline pattern:
        batch -> preprocessors -> model.forward -> postprocessors -> loss_calc(pred, target)
    """
    model: nn.Module
    optimizer: torch.optim.Optimizer
    accelerator: Accelerator
    dataset: DataLoader
    device: torch.device

    # Pipeline
    preprocessors: List[PreProcessor] = field(default_factory=list)
    postprocessors: List[PostProcessor] = field(default_factory=list)
    loss_calculator: LossCalculator = None  # type: ignore[assignment]

    # Training config
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    use_scheduler: bool = True
    grad_accum: int = 1
    fp16: bool = False
    use_bfloat16: bool = False
    output_dir: str = "./output"
    checkpoint_every: int = 1000
    save_final_checkpoint: bool = True
    hf_hub_token: Optional[str] = None
    seed: Optional[int] = None
    metrics_logger: Optional[Any] = None
    log_every: int = 10
    validation_callbacks: List[Callable[..., Any]] = field(default_factory=list)
    validation_every: int = 0
    max_grad_norm: Optional[float] = None
    shuffle_each_epoch: bool = True
    total_batches: Optional[int] = None
    is_fsdp: bool = False

    # Pre-loaded state (from CheckpointManager, loaded before Trainer creation)
    initial_state: Optional[TrainerState] = None
    initial_checkpoint_num: Optional[int] = None

    def __post_init__(self):
        if self.seed:
            set_seed(self.seed)

        os.makedirs(self.output_dir, exist_ok=True)

        self.batch_size = self.dataset.batch_size
        self.preprocessors = self.preprocessors or []
        self.postprocessors = self.postprocessors or []
        self.loss_calculator = self.loss_calculator or MSELoss()

        if self.use_scheduler and not self.scheduler:
            self.scheduler = default_scheduler_factory(self.optimizer)

        if self.fp16 and self.use_bfloat16:
            self.f16_dtype = torch.bfloat16
        elif self.fp16:
            self.f16_dtype = torch.float16

        if self.is_fsdp:
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                self.device = torch.device(f"cuda:{local_rank}")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = self.accelerator.device
        self._manual_device_placement = False

        if self.is_fsdp:
            self._manual_device_placement = True
        else:
            if self.use_scheduler:
                (
                    self.model,
                    self.optimizer,
                    self.dataset,
                    self.scheduler,
                ) = self.accelerator.prepare(
                    self.model, self.optimizer, self.dataset, self.scheduler
                )
            else:
                self.model, self.optimizer, self.dataset = self.accelerator.prepare(
                    self.model, self.optimizer, self.dataset
                )

        # Use pre-loaded state if provided, otherwise start fresh
        if self.initial_state is not None:
            self.state = self.initial_state
        else:
            self.state = TrainerState()
        self._run_start_total_steps = self.state.total_steps

        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(self.output_dir)
        self._checkpoint_iteration = _next_checkpoint_iteration(
            local_latest_checkpoint_num=self.checkpoint_manager.get_latest_checkpoint_num(),
            initial_checkpoint_num=self.initial_checkpoint_num,
            has_initial_state=self.initial_state is not None,
        )
        self._last_checkpoint_total_steps: int | None = None

    def _distributed_initialized(self) -> bool:
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    def _rank(self) -> int:
        if self._distributed_initialized():
            return torch.distributed.get_rank()
        return int(os.environ.get("RANK", 0))

    def _world_size(self) -> int:
        if self._distributed_initialized():
            return torch.distributed.get_world_size()
        return int(os.environ.get("WORLD_SIZE", 1))

    def _is_main_process(self) -> bool:
        if self.is_fsdp:
            return self._rank() == 0
        return self.accelerator.is_main_process

    def _num_processes(self) -> int:
        if self.is_fsdp:
            return self._world_size()
        return self.accelerator.num_processes

    def _wait_for_everyone(self) -> None:
        if self.is_fsdp:
            if self._distributed_initialized() and self._world_size() > 1:
                torch.distributed.barrier()
            return
        self.accelerator.wait_for_everyone()

    def _unwrap_model_for_save(self) -> nn.Module:
        if self.is_fsdp:
            return self.model
        return self.accelerator.unwrap_model(self.model)

    def _scaler_for_save(self) -> Optional[torch.amp.GradScaler]:
        if self.is_fsdp:
            return None
        return (
            self.accelerator.scaler
            if hasattr(self.accelerator, "scaler") and self.accelerator.scaler
            else None
        )

    def _accumulate_context(self):
        if self.is_fsdp:
            return contextlib.nullcontext()
        return self.accelerator.accumulate(self.model)

    def _backward(self, loss: torch.Tensor) -> None:
        if self.is_fsdp:
            loss.backward()
        else:
            self.accelerator.backward(loss)

    def _loss_for_backward(self, loss: torch.Tensor) -> torch.Tensor:
        if self.is_fsdp and self.grad_accum > 1:
            return loss / self.grad_accum
        return loss

    def _should_step_optimizer(
        self,
        completed_step: int,
        *,
        end_of_epoch: bool = False,
        stopping: bool = False,
    ) -> bool:
        if not self.is_fsdp:
            return True
        return completed_step % max(int(self.grad_accum), 1) == 0 or end_of_epoch or stopping

    def _clip_grad_norm_if_needed(self, *, should_step_optimizer: bool) -> None:
        if self.max_grad_norm is None or self.max_grad_norm <= 0:
            return
        if self.is_fsdp:
            if should_step_optimizer:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            return
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

    def _save(self, no_dist=False):
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        checkpoint_num = self._checkpoint_iteration
        training_state = self.state.state_dict()

        if self._num_processes() > 1 and not torch.distributed.is_initialized():
            _dist_debug(
                "skipping checkpoint save because the distributed process group is no longer initialized"
            )
            return None

        if self._is_main_process():
            logger.info(f"Saving checkpoint at step {self.state.steps} (total: {self.state.total_steps})")
        _dist_debug(
            f"entering _save(checkpoint_num={self._checkpoint_iteration}, "
            f"step={self.state.steps}, total_steps={self.state.total_steps})"
        )
        _dist_debug("waiting for everyone before checkpoint save")
        try:
            self._wait_for_everyone()
        except (RuntimeError, ValueError) as error:
            if rank == 0:
                logger.warning(
                    "Skipping checkpoint save because distributed synchronization is unavailable: %s",
                    error,
                )
            return None
        _dist_debug("passed wait_for_everyone before checkpoint save")

        self.checkpoint_manager.save(
            checkpoint_num=checkpoint_num,
            model=self._unwrap_model_for_save(),
            optimizer=self.optimizer,
            training_state=training_state,
            scheduler=self.scheduler if self.use_scheduler else None,
            scaler=self._scaler_for_save(),
            loss_calculator=self.loss_calculator,
            preprocessors=self.preprocessors,
            is_fsdp=self.is_fsdp,
            rank=rank,
            local_rank=local_rank,
        )
        if self.checkpoint_manager.ray_train_reporting_enabled():
            if rank == 0:
                self.checkpoint_manager.write_ray_train_metadata(
                    checkpoint_num=checkpoint_num,
                    training_state=training_state,
                    world_size=world_size,
                )
            try:
                self._wait_for_everyone()
            except (RuntimeError, ValueError) as error:
                if rank == 0:
                    logger.warning(
                        "Skipping Ray Train checkpoint report because post-save synchronization failed: %s",
                        error,
                    )
                self._checkpoint_iteration += 1
                return checkpoint_num

            self.checkpoint_manager.report_to_ray_train(
                checkpoint_num=checkpoint_num,
                training_state=training_state,
                rank=rank,
                world_size=world_size,
            )
        if rank == 0:
            self.checkpoint_manager.prune_old_checkpoints()
        try:
            self._wait_for_everyone()
        except (RuntimeError, ValueError) as error:
            if rank == 0:
                logger.warning(
                    "Checkpoint save completed but post-prune synchronization failed: %s",
                    error,
                )
        _dist_debug(f"checkpoint save completed for checkpoint_num={checkpoint_num}")
        self._last_checkpoint_total_steps = self.state.total_steps
        self._checkpoint_iteration += 1
        return checkpoint_num

    def _save_and_notify(self, ctx: Optional[dict[str, Any]] = None, *, reason: str = "periodic"):
        checkpoint_num = self._save()
        if checkpoint_num is None:
            return None
        checkpoint_path = self.checkpoint_manager.get_checkpoint_path(checkpoint_num)
        payload = {
            "checkpoint_num": int(checkpoint_num),
            "checkpoint_path": checkpoint_path,
            "reason": str(reason),
            "training_state": self.state.state_dict(),
            "output_dir": self.output_dir,
        }
        for component in [*self.preprocessors, *self.postprocessors, self.loss_calculator]:
            hook = getattr(component, "on_checkpoint_saved", None)
            if not callable(hook):
                continue
            try:
                hook(payload, ctx or {})
            except BaseException as error:
                if self._is_main_process():
                    logger.warning(
                        "Checkpoint hook failed for %s after checkpoint_%s: %s",
                        component.__class__.__name__,
                        checkpoint_num,
                        error,
                    )
        return checkpoint_num

    def _should_skip_final_checkpoint(self) -> bool:
        return (
            self._last_checkpoint_total_steps is not None
            and self._last_checkpoint_total_steps == self.state.total_steps
        )

    def _shutdown_pipeline_components(self) -> None:
        for component in [*self.preprocessors, *self.postprocessors, self.loss_calculator]:
            shutdown = getattr(component, "shutdown", None)
            if not callable(shutdown):
                continue
            try:
                shutdown()
            except BaseException as error:
                if self._is_main_process():
                    logger.warning(
                        "Pipeline component shutdown failed for %s: %s",
                        component.__class__.__name__,
                        error,
                    )

    def _log_pipeline(self):
        logger.info("Pipeline:")
        logger.info(f"  preprocessors:")
        for p in self.preprocessors:
            logger.info(f"    - {p}")
        logger.info(f"  model: {self.model.__class__.__name__} ({get_model_param_count(self.model, trainable_only=True):,} params)")
        logger.info(f"  postprocessors:")
        for p in self.postprocessors:
            logger.info(f"    - {p}")
        logger.info(f"  loss: {self.loss_calculator.__class__.__name__}")

    def _run_validation_callbacks(self, step: int) -> bool:
        if not self.validation_callbacks or self.validation_every <= 0:
            return False
        if step % self.validation_every != 0:
            return False

        stop_training = False
        for callback in self.validation_callbacks:
            result = callback(
                model=self.model,
                accelerator=self.accelerator,
                state=self.state,
                step=step,
                output_dir=self.output_dir,
                metrics_logger=self.metrics_logger,
            )
            if result is None:
                continue
            if not isinstance(result, dict):
                raise TypeError(
                    "Validation callbacks must return a dict of metrics, a dict with "
                    "stop_training, or None."
                )
            stop_training = stop_training or bool(result.get("stop_training", False))
            if self.metrics_logger is not None and self._is_main_process():
                for key, value in result.items():
                    if key == "stop_training":
                        continue
                    if isinstance(value, (int, float)):
                        self.metrics_logger.log_scalar(f"validation/{key}", float(value), step)

        if self._num_processes() > 1 and torch.distributed.is_initialized():
            stop_tensor = torch.tensor(
                1 if stop_training else 0,
                device=self.device,
                dtype=torch.int,
            )
            torch.distributed.all_reduce(stop_tensor, op=torch.distributed.ReduceOp.MAX)
            stop_training = bool(stop_tensor.item())

        return stop_training

    def _train_accelerate(self, epochs=1, max_steps=None):
        context_manager = contextlib.nullcontext()
        if self.fp16:
            context_manager = torch.autocast(device_type=self.device.type, dtype=self.f16_dtype)

        self.model.train()
        if self.total_batches is not None:
            total_batches = self.total_batches
        else:
            try:
                total_batches = len(self.dataset) * epochs
            except TypeError:
                total_batches = None
        start_time = time.time()

        atexit.register(self._save)

        stop_requested = False
        for ep in range(self.state.epoch, epochs):
            dataset = self.dataset
            try:
                batches_in_epoch = len(dataset)
            except TypeError:
                batches_in_epoch = None
            _dist_debug(f"starting epoch loop ep={ep}")

            if self.shuffle_each_epoch:
                if hasattr(dataset, 'set_epoch'):
                    dataset.set_epoch(ep)
                elif hasattr(dataset, "sampler") and hasattr(dataset.sampler, "set_epoch"):
                    dataset.sampler.set_epoch(ep)

            for batch_idx, batch in enumerate(dataset):
                if batch is None:
                    break

                if self._manual_device_placement:
                    batch = send_to_device(batch, self.device, non_blocking=True)

                original_batch = batch
                ctx: Context = {
                    "epoch": ep,
                    "step": self.state.steps,
                    "total_steps": self.state.total_steps,
                    "device": self.device,
                    "original_batch": original_batch,
                    "model": self.model,
                    "output_dir": self.output_dir,
                    "policy_checkpoint_id": f"{os.path.abspath(self.output_dir)}:step-{self.state.total_steps}",
                }

                for preprocessor in self.preprocessors:
                    result = preprocessor.process(batch, ctx)
                    if result[0] is None:
                        batch = None
                        break
                    batch, ctx = result

                if batch is None:
                    continue

                with self._accumulate_context():
                    # Micro-batching support: when ctx supplies `loss_micro_batch_size`,
                    # split the batch (and per-sequence ctx tensors) along dim 0 into
                    # equal-ish chunks, forward+backward each chunk separately, then
                    # let the outer step's clip + optimizer.step apply once. The loss
                    # contributions are scaled by 1/num_chunks so the accumulated
                    # gradient equals the full-batch gradient for sum-style losses
                    # (DR.GRPO/DR.GSPO) when chunks are equal size; small bias for
                    # unequal final chunk is negligible at micro-batch >> 1.
                    micro_bs = int(ctx.get("loss_micro_batch_size") or 0)
                    batch_size_total = int(batch.shape[0]) if hasattr(batch, "shape") else None
                    micro_batched = (
                        micro_bs > 0
                        and batch_size_total is not None
                        and batch_size_total > 0
                    )
                    if micro_batched:
                        num_chunks = max((batch_size_total + micro_bs - 1) // micro_bs, 1)
                        local_chunks = num_chunks
                        # Synchronize chunk count across ranks so all ranks issue the
                        # same number of forward+backward passes (each backward triggers
                        # FSDP's gradient all-reduce). If ranks disagreed on num_chunks
                        # we'd deadlock at the next collective.
                        if torch.distributed.is_available() and torch.distributed.is_initialized():
                            t = torch.tensor([num_chunks], device=self.device, dtype=torch.int64)
                            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
                            num_chunks = int(t.item())
                        # Recompute micro_bs given the synchronized num_chunks.
                        effective_micro_bs = (batch_size_total + num_chunks - 1) // num_chunks if num_chunks > 0 else batch_size_total
                        _dist_debug(
                            "loss microbatch start "
                            f"batch_size={batch_size_total} micro_bs={micro_bs} "
                            f"local_chunks={local_chunks} synced_chunks={num_chunks} "
                            f"effective_micro_bs={effective_micro_bs}"
                        )

                        full_forward_kwargs = dict(ctx.get("forward_kwargs", {}))
                        total_loss_scalar = 0.0
                        last_loss_output = None
                        micro_metric_rows: list[dict[str, Any]] = []
                        for chunk_idx in range(num_chunks):
                            s = chunk_idx * effective_micro_bs
                            e = min(s + effective_micro_bs, batch_size_total)
                            has_real_rows = s < e
                            if s >= e:
                                # No real records on this rank for this chunk slot:
                                # repeat the last row to keep collective ops aligned;
                                # the loss contribution is explicitly zeroed below.
                                s, e = batch_size_total - 1, batch_size_total
                            chunk_batch = batch[s:e]
                            chunk_fwd_kwargs = _slice_batch_mapping(
                                full_forward_kwargs,
                                s,
                                e,
                                batch_size_total,
                            )
                            chunk_ctx = _slice_batch_mapping(ctx, s, e, batch_size_total)
                            chunk_ctx["forward_kwargs"] = chunk_fwd_kwargs
                            if "mask" in chunk_ctx and isinstance(chunk_ctx["mask"], torch.Tensor):
                                if not has_real_rows:
                                    chunk_ctx["mask"] = torch.zeros_like(chunk_ctx["mask"])
                            if "advantages" in chunk_ctx and isinstance(chunk_ctx["advantages"], torch.Tensor):
                                if not has_real_rows:
                                    chunk_ctx["advantages"] = torch.zeros_like(chunk_ctx["advantages"])
                            with context_manager:
                                model_output = self.model(chunk_batch, **chunk_fwd_kwargs)
                                if not isinstance(model_output, tuple):
                                    model_output = (model_output,)
                                for postprocessor in self.postprocessors:
                                    model_output, chunk_ctx = postprocessor.process(model_output, chunk_ctx)
                                loss_output = self.loss_calculator.compute(model_output, chunk_ctx)
                                chunk_loss = loss_output.loss / max(num_chunks, 1)
                            if has_real_rows:
                                micro_metric_rows.append(dict(loss_output.metrics))
                            self._backward(self._loss_for_backward(chunk_loss))
                            total_loss_scalar += float(chunk_loss.item()) * num_chunks
                            last_loss_output = loss_output
                        _dist_debug("loss microbatch backward complete")
                        loss = torch.tensor(total_loss_scalar / max(num_chunks, 1), device=self.device)
                        micro_metrics = _mean_numeric_metric_dicts(micro_metric_rows)
                        micro_metrics["loss_microbatch_real_chunks"] = float(len(micro_metric_rows))
                        micro_metrics["loss_microbatch_synced_chunks"] = float(num_chunks)
                        micro_metrics["loss_microbatch_effective_micro_bs"] = float(effective_micro_bs)
                        loss_output = LossOutput(
                            loss=loss,
                            metrics=micro_metrics if micro_metrics else (
                                dict(last_loss_output.metrics) if last_loss_output is not None else {}
                            ),
                        )
                    else:
                        with context_manager:
                            model_output = self.model(batch, **ctx.get("forward_kwargs", {}))
                            if not isinstance(model_output, tuple):
                                model_output = (model_output,)

                            for postprocessor in self.postprocessors:
                                model_output, ctx = postprocessor.process(model_output, ctx)

                            loss_output = self.loss_calculator.compute(model_output, ctx)
                            loss = loss_output.loss

                        self._backward(self._loss_for_backward(loss))

                    completed_step = self.state.steps + 1
                    completed_total_steps = self.state.total_steps + 1
                    end_of_epoch = (
                        batches_in_epoch is not None and batch_idx + 1 >= batches_in_epoch
                    )
                    stopping_for_max = max_steps is not None and completed_total_steps >= max_steps
                    should_step_optimizer = self._should_step_optimizer(
                        completed_step,
                        end_of_epoch=end_of_epoch,
                        stopping=stopping_for_max,
                    )

                    self._clip_grad_norm_if_needed(should_step_optimizer=should_step_optimizer)

                    # Log gradients AFTER clip_grad_norm_ to avoid FSDP2 sync issues
                    # clip_grad_norm_ involves all-reduce, so all ranks must finish it first
                    if (self.metrics_logger is not None and
                        hasattr(self.metrics_logger, 'log_gradients') and
                        hasattr(self.metrics_logger, 'gradient_log_every') and
                        completed_step % self.metrics_logger.gradient_log_every == 0 and
                        self._is_main_process()):
                        self.metrics_logger.log_gradients(self.model, completed_total_steps)
                    batch_loss = loss.item()
                    if should_step_optimizer:
                        self.optimizer.step()
                        if self.use_scheduler and self.scheduler:
                            self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)

                    time_elapsed = time.time() - start_time
                    current_epoch_decimal, batch_info, progress_info = _format_progress_state(
                        total_batches=total_batches,
                        epochs=epochs,
                        run_start_total_steps=self._run_start_total_steps,
                        state_total_steps=completed_total_steps,
                        time_elapsed=time_elapsed,
                    )

                    log_metrics = loss_output.metrics
                    if completed_step % self.log_every == 0:
                        log_metrics = _distributed_mean_metrics(loss_output.metrics)

                    if completed_step % self.log_every == 0 and self._is_main_process():
                        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in log_metrics.items())
                        logger.info(
                            f"Epoch {current_epoch_decimal:.2f} | {batch_info} | "
                            f"{metrics_str} | {progress_info}"
                        )

                        if self.metrics_logger:
                            for k, v in log_metrics.items():
                                self.metrics_logger.log_scalar(f"train/{k}", v, completed_total_steps)
                            # Log learning rate if supported
                            if hasattr(self.metrics_logger, 'log_lr'):
                                self.metrics_logger.log_lr(self.optimizer, completed_total_steps)
                            # Log epoch progress
                            self.metrics_logger.log_scalar("train/epoch", current_epoch_decimal, completed_total_steps)

                    self.state.global_loss += batch_loss

                self.state.steps = completed_step
                self.state.total_steps = completed_total_steps

                if self._run_validation_callbacks(completed_total_steps):
                    stop_requested = True
                    break

                forced_checkpoint = bool(ctx.get("force_checkpoint_after_step"))
                saved_checkpoint = False
                if forced_checkpoint:
                    _dist_debug(
                        "triggering forced checkpoint at "
                        f"step={self.state.steps} reason={ctx.get('checkpoint_reason', 'requested')}"
                    )
                    saved_checkpoint = self._save_and_notify(
                        ctx,
                        reason=str(ctx.get("checkpoint_reason") or "requested"),
                    ) is not None

                if (
                    self.checkpoint_every > 0
                    and self.state.steps % self.checkpoint_every == 0
                    and not saved_checkpoint
                ):
                    _dist_debug(f"triggering periodic checkpoint at step={self.state.steps}")
                    self._save_and_notify(ctx, reason="periodic")

                if max_steps is not None and self.state.total_steps >= max_steps:
                    stop_requested = True
                    break

            if stop_requested:
                _dist_debug("max_steps reached, leaving epoch loop without resetting step state")
                break
            _dist_debug(f"epoch {ep} complete, waiting for everyone before epoch increment")
            self._wait_for_everyone()
            _dist_debug(f"epoch {ep} post-wait complete")
            self.state.epoch += 1
            self.state.steps = 0

        atexit.unregister(self._save)
        if self.save_final_checkpoint:
            if self._should_skip_final_checkpoint():
                _dist_debug("training loop complete, skipping duplicate final checkpoint")
                if self._is_main_process():
                    logger.info(
                        "Skipping final checkpoint at step %s (total: %s); latest checkpoint already covers this state",
                        self.state.steps,
                        self.state.total_steps,
                    )
                self._wait_for_everyone()
            else:
                _dist_debug("training loop complete, invoking final _save()")
                self._save_and_notify(reason="final")
                _dist_debug("final _save() returned")
        else:
            _dist_debug("training loop complete, final checkpoint disabled")
            self._wait_for_everyone()

        return self.state.global_loss / self.state.total_steps if self.state.total_steps > 0 else 0

    def train(self, epochs=1, max_steps=None):
        if self._is_main_process():
            logger.info("***** Running training *****")
            try:
                logger.info(f"  Num examples = {len(self.dataset):,}")
            except TypeError:
                logger.info("  Num examples = unknown (iterable dataset)")
            logger.info(f"  Num Epochs = {epochs:,}")
            if max_steps:
                logger.info(f"  Total optimization steps = {max_steps:,}")
            logger.info(f"  Instantaneous batch size per device = {self.batch_size:,}")
            logger.info(f"  Gradient Accumulation steps = {self.grad_accum}")
            logger.info(
                f"  Number of trainable parameters = {get_model_param_count(self.model, trainable_only=True):,}"
            )
            logger.info(f"  Loss calculator = {self.loss_calculator.__class__.__name__}")

        try:
            return self._train_accelerate(epochs=epochs, max_steps=max_steps)
        finally:
            self._shutdown_pipeline_components()
