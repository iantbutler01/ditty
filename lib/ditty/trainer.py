from dataclasses import dataclass, field
import time
from .utils import convert_seconds_to_string_time
from .loss import LossCalculator, MSELoss, LossOutput
from .processors import PreProcessor, PostProcessor, Context
from .checkpoint import CheckpointManager
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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
    hf_hub_token: Optional[str] = None
    seed: Optional[int] = None
    metrics_logger: Optional[Any] = None
    log_every: int = 10
    max_grad_norm: Optional[float] = None
    shuffle_each_epoch: bool = True
    total_batches: Optional[int] = None
    is_fsdp: bool = False

    # Pre-loaded state (from CheckpointManager, loaded before Trainer creation)
    initial_state: Optional[TrainerState] = None

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

        self.device = self.accelerator.device
        self._manual_device_placement = False

        if self.is_fsdp:
            dataloader_is_pre_sharded = (
                hasattr(self.dataset, "sampler") and isinstance(self.dataset.sampler, DistributedSampler)
            )
            if self.use_scheduler:
                if dataloader_is_pre_sharded:
                    self.optimizer, self.scheduler = self.accelerator.prepare(
                        self.optimizer, self.scheduler
                    )
                else:
                    self.optimizer, self.dataset, self.scheduler = self.accelerator.prepare(
                        self.optimizer, self.dataset, self.scheduler
                    )
            else:
                if dataloader_is_pre_sharded:
                    self.optimizer = self.accelerator.prepare(self.optimizer)
                else:
                    self.optimizer, self.dataset = self.accelerator.prepare(
                        self.optimizer, self.dataset
                    )
            self._manual_device_placement = dataloader_is_pre_sharded
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
        self._checkpoint_iteration = self.checkpoint_manager.get_latest_checkpoint_num() or 0
        if self.initial_state is not None:
            self._checkpoint_iteration += 1

    def _save(self, no_dist=False):
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if self.accelerator.num_processes > 1 and not torch.distributed.is_initialized():
            _dist_debug(
                "skipping checkpoint save because the distributed process group is no longer initialized"
            )
            return

        if self.accelerator.is_main_process:
            logger.info(f"Saving checkpoint at step {self.state.steps} (total: {self.state.total_steps})")
        _dist_debug(
            f"entering _save(checkpoint_num={self._checkpoint_iteration}, "
            f"step={self.state.steps}, total_steps={self.state.total_steps})"
        )
        _dist_debug("waiting for everyone before checkpoint save")
        try:
            self.accelerator.wait_for_everyone()
        except (RuntimeError, ValueError) as error:
            if rank == 0:
                logger.warning(
                    "Skipping checkpoint save because distributed synchronization is unavailable: %s",
                    error,
                )
            return
        _dist_debug("passed wait_for_everyone before checkpoint save")

        self.checkpoint_manager.save(
            checkpoint_num=self._checkpoint_iteration,
            model=self.accelerator.unwrap_model(self.model),
            optimizer=self.optimizer,
            training_state=self.state.state_dict(),
            scheduler=self.scheduler if self.use_scheduler else None,
            scaler=self.accelerator.scaler if hasattr(self.accelerator, 'scaler') and self.accelerator.scaler else None,
            loss_calculator=self.loss_calculator,
            is_fsdp=self.is_fsdp,
            rank=rank,
            local_rank=local_rank,
        )
        _dist_debug(f"checkpoint save completed for checkpoint_num={self._checkpoint_iteration}")
        self._checkpoint_iteration += 1

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

        for ep in range(self.state.epoch, epochs):
            dataset = self.dataset
            _dist_debug(f"starting epoch loop ep={ep}")

            if self.shuffle_each_epoch:
                if hasattr(dataset, 'set_epoch'):
                    dataset.set_epoch(ep)
                elif hasattr(dataset, "sampler") and hasattr(dataset.sampler, "set_epoch"):
                    dataset.sampler.set_epoch(ep)

            for batch in dataset:
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
                }

                for preprocessor in self.preprocessors:
                    result = preprocessor.process(batch, ctx)
                    if result[0] is None:
                        batch = None
                        break
                    batch, ctx = result

                if batch is None:
                    continue

                with self.accelerator.accumulate(self.model):
                    with context_manager:
                        model_output = self.model(batch, **ctx.get("forward_kwargs", {}))
                        if not isinstance(model_output, tuple):
                            model_output = (model_output,)

                        for postprocessor in self.postprocessors:
                            model_output, ctx = postprocessor.process(model_output, ctx)

                        loss_output = self.loss_calculator.compute(model_output, ctx)
                        loss = loss_output.loss

                    self.accelerator.backward(loss)

                    if self.max_grad_norm is not None and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    completed_step = self.state.steps + 1
                    completed_total_steps = self.state.total_steps + 1

                    # Log gradients AFTER clip_grad_norm_ to avoid FSDP2 sync issues
                    # clip_grad_norm_ involves all-reduce, so all ranks must finish it first
                    if (self.metrics_logger is not None and
                        hasattr(self.metrics_logger, 'log_gradients') and
                        hasattr(self.metrics_logger, 'gradient_log_every') and
                        completed_step % self.metrics_logger.gradient_log_every == 0 and
                        self.accelerator.is_main_process):
                        self.metrics_logger.log_gradients(self.model, completed_total_steps)
                    batch_loss = loss.item()
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

                    if completed_step % self.log_every == 0 and self.accelerator.is_main_process:
                        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in loss_output.metrics.items())
                        logger.info(
                            f"Epoch {current_epoch_decimal:.2f} | {batch_info} | "
                            f"{metrics_str} | {progress_info}"
                        )

                        if self.metrics_logger:
                            for k, v in loss_output.metrics.items():
                                self.metrics_logger.log_scalar(f"train/{k}", v, completed_total_steps)
                            # Log learning rate if supported
                            if hasattr(self.metrics_logger, 'log_lr'):
                                self.metrics_logger.log_lr(self.optimizer, completed_total_steps)
                            # Log epoch progress
                            self.metrics_logger.log_scalar("train/epoch", current_epoch_decimal, completed_total_steps)

                    self.state.global_loss += batch_loss

                self.state.steps = completed_step
                self.state.total_steps = completed_total_steps

                if max_steps is not None and self.state.total_steps >= max_steps:
                    break

                if self.state.steps % self.checkpoint_every == 0 and self.state.steps > 0:
                    _dist_debug(f"triggering periodic checkpoint at step={self.state.steps}")
                    self._save()

            _dist_debug(f"epoch {ep} complete, waiting for everyone before epoch increment")
            self.accelerator.wait_for_everyone()
            _dist_debug(f"epoch {ep} post-wait complete")
            self.state.epoch += 1
            self.state.steps = 0

        atexit.unregister(self._save)
        _dist_debug("training loop complete, invoking final _save()")
        self._save()
        _dist_debug("final _save() returned")

        return self.state.global_loss / self.state.total_steps if self.state.total_steps > 0 else 0

    def train(self, epochs=1, max_steps=None):
        if self.accelerator.is_main_process:
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

        return self._train_accelerate(epochs=epochs, max_steps=max_steps)
