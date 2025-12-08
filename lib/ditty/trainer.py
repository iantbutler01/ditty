from dataclasses import dataclass, field
import time
from .utils import convert_seconds_to_string_time
from .loss import LossCalculator, MSELoss, LossOutput
from .processors import PreProcessor, PostProcessor, Context
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers.trainer_pt_utils import get_model_param_count
import atexit
import contextlib
from logging import getLogger
from typing import Optional, Any, List, Union, Callable
import os


def default_scheduler_factory(optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


def get_number_from_checkpoint(filename: str) -> Optional[int]:
    parts = filename.split("_")
    if len(parts) != 2 or not parts[1].isdigit():
        return None
    return int(parts[1])


logger = getLogger("ditty_training")


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
        self.epoch = state_dict["epoch"]
        self.steps = state_dict["steps"]
        self.total_steps = state_dict["total_steps"]
        self.global_loss = state_dict["global_loss"]


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
    load_checkpoint: bool = False
    hf_hub_token: Optional[str] = None
    seed: Optional[int] = None
    metrics_logger: Optional[Any] = None
    log_every: int = 10
    max_grad_norm: Optional[float] = None
    shuffle_each_epoch: bool = True
    total_batches: Optional[int] = None
    is_fsdp: bool = False

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

        if self.is_fsdp:
            if self.use_scheduler:
                self.optimizer, self.dataset, self.scheduler = self.accelerator.prepare(
                    self.optimizer, self.dataset, self.scheduler
                )
                self.accelerator.register_for_checkpointing(self.scheduler)
            else:
                self.optimizer, self.dataset = self.accelerator.prepare(
                    self.optimizer, self.dataset
                )
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
                self.accelerator.register_for_checkpointing(self.scheduler)
            else:
                self.model, self.optimizer, self.dataset = self.accelerator.prepare(
                    self.model, self.optimizer, self.dataset
                )

        self.state = TrainerState()
        self.accelerator.register_for_checkpointing(self.state)

    def _save_dist(self):
        if self.accelerator.is_main_process:
            logger.info("Saving full model distribution.")

        is_fsdp = str(self.accelerator.distributed_type) == "FSDP"

        if not is_fsdp:
            model = self.accelerator.unwrap_model(self.model)
            model_state = model.state_dict()
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(f"{self.output_dir}/dist", state_dict=model_state, token=self.hf_hub_token)
            else:
                os.makedirs(f"{self.output_dir}/dist", exist_ok=True)
                torch.save(model_state, f"{self.output_dir}/dist/model.pt")
        else:
            model = self.accelerator.unwrap_model(self.model)
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(
                    f"{self.output_dir}/dist",
                    is_main_process=self.accelerator.is_main_process,
                    save_function=self.accelerator.save,
                    state_dict=self.accelerator.get_state_dict(self.model),
                    token=self.hf_hub_token
                )
            else:
                state_dict = self.accelerator.get_state_dict(self.model)
                if self.accelerator.is_main_process:
                    os.makedirs(f"{self.output_dir}/dist", exist_ok=True)
                    torch.save(state_dict, f"{self.output_dir}/dist/model.pt")

    def _save(self, no_dist=False):
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state()
        if not no_dist:
            self._save_dist()

    def _load_last_checkpoint(self):
        try:
            checkpoints_dir = f"{self.output_dir}/checkpoints/"
            if os.path.exists(checkpoints_dir) and os.listdir(checkpoints_dir):
                files = os.listdir(checkpoints_dir)
                checkpoint_files = [
                    f for f in files
                    if f.startswith("checkpoint_") and get_number_from_checkpoint(f) is not None
                ]
                checkpoint_files_sorted = sorted(checkpoint_files, key=lambda f: get_number_from_checkpoint(f) or 0)

                if checkpoint_files_sorted:
                    last_cp = checkpoint_files_sorted[-1]
                    if self.accelerator.is_main_process:
                        logger.info(f"Loading checkpoint: {last_cp}")
                    self.accelerator.load_state(f"{self.output_dir}/checkpoints/{last_cp}")
                    last_cp_num = last_cp.split("_")[-1]
                    self.accelerator.project_configuration.iteration = int(last_cp_num) + 1
                    return last_cp

        except FileNotFoundError as e:
            logger.warning(e)
        return None

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
        # Use passed total_batches, or try to calculate from dataset length
        if self.total_batches is not None:
            total_batches = self.total_batches
        else:
            try:
                total_batches = len(self.dataset) * epochs
            except TypeError:
                total_batches = None
        start_time = time.time()

        if self.load_checkpoint:
            last_cp = self._load_last_checkpoint()
            if self.accelerator.is_main_process:
                if last_cp:
                    logger.info(f"Resuming training state from: {last_cp}.")
                else:
                    logger.info("No training state checkpoint found, starting fresh (model weights loaded separately).")

        atexit.register(self._save)

        for ep in range(self.state.epoch, epochs):
            dataset = self.dataset

            if self.shuffle_each_epoch and hasattr(dataset, 'set_epoch'):
                dataset.set_epoch(ep)

            if self.state.steps > 0:
                if self.accelerator.is_main_process:
                    logger.info(f"Resuming from batch {self.state.steps}, skipping {self.state.steps} batches.")
                dataset = self.accelerator.skip_first_batches(self.dataset, self.state.steps)

            for batch in dataset:
                if batch is None:
                    break

                original_batch = batch
                ctx: Context = {
                    "epoch": ep,
                    "step": self.state.steps,
                    "total_steps": self.state.total_steps,
                    "device": self.device,
                    "original_batch": original_batch,
                }

                # Preprocessors: batch -> batch_transformed
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

                        # Postprocessors: transform model output tuple
                        for postprocessor in self.postprocessors:
                            model_output, ctx = postprocessor.process(model_output, ctx)

                        loss_output = self.loss_calculator.compute(model_output, ctx)
                        loss = loss_output.loss

                    self.accelerator.backward(loss)
                    if self.max_grad_norm is not None and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    batch_loss = loss.item()
                    self.optimizer.step()
                    if self.use_scheduler and self.scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    # Logging
                    time_elapsed = time.time() - start_time
                    if total_batches is not None:
                        batches_per_epoch = total_batches // epochs if epochs > 0 else total_batches
                        total_batches_done = ep * batches_per_epoch + self.state.steps
                        current_epoch_decimal = total_batches_done / total_batches if total_batches > 0 else 0
                        batches_remaining = total_batches - total_batches_done
                        estimated_time_remaining = (
                            (time_elapsed / total_batches_done) * batches_remaining
                            if total_batches_done > 0 else 0
                        )
                        estimated_time_remaining_ddhhmmss = convert_seconds_to_string_time(
                            estimated_time_remaining
                        )
                        percent_done = (total_batches_done / total_batches) * 100 if total_batches > 0 else 0
                        batch_info = f"Batch {self.state.steps}/{batches_per_epoch}"
                        progress_info = f"{percent_done:.2f}% done | ETA: {estimated_time_remaining_ddhhmmss}"
                    else:
                        current_epoch_decimal = ep + (self.state.steps / 1000)  # rough estimate
                        batch_info = f"Batch {self.state.steps}"
                        progress_info = f"elapsed: {convert_seconds_to_string_time(time_elapsed)}"

                    if self.state.steps % self.log_every == 0 and self.accelerator.is_main_process:
                        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in loss_output.metrics.items())
                        logger.info(
                            f"Epoch {current_epoch_decimal:.2f} | {batch_info} | "
                            f"{metrics_str} | {progress_info}"
                        )

                        # Log to external logger if provided
                        if self.metrics_logger:
                            for k, v in loss_output.metrics.items():
                                self.metrics_logger.log_scalar(f"train/{k}", v, self.state.total_steps)

                    self.state.global_loss += batch_loss

                self.state.steps += 1
                self.state.total_steps += 1

                if max_steps is not None and self.state.total_steps >= max_steps:
                    break

                if self.state.steps % self.checkpoint_every == 0 and self.state.steps > 0:
                    self._save()

            self.accelerator.wait_for_everyone()
            self.state.epoch += 1
            self.state.steps = 0

        atexit.unregister(self._save)
        self._save()

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