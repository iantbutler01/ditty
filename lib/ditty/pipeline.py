import logging
import os
import types
from logging import getLogger
from typing import Optional, List, Dict, Any, Union, Callable
import bitsandbytes as bnb
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import Dataset, IterableDataset
from .trainer import Trainer
from .data import Data
from .hf_utils import push_to_hub
from .model_factory import ModelFactory, TokenizerFactory
from .loss import LossCalculator, MSELoss
from .processors import PreProcessor, PostProcessor
from .contract import parse_contract, validate_pipeline_chain, format_pipeline_contracts, ContractParseError


logging.basicConfig(level=logging.INFO)

logger = getLogger("ditty_pipeline")


class Pipeline:
    def __init__(
        self,
        model_factory: ModelFactory,
        dataset: Union[Dataset, DataLoader],
        collate_fn: Optional[Callable] = None,
        tokenizer_factory: Optional[TokenizerFactory] = None,
        loss_calculator: LossCalculator = None,  # type: ignore[assignment]
        preprocessors: Optional[List[PreProcessor]] = None,
        postprocessors: Optional[List[PostProcessor]] = None,
        output_dir: str = "./output",
        fp16: bool = True,
        use_bfloat16: bool = False,
        seed: Optional[int] = None,
        batch_size: int = 4,
        grad_accum: int = 1,
        checkpoint_every: int = 1000,
        load_checkpoint: bool = True,
        gradient_checkpointing: bool = True,
        use_8bit_optim: bool = False,
        optim_backend: str = "torchao",  # "torch", "bnb", or "torchao"
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        epochs: int = 1,
        max_steps: Optional[int] = None,
        log_every: int = 10,
        metrics_logger: Optional[Any] = None,
        accelerator_kwargs: Dict[str, Any] = {},
        optimizer: Optional[torch.optim.Optimizer] = None,
        use_compile: bool = False,
        compile_mode: str = "default",
        # Hub options
        push_to_hub: bool = False,
        output_hub_repo: Optional[str] = None,
        hf_hub_token: Optional[str] = None,
        merge_adapters: bool = False,
        private_repo: bool = True,
        # Dataset options
        shuffle_each_epoch: bool = True,
        num_workers: int = 4,
        shuffle_buffer_size: int = 1000,
    ):
        self.model_factory = model_factory
        self._dataset = dataset
        self.collate_fn = collate_fn
        self.tokenizer_factory = tokenizer_factory
        self.loss_calculator = loss_calculator or MSELoss()
        self.preprocessors = preprocessors or []
        self.postprocessors = postprocessors or []
        self.output_dir = output_dir
        self.fp16 = fp16
        self.use_bfloat16 = use_bfloat16
        self.seed = seed
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.checkpoint_every = checkpoint_every
        self.load_checkpoint = load_checkpoint
        self.gradient_checkpointing = gradient_checkpointing
        self.use_8bit_optim = use_8bit_optim
        self.optim_backend = optim_backend
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.max_steps = max_steps
        self.log_every = log_every
        self.metrics_logger = metrics_logger
        self.accelerator_kwargs = accelerator_kwargs
        self.optimizer = optimizer
        self.use_compile = use_compile
        self.compile_mode = compile_mode
        self.push_to_hub = push_to_hub
        self.output_hub_repo = output_hub_repo
        self.hf_hub_token = hf_hub_token or os.environ.get("HF_TOKEN")
        self.merge_adapters = merge_adapters
        self.private_repo = private_repo
        self.shuffle_each_epoch = shuffle_each_epoch
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size

        # Calculate dataset size and create dataloader
        self.dataloader, self.dataset_size, self.total_batches = self._prepare_dataloader()

        if self.push_to_hub and not self.output_hub_repo:
            raise ValueError("Cannot enable push to hub without providing output_hub_repo.")

        self._validate_contracts()

    def _prepare_dataloader(self):
        """
        Prepare dataloader from dataset, handling both HF Dataset and DataLoader inputs.
        For HF Dataset: converts to iterable for streaming, calculates total_batches.
        For DataLoader: uses as-is, tries to get length.
        """
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))

        # If already a DataLoader, use it directly
        if isinstance(self._dataset, DataLoader):
            try:
                dataset_size = len(self._dataset.dataset)
                total_batches = (dataset_size // world_size + self.batch_size - 1) // self.batch_size * self.epochs
            except TypeError:
                dataset_size = None
                total_batches = None
            return self._dataset, dataset_size, total_batches

        # HF Dataset - get size before converting to iterable
        dataset_size = len(self._dataset)
        total_batches = (dataset_size // world_size + self.batch_size - 1) // self.batch_size * self.epochs

        if rank == 0:
            logger.info(f"Dataset: {dataset_size:,} examples, ~{total_batches // self.epochs:,} batches per GPU per epoch")

        # Convert to iterable dataset for streaming
        iterable_dataset = self._dataset.to_iterable_dataset(num_shards=128)
        iterable_dataset = iterable_dataset.shuffle(seed=42, buffer_size=self.shuffle_buffer_size)

        dataloader = DataLoader(
            iterable_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return dataloader, dataset_size, total_batches

    def _validate_contracts(self):
        """Validate pipeline contracts chain together correctly."""
        parse_errors = []

        def strict_parse(component, label):
            if not component.contract:
                return None
            try:
                return parse_contract(component.contract)
            except ContractParseError as e:
                parse_errors.append(f"{label}: {e}")
                return None

        preprocessor_contracts = []
        for p in self.preprocessors:
            contract = strict_parse(p, p.name)
            if contract:
                preprocessor_contracts.append(contract)

        model_contract = strict_parse(self.model_factory, "model")

        postprocessor_contracts = []
        for p in self.postprocessors:
            contract = strict_parse(p, p.name)
            if contract:
                postprocessor_contracts.append(contract)

        loss_contract = strict_parse(self.loss_calculator, "loss_calculator")

        if parse_errors:
            raise ContractParseError(
                "Invalid contracts:\n  " + "\n  ".join(parse_errors)
            )

        # Skip validation if any key contracts are missing
        if not model_contract or not loss_contract:
            logger.debug("Skipping contract validation - model or loss contract not specified")
            return

        # Validate chain
        errors = validate_pipeline_chain(
            preprocessor_contracts,
            model_contract,
            postprocessor_contracts,
            loss_contract,
        )

        if errors:
            # Log full pipeline contracts for debugging
            logger.info(format_pipeline_contracts(
                [(p.name, strict_parse(p, p.name)) for p in self.preprocessors if strict_parse(p, p.name)],
                ("model", model_contract),
                [(p.name, strict_parse(p, p.name)) for p in self.postprocessors if strict_parse(p, p.name)],
                ("loss", loss_contract),
            ))
            raise ContractParseError(
                "Pipeline contract validation errors:\n  " + "\n  ".join(errors)
            )

    def run(self):
        if self.tokenizer_factory:
            self.tokenizer = self.tokenizer_factory.build()

        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size > 1:
            rank = int(os.environ.get("RANK", 0))
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            logger.info(f"Distributed: rank {rank}, local_rank {local_rank}, world_size {world_size}")

        self.model = self.model_factory.build()

        if self.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if hasattr(self.model, "config"):
            self.model.config.use_cache = not self.gradient_checkpointing

        if self.use_compile:
            self.model = torch.compile(self.model, mode=self.compile_mode)

        acc_kwargs = {
            "gradient_accumulation_steps": self.grad_accum,
            "project_dir": self.output_dir,
            "project_config": ProjectConfiguration(
                project_dir=self.output_dir,
                automatic_checkpoint_naming=True,
                save_on_each_node=True,
            ),
            "mixed_precision": "bf16" if self.use_bfloat16 else ("fp16" if self.fp16 else "no"),
        }
        acc_kwargs.update(self.accelerator_kwargs)
        self.accelerator = Accelerator(**acc_kwargs)

        if self.accelerator.is_main_process:
            logger.info(f"Mixed precision: {self.accelerator.mixed_precision}")
            logger.info(f"Model: {self.model.__class__.__name__}")
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"  Total params: {total_params:,}")
            logger.info(f"  Trainable params: {trainable_params:,}")
            logger.info(f"  Loss calculator: {self.loss_calculator.__class__.__name__}")

        if self.optimizer is None:
            lr = self.lr * world_size if world_size > 1 else self.lr
            is_fsdp = self.model_factory.fsdp_config.enabled if self.model_factory.fsdp_config else False

            if self.use_8bit_optim:
                if self.optim_backend == "bnb":
                    if is_fsdp:
                        logger.warning("bitsandbytes 8-bit optimizer not compatible with FSDP2, falling back to torchao")
                        from torchao.optim import AdamW8bit
                        self.optimizer = AdamW8bit(
                            self.model.parameters(),
                            lr=lr,
                            weight_decay=self.weight_decay,
                            betas=(0.9, 0.999),
                            eps=1e-8,
                        )
                    else:
                        self.optimizer = bnb.optim.Adam8bit(
                            self.model.parameters(),
                            lr=lr,
                            weight_decay=self.weight_decay,
                            betas=(0.9, 0.999),
                            eps=1e-8,
                        )
                elif self.optim_backend == "torchao":
                    from torchao.optim import AdamW8bit
                    self.optimizer = AdamW8bit(
                        self.model.parameters(),
                        lr=lr,
                        weight_decay=self.weight_decay,
                        betas=(0.9, 0.999),
                        eps=1e-8,
                    )
                else:
                    raise ValueError(f"Unknown optim_backend: {self.optim_backend}")
            else:
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=self.weight_decay,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                )

        trainer = Trainer(
            model=self.model,  # type: ignore[arg-type]
            optimizer=self.optimizer,
            accelerator=self.accelerator,
            dataset=self.dataloader,
            device="cuda",
            preprocessors=self.preprocessors,
            postprocessors=self.postprocessors,
            loss_calculator=self.loss_calculator,
            grad_accum=self.grad_accum,
            fp16=self.fp16,
            use_bfloat16=self.use_bfloat16,
            output_dir=self.output_dir,
            checkpoint_every=self.checkpoint_every,
            load_checkpoint=self.load_checkpoint,
            seed=self.seed,
            use_scheduler=False,
            metrics_logger=self.metrics_logger,
            log_every=self.log_every,
            max_grad_norm=self.max_grad_norm,
            hf_hub_token=self.hf_hub_token,
            shuffle_each_epoch=self.shuffle_each_epoch,
            total_batches=self.total_batches,
            is_fsdp=self.model_factory.fsdp_config.enabled if self.model_factory.fsdp_config else False,
        )

        trainer.train(epochs=self.epochs, max_steps=self.max_steps)

        self.accelerator.wait_for_everyone()

        if self.push_to_hub:
            model = self.accelerator.unwrap_model(self.model)

            if self.merge_adapters and hasattr(model, "merge_and_unload"):
                logger.info("Merging adapters and unloading.")
                model = model.merge_and_unload(True)

            if self.accelerator.is_main_process:
                logger.info("Pushing to hub!")

            model.push_to_hub = types.MethodType(push_to_hub, model)
            model.push_to_hub(self.output_hub_repo, token=self.hf_hub_token, accelerator=self.accelerator, private=self.private_repo)

        if self.accelerator.is_main_process:
            logger.info("Training complete!")

        return self.model
