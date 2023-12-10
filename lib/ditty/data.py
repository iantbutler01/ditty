from dataclasses import dataclass, field
from typing import Optional
import torch
from torch.utils.data import DataLoader, RandomSampler
import datasets
from transformers.trainer_pt_utils import (
    LabelSmoother,
    LengthGroupedSampler,
)
from transformers.trainer_utils import RemoveColumnsCollator, set_seed
from transformers.data.data_collator import (
    DataCollator,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    default_data_collator,
)
from transformers import PreTrainedTokenizerBase
from typing import Callable

from logging import getLogger

logger = getLogger()


@dataclass(kw_only=True)
class Data:
    dataset: datasets.Dataset | None = None
    split: str = "train"
    tokenizer: PreTrainedTokenizerBase
    seed: Optional[int] = None
    batch_size: int = 8
    grad_accum: int = 1
    length_column_name: Optional[str] = None
    group_by_length: bool = False
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    dataloader_drop_last: bool = False
    load_kwargs: Optional[dict] = None
    collator: Optional[DataCollator] = None
    remove_unused_columns: bool = False

    def __post_init__(self):
        if self.dataset is None and self.load_kwargs is None:
            raise ValueError(
                "dataset and load_kwargs cannot both be None.  Please either pass an instance of Dataset or a dict of args to load the dataset with."
            )

        if self.dataset is None:
            kwargs = self.load_kwargs or {}

            self.dataset = datasets.load_dataset(**kwargs)[self.split]

        if not self.collator:
            collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, return_tensors="pt", mlm=False
            )
            self.collator = collator

    def _get_sampler(self) -> Optional[torch.utils.data.Sampler]:
        generator = torch.Generator()

        if self.seed:
            generator.manual_seed(self.seed)

        # Build the sampler.
        if self.group_by_length:
            lengths = (
                self.dataset[self.length_column_name]
                if self.length_column_name in self.dataset.column_names
                else None
            )
            model_input_name = self.tokenizer.model_input_names[0]
            return LengthGroupedSampler(
                self.batch_size * self.grad_accum,
                dataset=self.dataset,
                lengths=lengths,
                model_input_name=model_input_name,
                generator=generator,
            )
        else:
            return RandomSampler(self.dataset, generator=generator)

    def _get_collator_with_removed_columns(
        self,
        data_collator: Callable,
    ) -> Callable:
        """Wrap the data collator in a callable removing unused columns."""
        if not self.remove_unused_columns:
            return data_collator

        remove_columns_collator = RemoveColumnsCollator(
            data_collator=data_collator,
            signature_columns=None,
            logger=logger,
            description=self.split,
            model_name=self.model.__class__.__name__,
        )
        return remove_columns_collator

    def _remove_unused_columns(self, dataset: "datasets.Dataset"):
        if not self.remove_unused_columns:
            return dataset

        ignored_columns = list(set(dataset.column_names))

        return dataset.remove_columns(ignored_columns)

    def prepare(self, pipeline: list[(str, Callable, dict)]):
        if self.dataset is None:
            raise ValueError("Dataset not set.")

        for op_name, func, kwargs in pipeline:
            op = getattr(self.dataset, op_name)

            if not func:
                self.dataset = op(**kwargs)
            else:
                self.dataset = op(func, **kwargs)

        return self._get_dataloader()

    def _seed_worker(self):
        if not self.seed:
            worker_seed = torch.initial_seed() % 2**32
        else:
            worker_seed = self.seed

        set_seed(worker_seed)

    def _get_dataloader(self) -> DataLoader:
        """
        Returns a [`~torch.utils.data.DataLoader`].

        Will use no sampler if `dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        dataset = self.dataset
        dataset = self._remove_unused_columns(dataset)

        data_collator = self.collator
        data_collator = self._get_collator_with_removed_columns(data_collator)

        if isinstance(dataset, torch.utils.data.IterableDataset):
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                collate_fn=data_collator,
                num_workers=self.dataloader_num_workers,
                pin_memory=self.dataloader_pin_memory,
            )

        sampler = self._get_sampler()

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=data_collator,
            drop_last=self.dataloader_drop_last,
            num_workers=self.dataloader_num_workers,
            pin_memory=self.dataloader_pin_memory,
            worker_init_fn=self._seed_worker,
        )
