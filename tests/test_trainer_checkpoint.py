import inspect
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from ditty.pipeline import Pipeline
from ditty.trainer import Trainer, _mean_numeric_metric_dicts, _next_checkpoint_iteration


class TrainerCheckpointTests(unittest.TestCase):
    def test_final_checkpoint_skip_only_matches_current_total_steps(self):
        trainer = object.__new__(Trainer)

        trainer._last_checkpoint_total_steps = None
        trainer.state = type("State", (), {"total_steps": 10})()
        self.assertFalse(trainer._should_skip_final_checkpoint())

        trainer._last_checkpoint_total_steps = 9
        self.assertFalse(trainer._should_skip_final_checkpoint())

        trainer._last_checkpoint_total_steps = 10
        self.assertTrue(trainer._should_skip_final_checkpoint())

    def test_shutdown_pipeline_components_calls_shutdown_hooks(self):
        calls = []

        class Component:
            def __init__(self, name):
                self.name = name

            def shutdown(self):
                calls.append(self.name)

        trainer = object.__new__(Trainer)
        trainer.preprocessors = [Component("pre")]
        trainer.postprocessors = [object(), Component("post")]
        trainer.loss_calculator = Component("loss")
        trainer.accelerator = type("Accelerator", (), {"is_main_process": True})()

        trainer._shutdown_pipeline_components()

        self.assertEqual(calls, ["pre", "post", "loss"])

    def test_resume_checkpoint_iteration_uses_loaded_checkpoint_num(self):
        self.assertEqual(
            _next_checkpoint_iteration(
                local_latest_checkpoint_num=None,
                initial_checkpoint_num=2,
                has_initial_state=True,
            ),
            3,
        )
        self.assertEqual(
            _next_checkpoint_iteration(
                local_latest_checkpoint_num=2,
                initial_checkpoint_num=2,
                has_initial_state=True,
            ),
            3,
        )

    def test_loss_microbatch_path_synchronizes_small_local_batches(self):
        source = inspect.getsource(Trainer._train_accelerate)

        self.assertIn("and batch_size_total > 0", source)
        self.assertIn("has_real_rows = s < e", source)
        self.assertIn("torch.zeros_like(chunk_ctx[\"mask\"])", source)
        self.assertIn("torch.zeros_like(chunk_ctx[\"advantages\"])", source)
        self.assertIn("micro_metric_rows", source)

    def test_mean_numeric_metric_dicts_averages_present_numeric_values(self):
        metrics = _mean_numeric_metric_dicts(
            [
                {"ratio": 1.0, "loss": -0.5},
                {"ratio": 3.0, "loss": -1.5, "text": "ignored"},
                {"loss": -2.5},
            ]
        )

        self.assertEqual(metrics["ratio"], 2.0)
        self.assertEqual(metrics["loss"], -1.5)
        self.assertNotIn("text", metrics)

    def test_fsdp_trainer_does_not_prepare_optimizer_with_accelerate(self):
        class AcceleratorStub:
            device = torch.device("cpu")
            is_main_process = True
            num_processes = 1

            def prepare(self, *args, **kwargs):
                raise AssertionError("FSDP2 trainer must not call accelerator.prepare")

            def backward(self, *args, **kwargs):
                raise AssertionError("FSDP2 trainer must not call accelerator.backward")

            def clip_grad_norm_(self, *args, **kwargs):
                raise AssertionError("FSDP2 trainer must not call accelerator.clip_grad_norm_")

        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        dataloader = DataLoader(TensorDataset(torch.ones(2, 2)), batch_size=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                accelerator=AcceleratorStub(),
                dataset=dataloader,
                device=torch.device("cpu"),
                output_dir=tmpdir,
                use_scheduler=False,
                is_fsdp=True,
            )

            loss = trainer.model(torch.ones(1, 2)).sum()
            trainer._backward(loss)

        self.assertTrue(trainer._manual_device_placement)
        self.assertIs(trainer.optimizer, optimizer)
        self.assertIsNotNone(model.weight.grad)

    def test_fsdp_pipeline_shards_map_style_dataloader_inputs(self):
        pipeline = object.__new__(Pipeline)
        pipeline.model_factory = SimpleNamespace(fsdp_config=SimpleNamespace(enabled=True))
        pipeline._dataset = DataLoader(TensorDataset(torch.arange(10)), batch_size=2)
        pipeline.seed = 123
        pipeline.batch_size = 2
        pipeline.epochs = 1
        pipeline.shuffle_each_epoch = False
        pipeline.collate_fn = None
        pipeline.num_workers = 0

        with mock.patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "1"}):
            dataloader, dataset_size, total_batches = pipeline._prepare_dataloader()

        self.assertEqual(dataset_size, 10)
        self.assertEqual(total_batches, 3)
        self.assertIsInstance(dataloader.sampler, DistributedSampler)
        self.assertEqual(dataloader.sampler.num_replicas, 2)
        self.assertEqual(dataloader.sampler.rank, 1)

    def test_fsdp_grad_accum_steps_on_boundaries_and_final_partial(self):
        trainer = object.__new__(Trainer)
        trainer.is_fsdp = True
        trainer.grad_accum = 4

        self.assertFalse(trainer._should_step_optimizer(1))
        self.assertTrue(trainer._should_step_optimizer(4))
        self.assertTrue(trainer._should_step_optimizer(5, end_of_epoch=True))
        self.assertTrue(trainer._should_step_optimizer(6, stopping=True))


if __name__ == "__main__":
    unittest.main()
