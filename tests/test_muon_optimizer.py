import unittest
import tempfile
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

from ditty.optimizers import ChainedOptimizer
from ditty.loss import MSELoss
from ditty.model_factory import ModelFactory
from ditty.pipeline import Pipeline
from ditty.processors import PostProcessor


def assert_param_in(testcase, param, params):
    testcase.assertTrue(any(candidate is param for candidate in params))


class FakeMuon(torch.optim.Optimizer):
    last_instance = None

    def __init__(
        self,
        params,
        *,
        lr,
        weight_decay,
        momentum,
        nesterov,
        ns_steps,
        adjust_lr_fn,
    ):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        super().__init__(params, defaults)
        FakeMuon.last_instance = self

    def step(self, closure=None):
        return closure() if closure is not None else None


class TinyMuonModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.hidden = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 8, bias=False)


class TinyRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4, 4)
        self.readout = nn.Linear(4, 4)

    def forward(self, batch):
        return self.readout(torch.tanh(self.hidden(batch)))


class ZeroTarget(PostProcessor):
    def process(self, model_output, ctx):
        ctx["target"] = torch.zeros_like(model_output[0])
        return model_output, ctx


class MuonOptimizerTests(unittest.TestCase):
    def _pipeline_shell(self):
        pipeline = object.__new__(Pipeline)
        pipeline.loss_calculator = nn.Linear(1, 1)
        pipeline.weight_decay = 0.01
        pipeline.muon_lr = 0.02
        pipeline.muon_weight_decay = 0.0
        pipeline.muon_momentum = 0.95
        pipeline.muon_nesterov = True
        pipeline.muon_ns_steps = 5
        pipeline.muon_adjust_lr_fn = "original"
        return pipeline

    def test_muon_split_uses_torch_muon_for_hidden_matrices_only(self):
        model = TinyMuonModel()
        pipeline = self._pipeline_shell()

        with mock.patch.object(torch.optim, "Muon", FakeMuon, create=True):
            optimizer = pipeline._create_muon_optimizer(model, lr=3e-4)

        self.assertIsInstance(optimizer, ChainedOptimizer)
        self.assertIs(FakeMuon.last_instance.param_groups[0]["params"][0], model.hidden.weight)
        self.assertEqual(FakeMuon.last_instance.param_groups[0]["lr"], 0.02)
        self.assertEqual(FakeMuon.last_instance.param_groups[0]["weight_decay"], 0.0)
        self.assertEqual(FakeMuon.last_instance.param_groups[0]["momentum"], 0.95)

        adamw_params = optimizer.optimizers[0].param_groups[0]["params"]
        assert_param_in(self, model.embed.weight, adamw_params)
        assert_param_in(self, model.hidden.bias, adamw_params)
        assert_param_in(self, model.norm.weight, adamw_params)
        assert_param_in(self, model.norm.bias, adamw_params)
        assert_param_in(self, model.lm_head.weight, adamw_params)
        assert_param_in(self, pipeline.loss_calculator.weight, adamw_params)
        assert_param_in(self, pipeline.loss_calculator.bias, adamw_params)

    def test_muon_backend_requires_upstream_torch_optimizer(self):
        model = TinyMuonModel()
        pipeline = self._pipeline_shell()

        with mock.patch.object(torch.optim, "Muon", None, create=True):
            with self.assertRaisesRegex(ImportError, "torch.optim.Muon"):
                pipeline._create_muon_optimizer(model, lr=3e-4)

    def test_chained_optimizer_steps_and_serializes_child_states(self):
        first = nn.Parameter(torch.tensor([1.0]))
        second = nn.Parameter(torch.tensor([1.0]))
        optimizer = ChainedOptimizer(
            [
                torch.optim.SGD([first], lr=0.1, momentum=0.9),
                torch.optim.AdamW([second], lr=0.1),
            ]
        )

        first.grad = torch.tensor([1.0])
        second.grad = torch.tensor([1.0])
        optimizer.step()

        state = optimizer.state_dict()
        self.assertEqual(len(state["param_groups"]), 2)
        self.assertTrue(state["state"])

        restored = ChainedOptimizer(
            [
                torch.optim.SGD([nn.Parameter(torch.tensor([1.0]))], lr=0.1, momentum=0.9),
                torch.optim.AdamW([nn.Parameter(torch.tensor([1.0]))], lr=0.1),
            ]
        )
        restored.load_state_dict(state)
        self.assertEqual(len(restored.param_groups), 2)
        self.assertTrue(restored.state)

    @unittest.skipUnless(hasattr(torch.optim, "Muon"), "requires torch.optim.Muon")
    def test_pipeline_runs_two_steps_with_real_torch_muon(self):
        dataset = [torch.randn(4) for _ in range(4)]
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Pipeline(
                model_factory=ModelFactory.from_instance(TinyRegressionModel()),
                dataset=dataset,
                loss_calculator=MSELoss(),
                postprocessors=[ZeroTarget()],
                output_dir=tmp,
                fp16=False,
                accelerator_mixed_precision="no",
                batch_size=2,
                num_workers=0,
                optim_backend="muon",
                lr=1e-3,
                muon_lr=1e-3,
                epochs=1,
                max_steps=2,
                checkpoint_every=0,
                save_final_checkpoint=True,
                load_checkpoint=False,
                gradient_checkpointing=False,
                log_every=1,
            )

            pipeline.run()
            self.assertTrue((Path(tmp) / "checkpoints" / "checkpoint_0" / "optimizer.bin").exists())

        self.assertEqual(pipeline.optimizer.optimizers[1].__class__, torch.optim.Muon)


if __name__ == "__main__":
    unittest.main()
