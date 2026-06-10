import unittest
from unittest import mock

import torch
import torch.nn as nn

import ditty.loss as loss_mod
from ditty.loss import MTPAuxFusedCrossEntropyLoss
from ditty.trainer import _slice_batch_mapping


class RaisingHead(nn.Linear):
    def forward(self, input):  # pragma: no cover - this must not be called
        raise AssertionError("MTP loss must pass head weights to fused CE, not call the head")


class TinyMTPModel(nn.Module):
    def __init__(self, hidden_size=4, vocab_size=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.heads = nn.ModuleList([RaisingHead(hidden_size, vocab_size, bias=False) for _ in range(3)])
        self.calls = []

    def forward_mtp_step(
        self,
        *,
        input_ids,
        positions,
        previous_hidden_states,
        step_idx,
        attention_mask=None,
    ):
        self.calls.append(
            {
                "step_idx": step_idx,
                "input_ids": input_ids.detach().clone(),
                "positions": None if positions is None else positions.detach().clone(),
                "previous_shape": tuple(previous_hidden_states.shape),
                "attention_mask": None if attention_mask is None else attention_mask.detach().clone(),
            }
        )
        return previous_hidden_states + self.embed(input_ids)

    def get_mtp_output_embeddings(self, step_idx):
        return self.heads[step_idx]


class MTPAuxFusedCrossEntropyLossTests(unittest.TestCase):
    def test_trainer_context_slicing_keeps_batch_aligned_tensors(self):
        mapping = {
            "input_ids": torch.arange(12).view(3, 4),
            "position_ids": torch.arange(12).view(3, 4) + 100,
            "target": torch.arange(12).view(3, 4) + 200,
            "scalar": torch.tensor(1),
            "metadata": "kept",
        }

        sliced = _slice_batch_mapping(mapping, 1, 3, 3)

        self.assertEqual(sliced["input_ids"].tolist(), mapping["input_ids"][1:3].tolist())
        self.assertEqual(sliced["position_ids"].tolist(), mapping["position_ids"][1:3].tolist())
        self.assertEqual(sliced["target"].tolist(), mapping["target"][1:3].tolist())
        self.assertIs(sliced["metadata"], mapping["metadata"])
        self.assertTrue(torch.equal(sliced["scalar"], mapping["scalar"]))

    def test_mtp1_uses_shifted_teacher_inputs_targets_and_masks(self):
        model = TinyMTPModel()
        hidden = torch.randn(2, 4, 4, requires_grad=True)
        input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        target = torch.tensor([[2, 3, 4, -100], [6, -100, 8, -100]])
        mask = target.ne(-100)
        attention_mask = torch.ones_like(input_ids)
        captured = []

        def fake_fused(hidden, weight, target, *, bias=None, mask=None, **kwargs):
            del weight, bias, kwargs
            captured.append(
                {
                    "hidden_shape": tuple(hidden.shape),
                    "target": target.detach().clone(),
                    "mask": None if mask is None else mask.detach().clone(),
                }
            )
            return hidden.sum() * 0.0 + target.clamp_min(0).float().sum() * 0.0

        loss = MTPAuxFusedCrossEntropyLoss(
            depth=1,
            backend="chunked",
            target_key="target",
            mask_key="mask",
        )
        ctx = {
            "model": model,
            "input_ids": input_ids,
            "target": target,
            "mask": mask,
            "forward_kwargs": {"attention_mask": attention_mask},
        }
        with mock.patch.object(loss_mod, "fused_linear_cross_entropy", fake_fused):
            output = loss.compute((hidden,), ctx)

        self.assertEqual(len(model.calls), 1)
        self.assertTrue(torch.equal(model.calls[0]["input_ids"], input_ids[:, 1:]))
        self.assertEqual(model.calls[0]["previous_shape"], (2, 3, 4))
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["hidden_shape"], (2, 3, 4))
        self.assertTrue(torch.equal(captured[0]["target"], target[:, 1:]))
        self.assertTrue(torch.equal(captured[0]["mask"], mask[:, 1:]))
        self.assertIn("mtp_1_ce", output.metrics)
        self.assertEqual(output.metrics["tokens/mtp_1_valid"], float(mask[:, 1:].sum()))

    def test_all_head_mode_recurses_over_future_offsets(self):
        model = TinyMTPModel()
        hidden = torch.randn(1, 5, 4, requires_grad=True)
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        target = torch.tensor([[2, 3, 4, 5, -100]])
        captured_targets = []

        def fake_fused(hidden, weight, target, *, bias=None, mask=None, **kwargs):
            del hidden, weight, bias, mask, kwargs
            captured_targets.append(target.detach().clone())
            return torch.tensor(float(len(captured_targets)), requires_grad=True)

        loss = MTPAuxFusedCrossEntropyLoss(
            depth=3,
            beta=0.6,
            backend="chunked",
            target_key="target",
            mask_key="mask",
        )
        ctx = {
            "model": model,
            "input_ids": input_ids,
            "target": target,
            "mask": target.ne(-100),
            "forward_kwargs": {"attention_mask": torch.ones_like(input_ids)},
        }
        with mock.patch.object(loss_mod, "fused_linear_cross_entropy", fake_fused):
            output = loss.compute((hidden,), ctx)

        self.assertEqual([call["step_idx"] for call in model.calls], [0, 1, 2])
        self.assertEqual([tuple(call["input_ids"].shape) for call in model.calls], [(1, 4), (1, 3), (1, 2)])
        self.assertTrue(torch.equal(captured_targets[0], target[:, 1:]))
        self.assertTrue(torch.equal(captured_targets[1], target[:, 2:]))
        self.assertTrue(torch.equal(captured_targets[2], target[:, 3:]))
        weights = torch.tensor([1.0, 0.6, 0.36])
        weights = weights / weights.sum()
        expected = float((weights * torch.tensor([1.0, 2.0, 3.0])).sum())
        self.assertAlmostEqual(output.loss.item(), expected, places=6)

    def test_zero_valid_shifted_tokens_returns_zero_without_nan(self):
        model = TinyMTPModel()
        hidden = torch.randn(1, 3, 4, requires_grad=True)
        input_ids = torch.tensor([[1, 2, 3]])
        target = torch.full((1, 3), -100)
        mask = torch.zeros_like(target, dtype=torch.bool)
        loss = MTPAuxFusedCrossEntropyLoss(
            depth=1,
            backend="chunked",
            target_key="target",
            mask_key="mask",
        )
        output = loss.compute(
            (hidden,),
            {
                "model": model,
                "input_ids": input_ids,
                "target": target,
                "mask": mask,
            },
        )

        self.assertEqual(output.loss.item(), 0.0)
        self.assertFalse(torch.isnan(output.loss))
        self.assertEqual(output.metrics["tokens/mtp_1_valid"], 0.0)

    def test_actual_chunked_backend_backprops_without_calling_projection_forward(self):
        model = TinyMTPModel(hidden_size=4, vocab_size=16)
        hidden = torch.randn(2, 4, 4, requires_grad=True)
        input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        target = torch.tensor([[2, 3, 4, -100], [3, 2, 1, -100]])
        mask = target.ne(-100)
        loss = MTPAuxFusedCrossEntropyLoss(
            depth=1,
            backend="chunked",
            target_key="target",
            mask_key="mask",
        )

        output = loss.compute(
            (hidden,),
            {
                "model": model,
                "input_ids": input_ids,
                "target": target,
                "mask": mask,
            },
        )
        output.loss.backward()

        self.assertIsNotNone(hidden.grad)
        self.assertIsNotNone(model.heads[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
