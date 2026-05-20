from __future__ import annotations

import sys
import unittest
from unittest import mock
from types import SimpleNamespace

import torch
import torch.nn as nn

from ditty import GRPOConfig, RolloutRecord
from ditty.grpo import compute_grpo_loss, gather_completion_logprobs
from ditty.grpo_rollouts import (
    GRPORolloutPreProcessor,
    RolloutBatch,
    RolloutScheduler,
    RolloutSchedulerConfig,
    collate_rollouts,
    compute_old_logprobs,
    generate_rollouts,
    make_no_signal_keepalive_record,
    prepare_rollout_training_context,
    sort_tasks_for_generation_batching,
)
from ditty.loss import GRPOLoss


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 0


class TinyTokenizerOutput(dict):
    @property
    def input_ids(self):
        return self["input_ids"]


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, texts, **kwargs):
        del kwargs
        if isinstance(texts, str):
            return TinyTokenizerOutput(input_ids=[1])
        batch = len(texts)
        return TinyTokenizerOutput(
            input_ids=torch.ones((batch, 1), dtype=torch.long),
            attention_mask=torch.ones((batch, 1), dtype=torch.long),
        )

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in ids)


class FixedTokenLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(use_cache=False)
        self.probe = nn.Parameter(torch.tensor(0.0))
        self.forward_calls = 0

    def forward(self, input_ids=None, attention_mask=None):
        del attention_mask
        self.forward_calls += 1
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros((batch_size, seq_len, 3), dtype=torch.float32, device=input_ids.device)
        logits[:, :, 1] = 10.0
        return SimpleNamespace(logits=logits)


class EmptyRecordingScheduler:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def prime(self, tasks, task_signature_fn) -> None:
        del tasks, task_signature_fn

    def select(self, tasks, *, batch_size, step, task_signature_fn, worker_offset=0):
        del tasks, step, task_signature_fn, worker_offset
        self.batch_sizes.append(batch_size)
        return []

    def update(self, rollout_batch):
        del rollout_batch
        return {}

    def stats_snapshot(self):
        return {}


class CountingSelectiveLM(nn.Module):
    def __init__(self, *, vocab_size: int = 7) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.probe = nn.Parameter(torch.tensor(0.0))
        self.forward_calls = 0
        self.max_forward_batch = 0

    def forward(self, input_ids, *, attention_mask=None, logits_to_keep=None):
        del attention_mask
        self.forward_calls += 1
        self.max_forward_batch = max(self.max_forward_batch, int(input_ids.shape[0]))
        batch_size, seq_len = input_ids.shape
        if logits_to_keep is None:
            positions = torch.arange(seq_len, device=input_ids.device)
        elif isinstance(logits_to_keep, torch.Tensor):
            positions = logits_to_keep.to(device=input_ids.device, dtype=torch.long)
        else:
            keep = int(logits_to_keep)
            positions = torch.arange(seq_len - keep, seq_len, device=input_ids.device)
        token_values = input_ids.index_select(1, positions).float()
        vocab_offsets = torch.arange(self.vocab_size, device=input_ids.device, dtype=torch.float32)
        logits = token_values.unsqueeze(-1) * 0.1 + vocab_offsets.view(1, 1, -1)
        return SimpleNamespace(logits=logits)


class FakeSamplingParams:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class RecordingVllmEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[int]]] = []

    def generate(self, prompts, sampling_params, *, use_tqdm=False):
        del use_tqdm
        prompt_list = list(prompts)
        params = list(sampling_params) if isinstance(sampling_params, list) else [
            sampling_params for _ in prompt_list
        ]
        max_tokens = [int(param.max_tokens) for param in params]
        self.calls.append((prompt_list, max_tokens))
        return [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(token_ids=[max_tokens[prompt_index]])
                    for _ in range(int(params[prompt_index].n))
                ]
            )
            for prompt_index, _prompt in enumerate(prompt_list)
        ]


class GRPOCoreTests(unittest.TestCase):
    def test_sort_tasks_for_generation_batching_groups_generation_caps_stably(self) -> None:
        tasks = [
            {"id": "long-a", "metadata": {"rollout_max_new_tokens": 1024}},
            {"id": "short-a", "metadata": {"rollout_max_new_tokens": 128}},
            {"id": "long-b", "metadata": {"rollout_max_new_tokens": 1024}},
            {"id": "short-b", "metadata": {"rollout_max_new_tokens": 128}},
        ]

        ordered = sort_tasks_for_generation_batching(
            tasks,
            lambda task: int(task["metadata"]["rollout_max_new_tokens"]),
        )

        self.assertEqual([task["id"] for task in ordered], ["short-a", "short-b", "long-a", "long-b"])

    def test_rollout_scheduler_respects_expected_token_budget(self) -> None:
        tasks = [
            {"id": "long-a", "family": "a", "metadata": {"rollout_max_new_tokens": 800}},
            {"id": "long-b", "family": "b", "metadata": {"rollout_max_new_tokens": 800}},
            {"id": "short-a", "family": "c", "metadata": {"rollout_max_new_tokens": 200}},
            {"id": "short-b", "family": "d", "metadata": {"rollout_max_new_tokens": 200}},
        ]
        scheduler = RolloutScheduler(
            RolloutSchedulerConfig(
                seed=7,
                max_signature_fraction=1.0,
                max_expected_tokens_per_batch=1000,
            )
        )

        selected = scheduler.select(tasks, batch_size=4, step=0, task_signature_fn=None)
        total_tokens = sum(int(task["metadata"]["rollout_max_new_tokens"]) for task in selected)

        self.assertLess(len(selected), 4)
        self.assertLessEqual(total_tokens, 1000)
        self.assertEqual(scheduler.last_plan["selection_stopped_reason"], "expected_token_budget")
        self.assertEqual(scheduler.last_plan["selected_expected_tokens"], float(total_tokens))
        self.assertEqual(
            scheduler.last_plan["selected_expected_token_budget_headroom"],
            float(1000 - total_tokens),
        )
        self.assertAlmostEqual(
            scheduler.last_plan["selected_expected_token_budget_fraction"],
            float(total_tokens) / 1000.0,
        )

        batch = RolloutBatch.from_records(
            [
                RolloutRecord(
                    task=task,
                    group_id=str(task["id"]),
                    sample_id=str(task["id"]),
                    prompt_text="prompt",
                    prompt_ids=[1],
                    completion_ids=[2],
                    completion_text="completion",
                    reward=1.0,
                )
                for task in selected
            ],
            current_step=0,
            skip_zero_variance_groups=False,
        )
        metrics = scheduler.update(batch)
        self.assertEqual(
            metrics["rollout_scheduler_expected_token_budget_headroom"],
            float(1000 - total_tokens),
        )
        self.assertAlmostEqual(
            metrics["rollout_scheduler_expected_token_budget_fraction"],
            float(total_tokens) / 1000.0,
        )

    def test_old_logprobs_joins_distributed_chunk_sync_when_local_batch_is_small(self) -> None:
        model = FixedTokenLM()
        batch = {
            "input_ids": torch.ones((1, 4), dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "labels": torch.ones((1, 4), dtype=torch.long),
            "completion_mask": torch.tensor([[False, True, True, True]]),
        }

        with mock.patch("ditty.grpo_rollouts._distributed_max_int", return_value=3) as max_int:
            old_logprobs = compute_old_logprobs(
                model,
                batch,
                GRPOConfig(),
                micro_batch_size=8,
            )

        max_int.assert_called_once()
        self.assertEqual(model.forward_calls, 3)
        self.assertEqual(tuple(old_logprobs.shape), (1, 4))

    def test_prepare_rollout_context_populates_reference_logprobs_when_kl_enabled(self) -> None:
        record = RolloutRecord(
            task={"id": "t0"},
            group_id="g0",
            sample_id="s0",
            prompt_text="prompt",
            prompt_ids=[1, 1],
            completion_ids=[1, 1],
            completion_text="answer",
            reward=1.0,
        )
        ctx = {}

        def reference_logprobs(batch, config):
            self.assertEqual(config.kl_beta, 0.05)
            return torch.full_like(batch["labels"], -0.25, dtype=torch.float32)

        input_ids = prepare_rollout_training_context(
            model=FixedTokenLM(),
            tokenizer=DummyTokenizer(),
            records=[record],
            device=torch.device("cpu"),
            grpo_config=GRPOConfig(kl_beta=0.05),
            ctx=ctx,
            reference_logprob_fn=reference_logprobs,
        )

        self.assertEqual(tuple(input_ids.shape), (1, 4))
        self.assertIn("reference_logprobs", ctx)
        self.assertEqual(tuple(ctx["reference_logprobs"].shape), tuple(ctx["target"].shape))

    def test_grpo_loss_reads_reference_logprobs_default_key(self) -> None:
        logits = torch.zeros((1, 3, 2), dtype=torch.float32, requires_grad=True)
        labels = torch.tensor([[0, 1, 1]], dtype=torch.long)
        mask = torch.tensor([[False, True, True]])
        ctx = {
            "target": labels,
            "mask": mask,
            "old_logprobs": torch.zeros((1, 3), dtype=torch.float32),
            "reference_logprobs": torch.full((1, 3), -1.0, dtype=torch.float32),
            "advantages": torch.ones((1, 3), dtype=torch.float32),
        }

        output = GRPOLoss(
            config=GRPOConfig(kl_beta=0.5, loss_type="dr_grpo", max_completion_length=2),
            target_key="target",
            mask_key="mask",
        ).compute((SimpleNamespace(logits=logits),), ctx)

        self.assertGreater(output.metrics["grpo_kl"], 0.0)

    def test_keepalive_rollout_preserves_loss_microbatch_config(self) -> None:
        preprocessor = GRPORolloutPreProcessor(
            tokenizer=TinyTokenizer(),
            render_prompt=lambda task: "prompt",
            reward_fn=lambda task, completion: 0.0,
            rollouts_per_prompt=2,
            max_new_tokens=1,
            loss_micro_batch_size=8,
            old_logprob_micro_batch_size=8,
        )

        input_ids, ctx = preprocessor.process(
            [{"id": "zero-variance"}],
            {"model": FixedTokenLM(), "device": torch.device("cpu"), "total_steps": 0},
        )

        self.assertIsNotNone(input_ids)
        self.assertEqual(ctx["loss_micro_batch_size"], 8)
        self.assertEqual(ctx["rollout_metrics"]["rollout_no_signal_keepalive_records"], 1.0)

    def test_ray_vllm_generation_aligns_by_prompt_batches_not_task_count(self) -> None:
        engine = RecordingVllmEngine()
        tasks = [{"id": f"t{i}"} for i in range(5)]
        fake_dist = torch.distributed

        with (
            mock.patch.object(fake_dist, "is_available", return_value=True),
            mock.patch.object(fake_dist, "is_initialized", return_value=True),
            mock.patch.object(fake_dist, "get_world_size", return_value=2),
            mock.patch("ditty.grpo_rollouts._distributed_max_int", return_value=6),
            mock.patch.dict(sys.modules, {"vllm": SimpleNamespace(SamplingParams=FakeSamplingParams)}),
        ):
            records = generate_rollouts(
                model=FixedTokenLM(),
                tokenizer=TinyTokenizer(),
                tasks=tasks,
                render_prompt=lambda task: f"prompt {task['id']}",
                reward_fn=lambda task, completion, **kwargs: {"reward": 1.0, "metrics": {}},
                group_id_fn=lambda task: task["id"],
                rollouts_per_prompt=1,
                max_new_tokens=1,
                temperature=0.0,
                top_p=1.0,
                device=torch.device("cpu"),
                step=0,
                rollout_backend="ray_vllm",
                vllm_engine=engine,
                prompt_batch_size=16,
            )

        self.assertEqual(len(records), 5)
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(len(engine.calls[0][0]), 5)

    def test_no_signal_keepalive_uses_tiny_synthetic_noop_sequence(self) -> None:
        record = RolloutRecord(
            task={"id": "long-zero"},
            group_id="g",
            sample_id="s0",
            prompt_text="long prompt",
            prompt_ids=[10, 11, 12],
            completion_ids=[13, 14, 15, 16],
            completion_text="long completion",
            reward=1.0,
        )

        keepalive = make_no_signal_keepalive_record(record, fallback_token_id=2)

        self.assertEqual(keepalive.prompt_text, "")
        self.assertEqual(keepalive.prompt_ids, [2])
        self.assertEqual(keepalive.completion_ids, [2])
        self.assertEqual(keepalive.completion_text, "")
        self.assertEqual(keepalive.token_advantages, [0.0])
        self.assertEqual(keepalive.skip_reason, "no_signal_keepalive")

    def test_grpo_zero_active_tokens_keeps_differentiable_noop_loss(self) -> None:
        logits = torch.zeros((1, 3, 5), dtype=torch.float32, requires_grad=True)
        labels = torch.tensor([[1, 2, 3]], dtype=torch.long)
        mask = torch.zeros_like(labels, dtype=torch.bool)
        old_logprobs = torch.zeros_like(logits[..., 0])
        advantages = torch.zeros_like(logits[..., 0])

        loss, metrics = compute_grpo_loss(
            logits=logits,
            labels=labels,
            mask=mask,
            old_logprobs=old_logprobs,
            advantages=advantages,
            config=GRPOConfig(loss_type="dr_gspo"),
        )

        self.assertTrue(loss.requires_grad)
        self.assertEqual(metrics["grpo_zero_advantage_sample_frac"], 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_dr_gspo_ignores_empty_sequences_without_nan_in_fp16(self) -> None:
        logits = torch.zeros((2, 3, 1), dtype=torch.float16, requires_grad=True)
        labels = torch.zeros((2, 3), dtype=torch.long)
        mask = torch.tensor(
            [
                [False, True, True],
                [False, False, False],
            ],
            dtype=torch.bool,
        )
        old_logprobs = torch.zeros((2, 3), dtype=torch.float16)
        advantages = torch.tensor(
            [
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float16,
        )

        loss, metrics = compute_grpo_loss(
            logits=logits,
            labels=labels,
            mask=mask,
            old_logprobs=old_logprobs,
            advantages=advantages,
            config=GRPOConfig(
                kl_beta=0.0,
                loss_type="dr_gspo",
                max_completion_length=4,
            ),
        )

        self.assertTrue(torch.isfinite(loss.detach()))
        self.assertAlmostEqual(metrics["grpo_ratio_mean"], 1.0, places=3)
        self.assertAlmostEqual(metrics["grpo_sequence_ratio_mean"], 1.0, places=3)
        self.assertEqual(metrics["grpo_nonfinite_log_ratio_frac"], 0.0)

    def test_rollout_scheduler_allows_one_task_over_budget(self) -> None:
        tasks = [
            {"id": "huge", "family": "a", "metadata": {"rollout_max_new_tokens": 800}},
            {"id": "also-huge", "family": "b", "metadata": {"rollout_max_new_tokens": 700}},
        ]
        scheduler = RolloutScheduler(
            RolloutSchedulerConfig(
                seed=3,
                max_signature_fraction=1.0,
                max_expected_tokens_per_batch=100,
                min_batch_size=1,
            )
        )

        selected = scheduler.select(tasks, batch_size=2, step=0, task_signature_fn=None)

        self.assertEqual(len(selected), 1)
        self.assertEqual(scheduler.last_plan["selection_stopped_reason"], "expected_token_budget")

    def test_rollout_scheduler_resume_preserves_active_config(self) -> None:
        old_scheduler = RolloutScheduler(RolloutSchedulerConfig(seed=3))
        old_scheduler.select(
            [{"id": "old", "metadata": {"rollout_max_new_tokens": 128}}],
            batch_size=1,
            step=0,
            task_signature_fn=None,
        )

        resumed = RolloutScheduler(
            RolloutSchedulerConfig(
                seed=3,
                max_expected_tokens_per_batch=256,
                min_batch_size=1,
            )
        )
        resumed.load_state_dict(old_scheduler.state_dict())
        resumed.select(
            [
                {"id": "a", "metadata": {"rollout_max_new_tokens": 200}},
                {"id": "b", "metadata": {"rollout_max_new_tokens": 200}},
            ],
            batch_size=2,
            step=1,
            task_signature_fn=None,
        )

        self.assertEqual(resumed.config.max_expected_tokens_per_batch, 256)
        self.assertEqual(resumed.last_plan["max_expected_tokens_per_batch"], 256)
        self.assertEqual(resumed.last_plan["selection_stopped_reason"], "expected_token_budget")

    def test_generate_rollouts_records_completion_cap_metrics(self) -> None:
        records = generate_rollouts(
            model=FixedTokenLM(),
            tokenizer=TinyTokenizer(),
            tasks=[{"id": "t0"}],
            render_prompt=lambda task: "prompt",
            reward_fn=lambda task, completion: 1.0,
            group_id_fn=lambda task: task["id"],
            rollouts_per_prompt=1,
            max_new_tokens=2,
            temperature=0.0,
            top_p=1.0,
            device=torch.device("cpu"),
            step=0,
            rollout_backend="manual",
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(len(records[0].completion_ids), 2)
        self.assertEqual(records[0].reward_metrics["completion_tokens"], 2.0)
        self.assertEqual(records[0].reward_metrics["completion_max_new_tokens"], 2.0)
        self.assertEqual(records[0].reward_metrics["completion_hit_max_new_tokens"], 1.0)

    def test_ray_vllm_distributed_rollouts_use_rank_aligned_slots(self) -> None:
        engine = RecordingVllmEngine()
        tasks = [
            {"id": "a", "cap": 5},
            {"id": "b", "cap": 7},
            {"id": "c", "cap": 5},
        ]

        with (
            mock.patch.dict(sys.modules, {"vllm": SimpleNamespace(SamplingParams=FakeSamplingParams)}),
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", return_value=2),
            mock.patch("torch.distributed.all_reduce", side_effect=lambda tensor, op=None: None),
        ):
            records = generate_rollouts(
                model=FixedTokenLM(),
                tokenizer=TinyTokenizer(),
                tasks=tasks,
                render_prompt=lambda task: f"prompt-{task['id']}",
                reward_fn=lambda task, completion: 1.0,
                group_id_fn=lambda task: task["id"],
                rollouts_per_prompt=1,
                max_new_tokens=lambda task: int(task["cap"]),
                temperature=0.0,
                top_p=1.0,
                device=torch.device("cpu"),
                step=0,
                rollout_backend="ray_vllm",
                vllm_engine=engine,
                prompt_batch_size=2,
            )

        self.assertEqual(
            engine.calls,
            [
                (["prompt-a", "prompt-b"], [5, 7]),
                (["prompt-c"], [5]),
            ],
        )
        self.assertEqual(len(records), 3)

    def test_rollout_preprocessor_applies_scheduler_to_oversample_target(self) -> None:
        scheduler = EmptyRecordingScheduler()
        preprocessor = GRPORolloutPreProcessor(
            tokenizer=TinyTokenizer(),
            render_prompt=lambda task: "prompt",
            reward_fn=lambda task, completion: 0.0,
            rollouts_per_prompt=1,
            max_new_tokens=1,
            skip_zero_variance_groups=False,
            rollout_scheduler=scheduler,
            task_pool=[{"id": str(index)} for index in range(8)],
            oversample_multiplier=1.5,
        )

        _, ctx = preprocessor.process(
            [{"id": str(index)} for index in range(4)],
            {"model": FixedTokenLM(), "device": torch.device("cpu"), "total_steps": 0},
        )

        self.assertEqual(scheduler.batch_sizes, [6])
        self.assertEqual(ctx["oversampled_extra_tasks"], 0)
        self.assertEqual(ctx["scheduled_tasks"], [])

    def test_rollout_preprocessor_config_serializes_callable_max_new_tokens(self) -> None:
        preprocessor = GRPORolloutPreProcessor(
            tokenizer=DummyTokenizer(),
            render_prompt=lambda task: "prompt",
            reward_fn=lambda task, completion: 0.0,
            max_new_tokens=lambda task: int(task["cap"]),
            skip_zero_variance_groups=False,
        )

        self.assertEqual(preprocessor.config()["max_new_tokens"], "task_callable")

    def test_dr_grpo_uses_completion_constant_and_causal_alignment(self) -> None:
        labels = torch.zeros((1, 4), dtype=torch.long)
        logits = torch.zeros((1, 4, 1), dtype=torch.float32)
        completion_mask = torch.tensor([[False, False, True, True]])
        old_logprobs = torch.zeros((1, 4), dtype=torch.float32)
        advantages = torch.tensor([[0.0, 99.0, 2.0, 3.0]], dtype=torch.float32)

        loss, metrics = compute_grpo_loss(
            logits=logits,
            labels=labels,
            mask=completion_mask,
            old_logprobs=old_logprobs,
            advantages=advantages,
            config=GRPOConfig(
                clip_epsilon=0.2,
                kl_beta=0.0,
                loss_type="dr_grpo",
                max_completion_length=4,
            ),
        )

        self.assertAlmostEqual(float(loss), -1.25, places=6)
        self.assertAlmostEqual(metrics["grpo_policy_denominator"], 4.0, places=6)
        self.assertAlmostEqual(metrics["grpo_advantage_mean"], 2.5, places=6)

    def test_dr_gspo_uses_sequence_ratio_with_completion_constant(self) -> None:
        labels = torch.zeros((1, 3), dtype=torch.long)
        logits = torch.zeros((1, 3, 1), dtype=torch.float32)
        completion_mask = torch.tensor([[False, True, True]])
        old_logprobs = torch.tensor([[0.0, -0.69314718056, 0.69314718056]], dtype=torch.float32)
        advantages = torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float32)

        dr_gspo_loss, dr_gspo_metrics = compute_grpo_loss(
            logits=logits,
            labels=labels,
            mask=completion_mask,
            old_logprobs=old_logprobs,
            advantages=advantages,
            config=GRPOConfig(
                clip_epsilon=10.0,
                kl_beta=0.0,
                loss_type="dr_gspo",
                max_completion_length=4,
            ),
        )
        dr_grpo_loss, _ = compute_grpo_loss(
            logits=logits,
            labels=labels,
            mask=completion_mask,
            old_logprobs=old_logprobs,
            advantages=advantages,
            config=GRPOConfig(
                clip_epsilon=10.0,
                kl_beta=0.0,
                loss_type="dr_grpo",
                max_completion_length=4,
            ),
        )

        self.assertAlmostEqual(float(dr_gspo_loss), -0.5, places=6)
        self.assertAlmostEqual(dr_gspo_metrics["grpo_policy_denominator"], 4.0, places=6)
        self.assertAlmostEqual(dr_gspo_metrics["grpo_sequence_ratio_mean"], 1.0, places=6)
        self.assertAlmostEqual(float(dr_grpo_loss), -0.625, places=6)

    def test_dapo_normalizes_by_active_completion_tokens(self) -> None:
        labels = torch.zeros((1, 4), dtype=torch.long)
        logits = torch.zeros((1, 4, 1), dtype=torch.float32)
        completion_mask = torch.tensor([[False, False, True, True]])
        old_logprobs = torch.zeros((1, 4), dtype=torch.float32)
        advantages = torch.tensor([[0.0, 99.0, 2.0, 3.0]], dtype=torch.float32)

        loss, metrics = compute_grpo_loss(
            logits=logits,
            labels=labels,
            mask=completion_mask,
            old_logprobs=old_logprobs,
            advantages=advantages,
            config=GRPOConfig(kl_beta=0.0, loss_type="dapo"),
        )

        self.assertAlmostEqual(float(loss), -2.5, places=6)
        self.assertAlmostEqual(metrics["grpo_policy_denominator"], 2.0, places=6)

    def test_selective_logprob_backend_matches_dense_backend(self) -> None:
        logits = torch.tensor(
            [
                [[1.0, 0.0, -1.0], [0.5, 1.5, -0.5]],
                [[-0.25, 0.25, 1.25], [2.0, -1.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([[0, 1], [2, 0]], dtype=torch.long)
        mask = torch.tensor([[True, False], [True, True]])

        selective = gather_completion_logprobs(logits, labels, valid_mask=mask, backend="selective")
        dense = gather_completion_logprobs(logits, labels, valid_mask=mask, backend="dense")

        self.assertTrue(torch.allclose(selective[mask], dense[mask], atol=1e-6))
        self.assertEqual(float(selective[0, 1]), 0.0)

    def test_collate_places_explicit_token_advantages_on_completion_tokens(self) -> None:
        record = RolloutRecord(
            task={},
            group_id="g0",
            sample_id="s0",
            prompt_text="prompt",
            prompt_ids=[10, 11],
            completion_ids=[12, 13],
            completion_text="answer",
            reward=1.0,
            token_advantages=[0.5, -0.25],
        )

        batch = collate_rollouts([record], DummyTokenizer(), torch.device("cpu"), GRPOConfig())

        self.assertEqual(batch["completion_mask"].tolist(), [[False, False, True, True]])
        self.assertEqual(batch["advantages"].tolist(), [[0.0, 0.0, 0.5, -0.25]])

    def test_collate_respects_explicit_completion_mask_for_agent_observations(self) -> None:
        records = [
            RolloutRecord(
                task={},
                group_id="g0",
                sample_id="s0",
                prompt_text="prompt",
                prompt_ids=[10, 11],
                completion_ids=[12, 13, 14],
                completion_text="assistant plus observation",
                reward=1.0,
                completion_mask=[1, 0, 1],
            ),
            RolloutRecord(
                task={},
                group_id="g0",
                sample_id="s1",
                prompt_text="prompt",
                prompt_ids=[10, 11],
                completion_ids=[12, 13, 14],
                completion_text="assistant plus observation",
                reward=0.0,
                completion_mask=[1, 0, 1],
            ),
        ]

        batch = collate_rollouts(records, DummyTokenizer(), torch.device("cpu"), GRPOConfig())

        self.assertEqual(
            batch["completion_mask"].tolist(),
            [
                [False, False, True, False, True],
                [False, False, True, False, True],
            ],
        )
        self.assertEqual(batch["advantages"][0].tolist()[3], 0.0)
        self.assertEqual(batch["advantages"][1].tolist()[3], 0.0)
        self.assertGreater(batch["advantages"][0].tolist()[2], 0.0)
        self.assertLess(batch["advantages"][1].tolist()[2], 0.0)

    def test_collate_masks_explicit_token_advantages_too(self) -> None:
        record = RolloutRecord(
            task={},
            group_id="g0",
            sample_id="s0",
            prompt_text="prompt",
            prompt_ids=[10],
            completion_ids=[11, 12, 13],
            completion_text="answer",
            reward=1.0,
            completion_mask=[1, 0, 1],
            token_advantages=[0.5, 99.0, -0.25],
        )

        batch = collate_rollouts([record], DummyTokenizer(), torch.device("cpu"), GRPOConfig())

        self.assertEqual(batch["completion_mask"].tolist(), [[False, True, False, True]])
        self.assertEqual(batch["advantages"].tolist(), [[0.0, 0.5, 0.0, -0.25]])

    def test_old_logprobs_microbatch_matches_full_forward(self) -> None:
        labels = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6],
                [2, 3, 4, 5, 6, 1],
                [3, 4, 5, 6, 1, 2],
                [4, 5, 6, 1, 2, 3],
                [5, 6, 1, 2, 3, 4],
            ],
            dtype=torch.long,
        )
        batch = {
            "input_ids": labels.clone(),
            "labels": labels.clone(),
            "attention_mask": torch.ones_like(labels),
            "completion_mask": torch.tensor(
                [
                    [False, False, True, True, True, True],
                    [False, True, True, True, False, False],
                    [False, False, False, True, True, True],
                    [False, True, True, True, True, True],
                    [False, False, True, True, False, False],
                ]
            ),
        }
        config = GRPOConfig(logprob_backend="selective", logprob_chunk_size=2)

        full_model = CountingSelectiveLM()
        full = compute_old_logprobs(full_model, batch, config)

        micro_model = CountingSelectiveLM()
        micro = compute_old_logprobs(micro_model, batch, config, micro_batch_size=2)

        self.assertTrue(torch.allclose(micro, full, atol=1e-6))
        self.assertEqual(micro_model.forward_calls, 3)
        self.assertLessEqual(micro_model.max_forward_batch, 2)

if __name__ == "__main__":
    unittest.main()
