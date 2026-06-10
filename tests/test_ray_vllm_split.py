import inspect
import unittest
from pathlib import Path

import torch

from ditty.grpo_rollouts import GRPORolloutPreProcessor, generate_rollouts
from ditty.ray_vllm_engine import RayVllmActor, RayVllmRolloutEngine
from ditty.vllm_worker_extension import StatelessWeightUpdateWorkerExtension


class _TokenizerOutput(dict):
    @property
    def input_ids(self):
        return [1, 2]


class _Tokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            return _TokenizerOutput(input_ids=[1, 2])
        return _TokenizerOutput(
            input_ids=torch.ones((len(texts), 2), dtype=torch.long),
            attention_mask=torch.ones((len(texts), 2), dtype=torch.long),
        )

    def decode(self, ids, skip_special_tokens=True):
        return "{}"


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"use_cache": False})()


class RayVllmSplitTests(unittest.TestCase):
    def test_old_colocated_vllm_backend_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "removed colocated backend"):
            generate_rollouts(
                model=_Model(),
                tokenizer=_Tokenizer(),
                tasks=[{"id": "t0"}],
                render_prompt=lambda task: "prompt",
                reward_fn=lambda task, completion: {"reward": 1.0, "metrics": {}},
                group_id_fn=lambda task: task["id"],
                rollouts_per_prompt=1,
                max_new_tokens=1,
                temperature=0.0,
                top_p=1.0,
                device=torch.device("cpu"),
                step=0,
                rollout_backend="vllm",
                vllm_engine=object(),
            )

    def test_worker_extension_uses_stateless_side_group(self):
        source = inspect.getsource(StatelessWeightUpdateWorkerExtension)
        self.assertIn("StatelessProcessGroup.create", source)
        self.assertIn("PyNcclCommunicator", source)
        self.assertNotIn("dist.broadcast", source)

    def test_ray_engine_uses_placement_group_and_rank0_transfer(self):
        source = inspect.getsource(RayVllmRolloutEngine)
        self.assertIn("placement_group", source)
        self.assertIn("PlacementGroupSchedulingStrategy", source)
        self.assertIn("num_gpus=0", source)
        self.assertIn("dist.barrier()", source)
        self.assertIn("param.full_tensor()", source)
        self.assertIn("self.model_update_group.broadcast", source)

    def test_ray_actor_uses_vllm_ray_bundle_indices(self):
        source = inspect.getsource(RayVllmRolloutEngine)
        actor_source = inspect.getsource(RayVllmActor)
        self.assertIn("VLLM_RAY_BUNDLE_INDICES", actor_source)
        self.assertIn("unset_cuda_visible_devices", source)
        self.assertIn("unset_cuda_visible_devices", actor_source)
        self.assertIn("unset_cuda_visible_devices: bool = True", source)

    def test_ray_engine_can_disable_custom_all_reduce_and_cleans_pg(self):
        source = inspect.getsource(RayVllmRolloutEngine)
        actor_source = inspect.getsource(RayVllmActor)
        self.assertIn("disable_custom_all_reduce", source)
        self.assertIn("disable_custom_all_reduce", actor_source)
        self.assertIn("placement_group_ready_timeout_s", source)
        self.assertIn("remove_placement_group", source)

    def test_ray_engine_can_pin_vllm_physical_devices(self):
        source = inspect.getsource(RayVllmRolloutEngine)
        actor_source = inspect.getsource(RayVllmActor)
        executor_source = (Path(__file__).parents[1] / "lib" / "ditty" / "vllm_ray_executor.py").read_text()
        self.assertIn("cuda_visible_devices", source)
        self.assertIn('os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices', actor_source)
        self.assertIn("DITTY_VLLM_CUDA_VISIBLE_DEVICES", actor_source)
        self.assertIn("ditty.vllm_ray_executor.DittyRayDistributedExecutor", actor_source)
        self.assertIn("_update_noset_device_env_vars", executor_source)
        self.assertIn("current_platform.device_control_env_var", executor_source)

    def test_ray_engine_supports_mp_vllm_backend_without_vllm_pg(self):
        source = inspect.getsource(RayVllmRolloutEngine)
        actor_source = inspect.getsource(RayVllmActor)
        self.assertIn('distributed_executor_backend: str = "ray"', source)
        self.assertIn('self.distributed_executor_backend == "ray"', source)
        self.assertIn('distributed_executor_backend=self.distributed_executor_backend', source)
        self.assertIn('os.environ.pop("VLLM_RAY_BUNDLE_INDICES", None)', actor_source)

    def test_weight_sync_prefers_node_ip_rendezvous(self):
        source = inspect.getsource(__import__("ditty.ray_vllm_engine", fromlist=["_default_host"])._default_host)
        self.assertIn("DITTY_RAY_VLLM_WEIGHT_SYNC_HOST", source)
        self.assertIn("get_ip", source)
        self.assertIn("not host.startswith(\"127.\")", source)

    def test_ray_actor_prefers_native_vllm_weight_transfer(self):
        actor_source = inspect.getsource(RayVllmActor)
        engine_source = inspect.getsource(RayVllmRolloutEngine)
        self.assertIn("WeightTransferConfig", actor_source)
        self.assertIn("init_weight_transfer_engine", actor_source)
        self.assertIn('"update_weights"', actor_source)
        self.assertIn("def update_weights(", actor_source)
        self.assertIn('weight_transfer_backend: str | None = "nccl"', engine_source)
        self.assertIn("ray_vllm weight sync metadata start", engine_source)

    def test_ray_actor_shutdown_uses_vllm_engine_core_shutdown(self):
        actor_shutdown = inspect.getsource(RayVllmActor.shutdown)
        engine_shutdown = inspect.getsource(RayVllmRolloutEngine.shutdown)
        self.assertIn("llm_engine", actor_shutdown)
        self.assertIn("engine_core", actor_shutdown)
        self.assertIn("engine_core.shutdown", actor_shutdown)
        self.assertIn("__ray_shutdown__", inspect.getsource(RayVllmActor))
        self.assertIn("actor.shutdown.remote", engine_shutdown)

    def test_ray_engine_gracefully_shuts_down_actor_before_force_kill(self):
        cleanup_source = inspect.getsource(RayVllmRolloutEngine._cleanup_rank0)
        self.assertIn("actor.shutdown.remote", cleanup_source)
        self.assertIn("ray.kill", cleanup_source)

    def test_ray_vllm_generation_chunks_large_global_requests(self):
        engine = RayVllmRolloutEngine(
            model_path="/model",
            tensor_parallel_size=4,
            gpu_memory_utilization=0.8,
            max_sequences_per_request=216,
            max_expected_new_tokens_per_request=65536,
        )
        sampling_params = [
            {"n": 18, "max_tokens": 1600},
            {"n": 18, "max_tokens": 1600},
            {"n": 18, "max_tokens": 1600},
            {"n": 18, "max_tokens": 128},
            {"n": 18, "max_tokens": 128},
        ]
        self.assertEqual(
            engine._combined_request_chunks(sampling_params),
            [(0, 2), (2, 5)],
        )

    def test_ray_vllm_generation_batches_prompts_across_ranks(self):
        source = inspect.getsource(RayVllmRolloutEngine.generate)
        self.assertIn("combined_prompts", source)
        self.assertIn("combined_sampling_params", source)
        self.assertIn("_combined_request_chunks", source)
        self.assertIn("outputs_by_rank[payload_rank][prompt_index]", source)

    def test_ray_vllm_preprocessor_avoids_posthoc_object_rebalance(self):
        source = inspect.getsource(GRPORolloutPreProcessor.process)
        self.assertIn('self.rollout_backend != "ray_vllm"', source)
        self.assertIn("rollout_rebalance_skipped_rank_aligned_vllm", source)


if __name__ == "__main__":
    unittest.main()
