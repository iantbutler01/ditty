import inspect
import unittest

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


if __name__ == "__main__":
    unittest.main()
