import unittest
import inspect

from ditty.ray_training import (
    RayModuleLaunchConfig,
    RayTrainModuleLaunchConfig,
    _gcs_uri_from_checkpoint,
    _read_ditty_checkpoint_num,
    _restore_local_ray_train_checkpoint,
    _restore_gcs_ray_train_checkpoint,
    _normal_module_args,
    _parse_env,
    _runtime_env_with_env_vars,
    build_arg_parser,
    config_from_args,
    launch_ray_train_module,
    ray_train_config_from_args,
)
from ditty.checkpoint import CheckpointManager


class RayTrainingConfigTests(unittest.TestCase):
    def test_normal_module_args_strips_separator(self):
        self.assertEqual(_normal_module_args(["--", "--flag", "value"]), ["--flag", "value"])
        self.assertEqual(_normal_module_args(["--flag"]), ["--flag"])

    def test_parse_env_requires_key_value(self):
        self.assertEqual(_parse_env(["A=1", "B=two=three"]), {"A": "1", "B": "two=three"})
        with self.assertRaisesRegex(ValueError, "KEY=VALUE"):
            _parse_env(["BROKEN"])

    def test_runtime_env_merges_env_vars(self):
        self.assertEqual(
            _runtime_env_with_env_vars({"working_dir": "/repo", "env_vars": {"A": "old"}}, {"A": "1", "B": "2"}),
            {"working_dir": "/repo", "env_vars": {"A": "1", "B": "2"}},
        )
        self.assertEqual(_runtime_env_with_env_vars(None, {"PYTHONPATH": "/repo"}), {"env_vars": {"PYTHONPATH": "/repo"}})

    def test_cli_config_keeps_module_args_and_runtime_env(self):
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--module",
                "training.grpo_ditty_pipeline",
                "--num-workers",
                "4",
                "--num-cpus-per-worker",
                "8",
                "--env",
                "PYTHONPATH=/data/voidstorm:/data/ditty/lib",
                "--working-dir",
                "/data/voidstorm",
                "--",
                "--rollout-backend",
                "ray_vllm",
            ]
        )
        config = config_from_args(args)
        self.assertIsInstance(config, RayModuleLaunchConfig)
        self.assertEqual(config.module, "training.grpo_ditty_pipeline")
        self.assertEqual(config.num_workers, 4)
        self.assertEqual(config.num_cpus_per_worker, 8.0)
        self.assertEqual(config.env["PYTHONPATH"], "/data/voidstorm:/data/ditty/lib")
        self.assertEqual(config.runtime_env, {"working_dir": "/data/voidstorm"})
        self.assertEqual(config.module_args, ["--rollout-backend", "ray_vllm"])

    def test_ray_train_cli_config(self):
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--launcher",
                "ray-train",
                "--module",
                "training.grpo_ditty_pipeline",
                "--num-workers",
                "4",
                "--num-cpus-per-worker",
                "8",
                "--ray-train-storage-path",
                "/data/ray_results/grpo",
                "--ray-train-run-name",
                "stage1",
                "--ray-train-max-failures",
                "3",
                "--ray-train-num-checkpoints-to-keep",
                "3",
                "--restore-checkpoint-to",
                "/data/runs/grpo/stage1",
                "--",
                "--rollout-backend",
                "ray_vllm",
            ]
        )
        config = ray_train_config_from_args(args)
        self.assertIsInstance(config, RayTrainModuleLaunchConfig)
        self.assertEqual(config.num_workers, 4)
        self.assertEqual(config.storage_path, "/data/ray_results/grpo")
        self.assertEqual(config.run_name, "stage1")
        self.assertEqual(config.max_failures, 3)
        self.assertEqual(config.num_checkpoints_to_keep, 3)
        self.assertEqual(config.restore_checkpoint_to, "/data/runs/grpo/stage1")
        self.assertEqual(config.module_args, ["--rollout-backend", "ray_vllm"])

    def test_ray_train_launcher_uses_train_fault_tolerance(self):
        source = inspect.getsource(launch_ray_train_module)
        self.assertIn("TorchTrainer", source)
        self.assertIn("FailureConfig", source)
        self.assertIn("CheckpointConfig", source)
        self.assertIn("ScalingConfig", source)
        self.assertIn("TorchConfig", source)
        self.assertIn("DITTY_RAY_TRAIN_DURABLE_ROOT", source)
        self.assertIn("_enable_ray_child_process_cleanup_env", source)

    def test_ray_launcher_enables_child_process_cleanup_before_ray_init(self):
        import ditty.ray_training as ray_training

        helper_source = inspect.getsource(ray_training._enable_ray_child_process_cleanup_env)
        self.assertIn("RAY_process_group_cleanup_enabled", helper_source)
        self.assertIn("RAY_kill_child_processes_on_worker_exit_with_raylet_subreaper", helper_source)
        self.assertIn(
            "_enable_ray_child_process_cleanup_env",
            inspect.getsource(ray_training.launch_ray_module),
        )

    def test_read_ditty_checkpoint_num(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            self.assertIsNone(_read_ditty_checkpoint_num(path))
            (path / "ditty_checkpoint_metadata.json").write_text(
                '{"checkpoint_num": 12}\n',
                encoding="utf-8",
            )
            self.assertEqual(_read_ditty_checkpoint_num(path), 12)

    def test_gcs_uri_from_checkpoint_uses_public_ray_fields(self):
        from types import SimpleNamespace

        checkpoint = SimpleNamespace(
            path="bucket/path/to/checkpoint",
            filesystem=SimpleNamespace(type_name="gcs"),
        )
        self.assertEqual(
            _gcs_uri_from_checkpoint(checkpoint),
            "gs://bucket/path/to/checkpoint",
        )
        checkpoint.path = "gs://bucket/path/to/checkpoint"
        self.assertEqual(
            _gcs_uri_from_checkpoint(checkpoint),
            "gs://bucket/path/to/checkpoint",
        )
        checkpoint.filesystem.type_name = "local"
        checkpoint.path = "/tmp/checkpoint"
        self.assertIsNone(_gcs_uri_from_checkpoint(checkpoint))

    def test_restore_gcs_ray_train_checkpoint_syncs_into_ditty_tree(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            remote = tmp_path / "remote"
            remote.mkdir()
            (remote / "ditty_checkpoint_metadata.json").write_text(
                '{"checkpoint_num": 7}\n',
                encoding="utf-8",
            )
            distributed = remote / "distributed"
            distributed.mkdir()
            (distributed / ".metadata").write_text("dcp metadata", encoding="utf-8")

            run_root = tmp_path / "run"
            checkpoint = SimpleNamespace(
                path="bucket/run/ditty_checkpoint_7",
                filesystem=SimpleNamespace(type_name="gcs"),
            )

            def fake_sync(uri, target):
                self.assertEqual(uri, "gs://bucket/run/ditty_checkpoint_7")
                import shutil

                shutil.copytree(remote, target, dirs_exist_ok=True)

            with mock.patch("ditty.ray_training._sync_gcs_checkpoint", side_effect=fake_sync):
                self.assertEqual(_restore_gcs_ray_train_checkpoint(checkpoint, run_root), 7)

            restored = run_root / "checkpoints" / "checkpoint_7"
            self.assertTrue((restored / "ditty_checkpoint_metadata.json").exists())
            self.assertTrue((restored / "distributed" / ".metadata").exists())
            self.assertFalse((run_root / "checkpoints" / ".ray_train_gcs_restore_tmp").exists())

    def test_restore_local_ray_train_pointer_syncs_durable_checkpoint(self):
        import tempfile
        from pathlib import Path
        from unittest import mock

        class FakeCheckpoint:
            def __init__(self, path):
                self.path = path

            def as_directory(self):
                class Context:
                    def __enter__(inner_self):
                        return self.path

                    def __exit__(inner_self, exc_type, exc, tb):
                        return False

                return Context()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pointer = tmp_path / "pointer"
            pointer.mkdir()
            (pointer / "ditty_checkpoint_metadata.json").write_text(
                '{"checkpoint_num": 8}\n',
                encoding="utf-8",
            )
            (pointer / "ditty_ray_checkpoint_pointer.json").write_text(
                '{"checkpoint_num": 8, "durable_uri": "gs://bucket/run/ditty_checkpoints/ditty_checkpoint_8"}\n',
                encoding="utf-8",
            )
            run_root = tmp_path / "run"

            with mock.patch(
                "ditty.checkpoint.CheckpointManager.restore_durable_checkpoint",
                return_value=8,
            ) as restore:
                self.assertEqual(_restore_local_ray_train_checkpoint(FakeCheckpoint(str(pointer)), run_root), 8)
            restore.assert_called_once_with(
                "gs://bucket/run/ditty_checkpoints/ditty_checkpoint_8",
                run_root,
            )

    def test_checkpoint_manager_writes_ray_train_metadata(self):
        import tempfile
        import json
        import os
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            manager = CheckpointManager(tmp)
            checkpoint_path = Path(manager.get_checkpoint_path(5))
            checkpoint_path.mkdir(parents=True)
            metadata_path = manager.write_ray_train_metadata(
                checkpoint_num=5,
                training_state={"steps": 4, "total_steps": 40},
                world_size=4,
            )
            metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
            self.assertEqual(metadata["checkpoint_num"], 5)
            self.assertEqual(metadata["training_state"]["total_steps"], 40)
            self.assertEqual(metadata["world_size"], 4)
            self.assertEqual(CheckpointManager.read_ray_train_checkpoint_num(str(checkpoint_path)), 5)

        old_value = os.environ.get("DITTY_RAY_TRAIN_REPORT_CHECKPOINTS")
        try:
            os.environ["DITTY_RAY_TRAIN_REPORT_CHECKPOINTS"] = "1"
            with tempfile.TemporaryDirectory() as tmp:
                manager = CheckpointManager(tmp)
                complete = Path(manager.get_checkpoint_path(1))
                partial = Path(manager.get_checkpoint_path(2))
                complete.mkdir(parents=True)
                partial.mkdir(parents=True)
                (complete / "training_state.pt").write_bytes(b"state")
                (partial / "training_state.pt").write_bytes(b"state")
                manager.write_ray_train_metadata(
                    checkpoint_num=1,
                    training_state={"steps": 1, "total_steps": 1},
                    world_size=4,
                )
                self.assertEqual(manager.get_latest_checkpoint_num(), 1)
        finally:
            if old_value is None:
                os.environ.pop("DITTY_RAY_TRAIN_REPORT_CHECKPOINTS", None)
            else:
                os.environ["DITTY_RAY_TRAIN_REPORT_CHECKPOINTS"] = old_value

    def test_checkpoint_manager_uses_ray_pointer_for_durable_gcs_dcp(self):
        import os
        import tempfile
        from pathlib import Path

        old_root = os.environ.get("DITTY_RAY_TRAIN_DURABLE_ROOT")
        old_mode = os.environ.get("DITTY_RAY_TRAIN_CHECKPOINT_MODE")
        try:
            os.environ["DITTY_RAY_TRAIN_DURABLE_ROOT"] = "gs://bucket/run/ditty_checkpoints"
            os.environ.pop("DITTY_RAY_TRAIN_CHECKPOINT_MODE", None)
            with tempfile.TemporaryDirectory() as tmp:
                manager = CheckpointManager(tmp)
                checkpoint_path = Path(manager.get_checkpoint_path(0))
                (checkpoint_path / "distributed").mkdir(parents=True)
                (checkpoint_path / "distributed" / "__0_0.distcp").write_bytes(b"shard")
                self.assertEqual(manager.ray_train_report_mode(0), "pointer")
        finally:
            if old_root is None:
                os.environ.pop("DITTY_RAY_TRAIN_DURABLE_ROOT", None)
            else:
                os.environ["DITTY_RAY_TRAIN_DURABLE_ROOT"] = old_root
            if old_mode is None:
                os.environ.pop("DITTY_RAY_TRAIN_CHECKPOINT_MODE", None)
            else:
                os.environ["DITTY_RAY_TRAIN_CHECKPOINT_MODE"] = old_mode

    def test_checkpoint_manager_uses_ray_pointer_for_direct_gcs_dcp(self):
        import os
        import tempfile
        from pathlib import Path

        old_root = os.environ.get("DITTY_RAY_TRAIN_DURABLE_ROOT")
        old_mode = os.environ.get("DITTY_RAY_TRAIN_CHECKPOINT_MODE")
        try:
            os.environ["DITTY_RAY_TRAIN_DURABLE_ROOT"] = "gs://bucket/run/ditty_checkpoints"
            os.environ.pop("DITTY_RAY_TRAIN_CHECKPOINT_MODE", None)
            with tempfile.TemporaryDirectory() as tmp:
                manager = CheckpointManager(tmp)
                checkpoint_path = Path(manager.get_checkpoint_path(0))
                checkpoint_path.mkdir(parents=True)
                manager._write_distributed_checkpoint_pointer(
                    str(checkpoint_path),
                    distributed_uri=(
                        "gs://bucket/run/ditty_checkpoints/ditty_checkpoint_0/distributed"
                    ),
                )
                self.assertEqual(manager.ray_train_report_mode(0), "pointer")
        finally:
            if old_root is None:
                os.environ.pop("DITTY_RAY_TRAIN_DURABLE_ROOT", None)
            else:
                os.environ["DITTY_RAY_TRAIN_DURABLE_ROOT"] = old_root
            if old_mode is None:
                os.environ.pop("DITTY_RAY_TRAIN_CHECKPOINT_MODE", None)
            else:
                os.environ["DITTY_RAY_TRAIN_CHECKPOINT_MODE"] = old_mode

    def test_checkpoint_manager_selects_direct_gcs_dcp_writer(self):
        import os
        import tempfile
        from unittest import mock

        old_root = os.environ.get("DITTY_RAY_TRAIN_DURABLE_ROOT")
        old_direct = os.environ.get("DITTY_RAY_TRAIN_DIRECT_DCP_TO_GCS")
        try:
            os.environ["DITTY_RAY_TRAIN_DURABLE_ROOT"] = "gs://bucket/run/ditty_checkpoints"
            os.environ["DITTY_RAY_TRAIN_DIRECT_DCP_TO_GCS"] = "required"
            with tempfile.TemporaryDirectory() as tmp:
                manager = CheckpointManager(tmp)
                writer = object()
                with mock.patch.object(manager, "_make_gcs_dcp_writer", return_value=writer):
                    uri, selected = manager._direct_dcp_writer_for_checkpoint(3)
                self.assertEqual(
                    uri,
                    "gs://bucket/run/ditty_checkpoints/ditty_checkpoint_3/distributed",
                )
                self.assertIs(selected, writer)
        finally:
            if old_root is None:
                os.environ.pop("DITTY_RAY_TRAIN_DURABLE_ROOT", None)
            else:
                os.environ["DITTY_RAY_TRAIN_DURABLE_ROOT"] = old_root
            if old_direct is None:
                os.environ.pop("DITTY_RAY_TRAIN_DIRECT_DCP_TO_GCS", None)
            else:
                os.environ["DITTY_RAY_TRAIN_DIRECT_DCP_TO_GCS"] = old_direct

    def test_checkpoint_manager_default_gcs_dcp_writer_is_parallel_composite(self):
        import os
        import tempfile
        from unittest import mock

        old_writer = os.environ.get("DITTY_RAY_TRAIN_DIRECT_DCP_GCS_WRITER")
        try:
            os.environ.pop("DITTY_RAY_TRAIN_DIRECT_DCP_GCS_WRITER", None)
            with tempfile.TemporaryDirectory() as tmp:
                manager = CheckpointManager(tmp)
                writer = object()
                with mock.patch("ditty.checkpoint._GcsParallelCompositeDcpWriter", return_value=writer) as ctor:
                    selected = manager._make_gcs_dcp_writer(
                        "gs://bucket/run/ditty_checkpoints/ditty_checkpoint_3/distributed"
                    )
                self.assertIs(selected, writer)
                ctor.assert_called_once_with(
                    "gs://bucket/run/ditty_checkpoints/ditty_checkpoint_3/distributed"
                )
        finally:
            if old_writer is None:
                os.environ.pop("DITTY_RAY_TRAIN_DIRECT_DCP_GCS_WRITER", None)
            else:
                os.environ["DITTY_RAY_TRAIN_DIRECT_DCP_GCS_WRITER"] = old_writer

    def test_checkpoint_manager_uploads_durable_manifest(self):
        import os
        import tempfile
        from pathlib import Path
        from unittest import mock

        old_root = os.environ.get("DITTY_RAY_TRAIN_DURABLE_ROOT")
        try:
            os.environ["DITTY_RAY_TRAIN_DURABLE_ROOT"] = "gs://bucket/run/ditty_checkpoints"
            with tempfile.TemporaryDirectory() as tmp:
                manager = CheckpointManager(tmp)
                checkpoint_path = Path(manager.get_checkpoint_path(0))
                distributed = checkpoint_path / "distributed"
                distributed.mkdir(parents=True)
                (checkpoint_path / "ditty_checkpoint_metadata.json").write_text(
                    '{"checkpoint_num": 0}\n',
                    encoding="utf-8",
                )
                (checkpoint_path / "training_state.pt").write_bytes(b"state")
                (checkpoint_path / "rng_state_0.pt").write_bytes(b"rng")
                (distributed / ".metadata").write_bytes(b"meta")
                (distributed / "__0_0.distcp").write_bytes(b"shard")
                expected_sizes = {
                    "gs://bucket/run/ditty_checkpoints/ditty_checkpoint_0/ditty_checkpoint_metadata.json": len('{"checkpoint_num": 0}\n'),
                    "gs://bucket/run/ditty_checkpoints/ditty_checkpoint_0/training_state.pt": 5,
                    "gs://bucket/run/ditty_checkpoints/ditty_checkpoint_0/rng_state_0.pt": 3,
                    "gs://bucket/run/ditty_checkpoints/ditty_checkpoint_0/distributed/.metadata": 4,
                    "gs://bucket/run/ditty_checkpoints/ditty_checkpoint_0/distributed/__0_0.distcp": 5,
                }
                copied: list[str] = []

                def fake_copy(source, destination_uri):
                    copied.append(destination_uri)

                def fake_size(uri):
                    return expected_sizes[uri]

                with mock.patch("ditty.checkpoint._gcs_copy_file", side_effect=fake_copy), mock.patch(
                    "ditty.checkpoint._gcs_object_size",
                    side_effect=fake_size,
                ):
                    uri = manager.upload_durable_checkpoint(
                        checkpoint_num=0,
                        training_state={"steps": 0, "total_steps": 0},
                        rank=0,
                        world_size=1,
                    )
                self.assertEqual(uri, "gs://bucket/run/ditty_checkpoints/ditty_checkpoint_0")
                self.assertIn(
                    "gs://bucket/run/ditty_checkpoints/ditty_checkpoint_0/ditty_durable_manifest.json",
                    copied,
                )
                self.assertIn(
                    "gs://bucket/run/ditty_checkpoints/ditty_checkpoint_0/.complete",
                    copied,
                )
        finally:
            if old_root is None:
                os.environ.pop("DITTY_RAY_TRAIN_DURABLE_ROOT", None)
            else:
                os.environ["DITTY_RAY_TRAIN_DURABLE_ROOT"] = old_root

    def test_checkpoint_manager_manifest_includes_direct_gcs_dcp_files(self):
        import json
        import os
        import tempfile
        from pathlib import Path
        from unittest import mock

        old_root = os.environ.get("DITTY_RAY_TRAIN_DURABLE_ROOT")
        try:
            os.environ["DITTY_RAY_TRAIN_DURABLE_ROOT"] = "gs://bucket/run/ditty_checkpoints"
            with tempfile.TemporaryDirectory() as tmp:
                manager = CheckpointManager(tmp)
                checkpoint_path = Path(manager.get_checkpoint_path(0))
                checkpoint_path.mkdir(parents=True)
                (checkpoint_path / "ditty_checkpoint_metadata.json").write_text(
                    '{"checkpoint_num": 0}\n',
                    encoding="utf-8",
                )
                (checkpoint_path / "training_state.pt").write_bytes(b"state")
                (checkpoint_path / "rng_state_0.pt").write_bytes(b"rng")
                manager._write_distributed_checkpoint_pointer(
                    str(checkpoint_path),
                    distributed_uri=(
                        "gs://bucket/run/ditty_checkpoints/ditty_checkpoint_0/distributed"
                    ),
                )
                copied_sizes: dict[str, int] = {}
                manifest: dict[str, object] = {}

                def fake_copy(source, destination_uri):
                    copied_sizes[destination_uri] = Path(source).stat().st_size
                    if destination_uri.endswith("/ditty_durable_manifest.json"):
                        manifest.update(json.loads(Path(source).read_text(encoding="utf-8")))

                def fake_size(uri):
                    return copied_sizes[uri]

                with mock.patch("ditty.checkpoint._gcs_copy_file", side_effect=fake_copy), mock.patch(
                    "ditty.checkpoint._gcs_object_size",
                    side_effect=fake_size,
                ), mock.patch(
                    "ditty.checkpoint._gcs_list_objects",
                    return_value=[
                        {
                            "uri": (
                                "gs://bucket/run/ditty_checkpoints/"
                                "ditty_checkpoint_0/distributed/.metadata"
                            ),
                            "bytes": 4,
                        },
                        {
                            "uri": (
                                "gs://bucket/run/ditty_checkpoints/"
                                "ditty_checkpoint_0/distributed/__0_0.distcp"
                            ),
                            "bytes": 5,
                        },
                    ],
                ):
                    uri = manager.upload_durable_checkpoint(
                        checkpoint_num=0,
                        training_state={"steps": 0, "total_steps": 0},
                        rank=0,
                        world_size=1,
                    )
                self.assertEqual(uri, "gs://bucket/run/ditty_checkpoints/ditty_checkpoint_0")
                manifest_paths = {str(item["path"]) for item in manifest["files"]}
                self.assertIn("distributed/.metadata", manifest_paths)
                self.assertIn("distributed/__0_0.distcp", manifest_paths)
                self.assertNotIn(
                    "gs://bucket/run/ditty_checkpoints/ditty_checkpoint_0/distributed/__0_0.distcp",
                    copied_sizes,
                )
        finally:
            if old_root is None:
                os.environ.pop("DITTY_RAY_TRAIN_DURABLE_ROOT", None)
            else:
                os.environ["DITTY_RAY_TRAIN_DURABLE_ROOT"] = old_root

    def test_checkpoint_manager_restores_direct_gcs_dcp_as_pointer(self):
        import json
        import shutil
        import tempfile
        from pathlib import Path
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            remote = tmp_path / "remote"
            remote.mkdir()
            (remote / "ditty_checkpoint_metadata.json").write_text(
                '{"checkpoint_num": 9}\n',
                encoding="utf-8",
            )
            (remote / "training_state.pt").write_bytes(b"state")
            (remote / "rng_state_0.pt").write_bytes(b"rng")
            (remote / "ditty_distributed_checkpoint.json").write_text(
                json.dumps(
                    {
                        "format": "ditty.distributed_checkpoint_pointer.v1",
                        "distributed_uri": "gs://bucket/run/ditty_checkpoint_9/distributed",
                    }
                ),
                encoding="utf-8",
            )
            manifest = {
                "format": "ditty.durable_checkpoint_manifest.v1",
                "checkpoint_num": 9,
                "training_state": {"steps": 4, "total_steps": 44},
                "world_size": 4,
                "complete": True,
                "files": [
                    {"path": "distributed/.metadata", "bytes": 4, "rank": "direct_gcs"},
                    {"path": "distributed/__0_0.distcp", "bytes": 5, "rank": "direct_gcs"},
                    {
                        "path": "ditty_checkpoint_metadata.json",
                        "bytes": (remote / "ditty_checkpoint_metadata.json").stat().st_size,
                        "rank": 0,
                    },
                    {"path": "training_state.pt", "bytes": (remote / "training_state.pt").stat().st_size, "rank": 0},
                    {"path": "rng_state_0.pt", "bytes": (remote / "rng_state_0.pt").stat().st_size, "rank": 0},
                    {
                        "path": "ditty_distributed_checkpoint.json",
                        "bytes": (remote / "ditty_distributed_checkpoint.json").stat().st_size,
                        "rank": 0,
                    },
                ],
            }
            (remote / "ditty_durable_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            run_root = tmp_path / "run"

            def fake_download(source_uri, target):
                rel = source_uri.replace("gs://bucket/run/ditty_checkpoint_9/", "")
                shutil.copyfile(remote / rel, target)

            with mock.patch("ditty.checkpoint._gcs_download_file", side_effect=fake_download), mock.patch(
                "ditty.checkpoint._gcs_rsync",
            ) as rsync:
                checkpoint_num = CheckpointManager.restore_durable_checkpoint(
                    "gs://bucket/run/ditty_checkpoint_9",
                    run_root,
                )

            self.assertEqual(checkpoint_num, 9)
            rsync.assert_not_called()
            restored = run_root / "checkpoints" / "checkpoint_9"
            self.assertTrue((restored / "ditty_checkpoint_metadata.json").exists())
            self.assertTrue((restored / "training_state.pt").exists())
            self.assertEqual(
                CheckpointManager.read_distributed_checkpoint_pointer(str(restored)),
                "gs://bucket/run/ditty_checkpoint_9/distributed",
            )
            self.assertFalse((restored / "distributed" / "__0_0.distcp").exists())

    def test_checkpoint_manager_can_delete_local_after_ray_upload(self):
        source = inspect.getsource(CheckpointManager.report_to_ray_train)
        self.assertIn("DITTY_RAY_TRAIN_DELETE_LOCAL_AFTER_UPLOAD", source)
        self.assertIn("delete_after_upload if mode == \"full\" else True", source)

    def test_checkpoint_manager_prunes_old_complete_checkpoints(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            manager = CheckpointManager(tmp)
            for num in range(5):
                checkpoint_path = Path(manager.get_checkpoint_path(num))
                checkpoint_path.mkdir(parents=True)
                (checkpoint_path / "training_state.pt").write_bytes(b"state")
            removed = manager.prune_old_checkpoints(keep_last=2)
            self.assertEqual(removed, [0, 1, 2])
            self.assertFalse(Path(manager.get_checkpoint_path(2)).exists())
            self.assertTrue(Path(manager.get_checkpoint_path(3)).exists())
            self.assertTrue(Path(manager.get_checkpoint_path(4)).exists())


if __name__ == "__main__":
    unittest.main()
