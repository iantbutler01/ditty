import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import torch
import torch.nn as nn
from safetensors.torch import save_file

from ditty.model_factory import ModelFactory
from ditty.models.step3p5 import (
    Step3p5ForMTPTraining,
    _step3p5_config_from_payload,
    _step3p5_mtp_config_view,
    load_step3p5_mtp_weights,
    rewrite_step3p5_mtp_weight_name,
    summarize_step3p5_mtp_checkpoint_keys,
)


class Scale(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        del eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        return x * self.weight


class FakeBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.attention_type = "full_attention"

    def forward(self, hidden_states, **kwargs):
        del kwargs
        return self.proj(hidden_states)


class FakeDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([FakeBlock(config, 0)])
        self.norm = Scale(config.hidden_size)

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        del kwargs
        hidden = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        return SimpleNamespace(last_hidden_state=self.norm(hidden))


class FakeBaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = FakeDecoder(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


class SharedHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.norm = Scale(hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size, bias=False)


class MinimalMTPLayer(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.enorm = Scale(hidden_size)
        self.hnorm = Scale(hidden_size)
        self.eh_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.shared_head = SharedHead(hidden_size, vocab_size)
        self.mtp_block = nn.Module()
        self.mtp_block.self_attn = nn.Module()
        self.mtp_block.self_attn.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)


class MinimalMTPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=45,
            num_nextn_predict_layers=1,
            hidden_size=3,
            vocab_size=5,
        )
        self.mtp_embed_tokens = nn.Embedding(5, 3)
        self.mtp_layers = nn.ModuleDict({"45": MinimalMTPLayer(3, 5)})


class ValidatingStep3p5Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("invalid layer_types length")


class Step3p5MTPAdapterTests(unittest.TestCase):
    def test_config_loader_trims_base_layer_types_and_preserves_mtp_view(self):
        payload = {
            "num_hidden_layers": 45,
            "num_nextn_predict_layers": 3,
            "layer_types": ["full_attention"] * 45 + ["sliding_attention"] * 3,
        }

        config = _step3p5_config_from_payload(payload, ValidatingStep3p5Config)
        mtp_config = _step3p5_mtp_config_view(config)

        self.assertEqual(len(config.layer_types), 45)
        self.assertEqual(len(mtp_config.layer_types), 48)
        self.assertEqual(mtp_config.layer_types[45], "sliding_attention")

    def test_model_factory_uses_normal_custom_model_class_path(self):
        sentinel = nn.Linear(1, 1)
        with mock.patch.object(
            Step3p5ForMTPTraining,
            "from_pretrained",
            return_value=sentinel,
        ) as from_pretrained:
            factory = ModelFactory.from_huggingface(
                {
                    "model_path": "stepfun-ai/Step-3.5-Flash",
                    "model_class_name": "ditty.models.step3p5.Step3p5ForMTPTraining",
                },
                trust_remote_code=True,
                load_mtp_weights=False,
            )
            model = factory.build()

        self.assertIs(model, sentinel)
        from_pretrained.assert_called_once()
        _, kwargs = from_pretrained.call_args
        self.assertEqual(from_pretrained.call_args.args[0], "stepfun-ai/Step-3.5-Flash")
        self.assertFalse(kwargs["load_mtp_weights"])
        self.assertTrue(kwargs["trust_remote_code"])

    def test_from_pretrained_injects_step3p5_config_before_auto_model_load(self):
        config = SimpleNamespace(
            model_type="step3p5",
            num_hidden_layers=45,
            num_nextn_predict_layers=3,
            hidden_size=4,
            vocab_size=9,
            rms_norm_eps=1e-5,
            pad_token_id=None,
            layer_types=["full_attention"] * 45,
            _ditty_full_layer_types=tuple(["full_attention"] * 45 + ["sliding_attention"] * 3),
        )

        def fake_from_pretrained(model_path, *args, **kwargs):
            del model_path, args
            self.assertIs(kwargs["config"], config)
            return FakeBaseModel(kwargs["config"])

        with mock.patch(
            "ditty.models.step3p5._load_step3p5_config",
            return_value=config,
        ) as load_config, mock.patch(
            "ditty.models.step3p5.AutoModelForCausalLM.from_pretrained",
            side_effect=fake_from_pretrained,
        ):
            model = Step3p5ForMTPTraining.from_pretrained(
                "stepfun-ai/Step-3.5-Flash",
                load_mtp_weights=False,
                trust_remote_code=True,
            )

        load_config.assert_called_once()
        self.assertIs(model.config, config)
        self.assertEqual(model.mtp_layers["45"].mtp_block.layer_idx, 45)

    def test_weight_name_rewrite_matches_stepfun_mtp_shapes(self):
        config = SimpleNamespace(num_hidden_layers=45, num_nextn_predict_layers=3)

        self.assertEqual(
            rewrite_step3p5_mtp_weight_name(config, "model.embed_tokens.weight"),
            "mtp_embed_tokens.weight",
        )
        self.assertEqual(
            rewrite_step3p5_mtp_weight_name(config, "model.layers.45.enorm.weight"),
            "mtp_layers.45.enorm.weight",
        )
        self.assertEqual(
            rewrite_step3p5_mtp_weight_name(config, "model.layers.46.self_attn.q_proj.weight"),
            "mtp_layers.46.mtp_block.self_attn.q_proj.weight",
        )
        self.assertEqual(
            rewrite_step3p5_mtp_weight_name(
                config,
                "model.layers.47.transformer.shared_head.output.weight",
            ),
            "mtp_layers.47.shared_head.output.weight",
        )
        self.assertIsNone(
            rewrite_step3p5_mtp_weight_name(config, "model.layers.44.self_attn.q_proj.weight")
        )

    def test_loader_hard_fails_when_mtp_params_are_missing(self):
        model = MinimalMTPModel()
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/model.safetensors"
            save_file({}, path)
            with self.assertRaisesRegex(RuntimeError, "Missing Step3p5 MTP checkpoint parameters"):
                load_step3p5_mtp_weights(model, tmp, strict=True)

    def test_loader_rewrites_and_loads_all_mtp_params(self):
        model = MinimalMTPModel()
        tensors = {
            "model.embed_tokens.weight": torch.full_like(model.mtp_embed_tokens.weight, 1.0),
            "model.layers.45.enorm.weight": torch.full_like(
                model.mtp_layers["45"].enorm.weight, 2.0
            ),
            "model.layers.45.hnorm.weight": torch.full_like(
                model.mtp_layers["45"].hnorm.weight, 3.0
            ),
            "model.layers.45.eh_proj.weight": torch.full_like(
                model.mtp_layers["45"].eh_proj.weight, 4.0
            ),
            "model.layers.45.transformer.shared_head.norm.weight": torch.full_like(
                model.mtp_layers["45"].shared_head.norm.weight, 5.0
            ),
            "model.layers.45.transformer.shared_head.output.weight": torch.full_like(
                model.mtp_layers["45"].shared_head.output.weight, 6.0
            ),
            "model.layers.45.self_attn.q_proj.weight": torch.full_like(
                model.mtp_layers["45"].mtp_block.self_attn.q_proj.weight, 7.0
            ),
        }

        with tempfile.TemporaryDirectory() as tmp:
            save_file(tensors, f"{tmp}/model.safetensors")
            report = load_step3p5_mtp_weights(model, tmp, strict=True)

        self.assertFalse(report.missing)
        self.assertFalse(report.unexpected)
        self.assertTrue(torch.equal(model.mtp_embed_tokens.weight, tensors["model.embed_tokens.weight"]))
        self.assertTrue(torch.equal(model.mtp_layers["45"].enorm.weight, tensors["model.layers.45.enorm.weight"]))
        self.assertTrue(
            torch.equal(
                model.mtp_layers["45"].shared_head.output.weight,
                tensors["model.layers.45.transformer.shared_head.output.weight"],
            )
        )

    def test_metadata_summary_finds_mtp_layers_without_model_load(self):
        config = SimpleNamespace(num_hidden_layers=45, num_nextn_predict_layers=3)
        tensors = {
            "model.embed_tokens.weight": torch.zeros((2, 2)),
            "model.layers.45.enorm.weight": torch.zeros((2,)),
            "model.layers.45.hnorm.weight": torch.zeros((2,)),
            "model.layers.45.eh_proj.weight": torch.zeros((2, 4)),
            "model.layers.45.transformer.shared_head.output.weight": torch.zeros((3, 2)),
            "model.layers.45.self_attn.q_proj.weight": torch.zeros((2, 2)),
            "model.layers.46.enorm.weight": torch.zeros((2,)),
            "model.layers.46.self_attn.q_proj.weight": torch.zeros((2, 2)),
            "model.layers.47.hnorm.weight": torch.zeros((2,)),
            "model.layers.47.transformer.shared_head.norm.weight": torch.zeros((2,)),
            "model.layers.44.self_attn.q_proj.weight": torch.zeros((2, 2)),
        }
        with tempfile.TemporaryDirectory() as tmp:
            save_file(tensors, f"{tmp}/model.safetensors")
            summary = summarize_step3p5_mtp_checkpoint_keys(tmp, config)

        self.assertEqual(summary.layers, (45, 46, 47))
        self.assertTrue(summary.has_embed_tokens)
        self.assertIn("mtp_block", summary.key_classes_by_layer[45])
        self.assertIn("shared_head", summary.key_classes_by_layer[47])

    def test_clone_mtp_step_copies_parameters_exactly(self):
        config = SimpleNamespace(
            num_hidden_layers=45,
            num_nextn_predict_layers=3,
            hidden_size=4,
            vocab_size=9,
            rms_norm_eps=1e-5,
            pad_token_id=None,
        )
        model = Step3p5ForMTPTraining(FakeBaseModel(config))
        with torch.no_grad():
            for param in model.mtp_layers["45"].parameters():
                param.fill_(0.25)

        model.clone_mtp_step(src_step=0, dst_steps=[1, 2], overwrite=True)

        src_state = model.mtp_layers["45"].state_dict()
        for layer_key in ("46", "47"):
            dst_state = model.mtp_layers[layer_key].state_dict()
            self.assertEqual(set(src_state), set(dst_state))
            for name, src_value in src_state.items():
                self.assertTrue(torch.equal(src_value, dst_state[name]))


if __name__ == "__main__":
    unittest.main()
