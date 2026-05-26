import unittest

import torch
from torch import nn

from ditty.model_factory import _patch_float8_unsafe_causal_mask_builders


class DummyNemotronMaskModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_dtype = None
        self.seen_shape = None
        self.seen_device = None

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
        del attention_mask, cache_position
        self.seen_dtype = input_tensor.dtype
        self.seen_shape = tuple(input_tensor.shape)
        self.seen_device = input_tensor.device
        return torch.empty((input_tensor.shape[0],), device=input_tensor.device, dtype=input_tensor.dtype)


DummyNemotronMaskModule.__module__ = "transformers_modules.nemotron_h.modeling_nemotron_h"


class DummyOtherMaskModule(DummyNemotronMaskModule):
    pass


DummyOtherMaskModule.__module__ = "other_modeling"


class Float8ModelFactoryTests(unittest.TestCase):
    def test_nemotron_causal_mask_builder_uses_non_float8_shape_proxy(self):
        module = DummyNemotronMaskModule()
        patched = _patch_float8_unsafe_causal_mask_builders(module)

        self.assertEqual(patched, 1)
        fp8_input = torch.empty((2, 4, 8), dtype=torch.float8_e4m3fn)
        mask = module._update_causal_mask(None, fp8_input, None)

        self.assertEqual(module.seen_dtype, torch.float32)
        self.assertEqual(module.seen_shape, (2, 4, 8))
        self.assertEqual(module.seen_device, fp8_input.device)
        self.assertEqual(mask.dtype, torch.float32)

    def test_non_nemotron_causal_mask_builder_is_left_unpatched(self):
        module = DummyOtherMaskModule()
        patched = _patch_float8_unsafe_causal_mask_builders(module)

        self.assertEqual(patched, 0)
        fp8_input = torch.empty((1, 2, 3), dtype=torch.float8_e4m3fn)
        mask = module._update_causal_mask(None, fp8_input, None)

        self.assertEqual(module.seen_dtype, torch.float8_e4m3fn)
        self.assertEqual(mask.dtype, torch.float8_e4m3fn)


if __name__ == "__main__":
    unittest.main()
