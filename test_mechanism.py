import copy
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from cell_demo import (CELLHeads, COEFFICIENTS, PRESETS, cell_distribution,
                       train_step, training_prior, window_inputs)


class MechanismTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randn(11, 4, requires_grad=True)
        self.y = torch.arange(11) % 3
        self.inputs = window_inputs(self.x, 7)
        self.model = CELLHeads(4, 3, training_prior(self.y, 3))

    def test_window_content_and_detachment(self):
        x = torch.arange(7.).reshape(7, 1).requires_grad_()
        values = window_inputs(x, 7)
        self.assertEqual(values["h"][3, :, 0].tolist(), [0, 1, 2, 4, 5, 6])
        self.assertEqual(values["z"][3, :, 0].tolist(), [0, 1, 5, 6])
        self.assertEqual(values["as1"][3].tolist(), [2, 3])
        self.assertTrue(all(not v.requires_grad for v in values.values()))

    def test_fusion_against_probability_product(self):
        logits = self.model(self.inputs, "full_cell")
        p = {k: v.softmax(-1) for k, v in logits.items()}
        expected = self.model.prior * p["h"] * p["as1"] * p["as2"] / (p["a"] * p["s1"] * p["s2"])
        expected = expected / expected.sum(-1, keepdim=True)
        torch.testing.assert_close(cell_distribution(logits, self.model.prior), expected)
        for name in COEFFICIENTS:
            shifted = dict(logits)
            shifted[name] = logits[name] + torch.randn(len(self.y), 1) * 10
            torch.testing.assert_close(cell_distribution(shifted, self.model.prior), expected)

    def test_soft_loss_updates_only_remote_head(self):
        logits = self.model(self.inputs, "full_cell")
        self.model.losses(logits, self.y)["z"].backward()
        for name, head in self.model.heads.items():
            self.assertTrue(all((p.grad is not None) == (name == "z") for p in head.parameters()))
        self.assertIsNone(self.x.grad)

    def test_source_losses_train_all_source_heads(self):
        logits = self.model(self.inputs, "full_cell")
        losses = self.model.losses(logits, self.y)
        sum(v for k, v in losses.items() if k != "z").backward()
        for name in COEFFICIENTS:
            self.assertTrue(all(p.grad is not None for p in self.model.heads[name].parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.heads["z"].parameters()))

    def test_disabled_heads_and_isolated_backbone_update(self):
        for preset in PRESETS:
            heads = copy.deepcopy(self.model)
            backbone = nn.Linear(4, 3)
            reference = copy.deepcopy(backbone)
            reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
            reference_optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(reference(self.x.detach()), self.y).backward()
            nn.utils.clip_grad_norm_(reference.parameters(), 10.0)
            reference_optimizer.step()
            before = {k: p.detach().clone() for k, p in heads.named_parameters()}
            train_step(backbone, heads, self.inputs, self.x, self.y, preset,
                       [torch.optim.SGD(backbone.parameters(), lr=0.01),
                        torch.optim.SGD(heads.parameters(), lr=0.01)])
            for actual, expected in zip(backbone.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for name, parameter in heads.named_parameters():
                if name.split(".")[1] not in PRESETS[preset]:
                    torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
            active_before = heads(self.inputs, preset)
            with torch.no_grad():
                for name, head in heads.heads.items():
                    if name not in PRESETS[preset]:
                        for parameter in head.parameters():
                            parameter.add_(100)
            for name, actual in heads(self.inputs, preset).items():
                torch.testing.assert_close(actual, active_before[name], rtol=0, atol=0)

    def test_reject_missing_sources_invalid_prior_and_window(self):
        with self.assertRaises(ValueError):
            cell_distribution({}, self.model.prior)
        with self.assertRaises(ValueError):
            cell_distribution(self.model(self.inputs, "full_cell"), torch.tensor([0., 1., 1.]))
        with self.assertRaises(ValueError):
            window_inputs(self.x, 4)


if __name__ == "__main__":
    unittest.main()
