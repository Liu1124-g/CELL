"""Executable CELL mechanism demo on synthetic features (CPU)."""
import argparse
import json

import torch
from torch import nn
from torch.nn import functional as F


# Public names describe inputs; the README maps these to implementation names.
COEFFICIENTS = {"h": 1, "s1": -1, "s2": -1, "as1": 1, "as2": 1, "a": -1}
PRESETS = {
    "unary_only": ("a",),
    "unary_left": ("a", "as1"),
    "unary_right": ("a", "as2"),
    "local_only": ("a", "as1", "as2"),
    "local_remote": ("a", "as1", "as2", "h"),
    "full_cell": tuple(COEFFICIENTS) + ("z",),
}


def window_inputs(features, width):
    """[T,D] -> per-centre inputs; replicate padding is explicit demo policy."""
    if features.ndim != 2 or len(features) == 0:
        raise ValueError("Expected a nonempty [T,D] feature tensor")
    if width < 5 or width % 2 == 0:
        raise ValueError("Window width must be odd and >= 5")
    x = features.detach()
    mid = width // 2
    indices = torch.arange(len(x), device=x.device)[:, None]
    offsets = torch.arange(-mid, mid + 1, device=x.device)[None, :]
    windows = x[(indices + offsets).clamp(0, len(x) - 1)]
    a, s1, s2 = windows[:, mid], windows[:, mid - 1], windows[:, mid + 1]
    h_indices = [j for j in range(width) if j != mid]
    z_indices = [j for j in range(width) if j not in (mid - 1, mid, mid + 1)]
    return {"a": a, "s1": s1, "s2": s2,
            "as1": torch.cat((s1, a), -1), "as2": torch.cat((a, s2), -1),
            "h": windows[:, h_indices], "z": windows[:, z_indices]}


def training_prior(labels, classes, alpha=1.0):
    """Smoothed frame-frequency prior, computed from training labels only."""
    if alpha <= 0 or not torch.isfinite(torch.tensor(alpha)):
        raise ValueError("Demo smoothing alpha must be finite and positive")
    counts = torch.bincount(labels.reshape(-1), minlength=classes).float()
    if len(counts) != classes:
        raise ValueError("Label outside the class range")
    return (counts + alpha) / (counts.sum() + alpha * classes)


def cell_distribution(logits, prior):
    """Normalized structured combination; z is the student, never a source."""
    missing = set(COEFFICIENTS) - set(logits)
    if missing:
        raise ValueError(f"Full CELL requires source heads: {sorted(missing)}")
    reference = logits["h"]
    prior = torch.as_tensor(prior, device=reference.device, dtype=torch.float32)
    if prior.shape != (reference.shape[-1],) or not torch.isfinite(prior).all() or (prior <= 0).any():
        raise ValueError("Expected a finite, strictly positive class prior")
    score = (prior / prior.sum()).log()
    for name, coefficient in COEFFICIENTS.items():
        if logits[name].shape != reference.shape:
            raise ValueError("All source logit tensors must have the same shape")
        score = score + coefficient * F.log_softmax(logits[name].float(), -1)
    return score.softmax(-1)


class CELLHeads(nn.Module):
    def __init__(self, dim, classes, prior, hidden=16):
        super().__init__()
        self.heads = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(dim * (2 if name in ("as1", "as2") else 1), hidden),
                                nn.ReLU(), nn.Linear(hidden, classes))
            for name in (*COEFFICIENTS, "z")
        })
        self.register_buffer("prior", prior.detach().clone())

    def forward(self, inputs, preset):
        output = {}
        for name in PRESETS[preset]:
            values = self.heads[name](inputs[name])
            # Context heads classify positions, then average their class logits.
            output[name] = values.mean(1) if name in ("h", "z") else values
        return output

    def losses(self, logits, labels):
        terms = {name: F.cross_entropy(value, labels)
                 for name, value in logits.items() if name != "z"}
        if "z" in logits:
            teacher = cell_distribution(logits, self.prior).detach()
            terms["z"] = -(teacher * F.log_softmax(logits["z"], -1)).sum(-1).mean()
        return terms


def train_step(backbone, heads, inputs, features, labels, preset, optimizers,
               mode="isolated", weight=0.1):
    """Joint training or an explicitly isolated mechanism-control update."""
    expected = 2 if mode == "isolated" else 1
    if mode not in ("isolated", "joint") or len(optimizers) != expected:
        raise ValueError("Use one optimizer for joint, two for isolated mode")
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    logits = heads(inputs, preset)
    terms = heads.losses(logits, labels)
    auxiliary = torch.stack(list(terms.values())).sum()
    task = F.cross_entropy(backbone(features.detach()), labels)
    if mode == "joint":
        (task + weight * auxiliary).backward()
        nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(heads.parameters()), 10.0)
    else:
        task.backward()
        (weight * auxiliary).backward()
        nn.utils.clip_grad_norm_(backbone.parameters(), 10.0)
        nn.utils.clip_grad_norm_(heads.parameters(), 10.0)
    for optimizer in optimizers:
        optimizer.step()
    return {"task_loss": float(task.detach()),
            "head_losses": {k: float(v.detach()) for k, v in terms.items()}}


def run(preset, steps, seed, width, mode):
    torch.manual_seed(seed)
    features = torch.randn(32, 8)
    labels = torch.arange(32).div(8, rounding_mode="floor") % 3
    inputs = window_inputs(features, width)
    heads = CELLHeads(8, 3, training_prior(labels, 3))
    backbone = nn.Linear(8, 3)  # Small task classifier used to expose update routes.
    optimizers = ([torch.optim.SGD(backbone.parameters(), lr=0.01),
                   torch.optim.SGD(heads.parameters(), lr=0.01)] if mode == "isolated"
                  else [torch.optim.SGD(list(backbone.parameters()) + list(heads.parameters()), lr=0.01)])
    for _ in range(steps):
        result = train_step(backbone, heads, inputs, features, labels, preset, optimizers, mode)
    heads.eval()
    with torch.no_grad():
        logits = heads(inputs, preset)
        result.update(preset=preset, mode=mode, seed=seed, window=width,
                      active_heads=list(logits), tensor_shape=list(features.shape),
                      branch_probabilities_at_t16={k: v[16].softmax(-1).tolist() for k, v in logits.items()})
        if preset == "full_cell":
            q = cell_distribution(logits, heads.prior)
            result["teacher_at_t16"] = q[16].tolist()
            result["teacher_max_normalization_error"] = float((q.sum(-1) - 1).abs().max())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=[*PRESETS, "all"], default="all")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=7)
    parser.add_argument("--mode", choices=["isolated", "joint"], default="isolated")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    presets = PRESETS if args.preset == "all" else [args.preset]
    print(json.dumps([run(p, args.steps, args.seed, args.window, args.mode) for p in presets], indent=2))
