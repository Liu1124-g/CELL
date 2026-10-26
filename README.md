# CELL: ablation mechanism demo

A small CPU example of the conditional classifiers, structured soft target, and component switches used in CELL. It runs on synthetic frame features and exposes the training computation directly. The printed losses and probabilities are demonstration outputs, not dataset benchmark results.

## Run

Use Python 3.10 or newer with PyTorch:

```bash
python -m pip install -r requirements.txt
python cell_demo.py --preset all --steps 3
python cell_demo.py --preset full_cell --mode joint
python -m unittest -v
```

No videos, pretrained checkpoints, GPU, or external data downloads are required. Each preset starts from the same seed. The JSON output lists the active heads, individual losses, and class probabilities for one centre time. Full CELL also reports its fused teacher distribution and normalization error.

## Window and head definitions

For a window centred at time `t`, `a` is the current observation, `s1` and `s2` are its immediate neighbours, and `z` contains the remaining remote-context observations in that window. `h` denotes the combined input `(z,s1,s2)`. Each window uses the action label of its own centre frame. Each classifier shares its parameters across all window positions.

The demo uses an odd window of at least five frames, with replicated edge padding. Features are detached before window construction. Pair inputs are concatenated. Context heads classify each selected position and average the resulting class logits before softmax.

| Demo head | Existing implementation name | Input | Supervision |
|---|---|---|---|
| `a` | `pseudo_a` | Current observation | Centre label |
| `s1` | `pseudo_a1` | Left neighbour | Centre label |
| `s2` | `pseudo_a2` | Right neighbour | Centre label |
| `as1` | `pseudo_s2` | `(s1,a)` | Centre label |
| `as2` | `pseudo_s1` | `(a,s2)` | Centre label |
| `h` | `pseudo_main` | All window positions except `a` | Centre label |
| `z` | `pseudo_z` | Window positions excluding `(s1,a,s2)` | Detached fused soft target |

The existing pair-head names refer to the omitted neighbour: `pseudo_s2` therefore receives `(s1,a)`.

## Structured fusion and supervision

For class probability vectors `p`, the teacher is

```text
r = log(p_h) + log(p_as1) + log(p_as2)
    - log(p_a) - log(p_s1) - log(p_s2) + log(prior)
q = softmax(r)
```

The fixed prior is estimated from training-frame class counts with additive smoothing. It is stored as a model buffer. The remote student `p_z` is excluded from teacher construction.

```text
source_loss = sum CE(source_head_logits, centre_label)
soft_loss   = -mean sum stop_gradient(q) * log(p_z)
head_loss   = source_loss + soft_loss
```

The source heads learn through their own label losses. Detaching `q` blocks the soft loss from updating those heads through the target construction; the remote head learns from that soft loss. As the centre moves, the same observation participates in different windows with their respective centre labels.

## Six component settings

| Setting | Enabled heads | Remote soft supervision |
|---|---|---|
| Unary only | `a` | Off |
| Unary + left | `a, as1` | Off |
| Unary + right | `a, as2` | Off |
| Local only | `a, as1, as2` | Off |
| Local + remote | `a, as1, as2, h` | Off |
| Full CELL | All seven heads | On |

Disabled heads are omitted from forward computation and losses. Only Full CELL constructs the complete structured teacher. Reduced settings expose their enabled branch probabilities; they do not substitute an arbitrary partial formula for the full teacher. These switches demonstrate component participation in supervision.

## Parameter updates and output objects

The small linear task classifier stands in for a backbone so that optimization routes can be inspected. It and the CELL heads read fixed input features and have disjoint parameters.

- `--mode joint`: sum the task and weighted auxiliary losses, backpropagate once, clip all parameters together, and take one optimizer step.
- `--mode isolated`: use separate optimizers and gradient clipping for the task classifier and heads. This mode demonstrates a controlled update in which the auxiliary losses do not change the task-classifier update.

The demo explicitly distinguishes a fused teacher, a remote-head prediction, and a task-classifier prediction. It does not report synthetic segmentation accuracy or treat these outputs as interchangeable. Integrating a dataset evaluator requires specifying which output supplies the frame labels. The full training pipeline, dataset loaders, temporal backbone, and benchmark evaluation are outside this compact mechanism demo.

## Checks

Verified on CPU with Python 3.10.13 and PyTorch 2.6.0: all six tests passed; all six presets ran in both optimization modes. The maximum teacher normalization error in these smoke runs was approximately `1.2e-7`.

`test_mechanism.py` checks:

1. Window membership, pair ordering, and detached feature inputs.
2. Agreement between log-space fusion and the normalized probability product, including invariance to class-independent logit shifts.
3. Gradient separation between the soft loss and source classifiers.
4. Source-head learning from centre-label supervision.
5. Exact one-step task-classifier update equality in isolated mode, unchanged disabled parameters, and invariance to disabled-head perturbations.
6. Rejection of incomplete fusion sources, invalid priors, and invalid window widths.

## Availability

This repository contains selected demonstration code. The complete implementation and selected experimental results are planned for release after the review process.
