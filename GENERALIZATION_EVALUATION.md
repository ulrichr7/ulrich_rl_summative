# Smart Irrigation RL: Generalization Evaluation

This evaluation tests the selected best checkpoint for each method on **50 new shared seeds** (20000–20049 by default). The seeds are outside the training range (7–126) and the first evaluation range (10007–10016).

## Install into the project

From the project root, extract this package so that `analysis/generalization_evaluation.py` is inside the existing `analysis/` directory. Do not replace the rest of the project.

## Run

```bash
uv run python -m analysis.generalization_evaluation
```

Optional: evaluate 100 fresh seeds instead of 50:

```bash
uv run python -m analysis.generalization_evaluation --episodes 100 --seed-start 20000
```

The four model paths default to the best checkpoints already selected in the project. The script loads the architecture from each training module, validates the checkpoint input/output dimensions, and uses greedy deterministic action selection, matching the existing evaluation convention.

## Outputs

The script writes these files into `analysis/outputs/`:

- `generalization_episode_results.csv` — one row per algorithm and seed, including reward, yield, health, irrigation, stress, and starting conditions.
- `generalization_summary.csv` — mean, sample standard deviation, 95% confidence intervals, completion rate, and action counts.
- `generalization_paired_comparisons.csv` — paired per-seed reward differences for DQN versus each other method.
- `generalization_protocol.json` — seeds, checkpoint names, evaluation protocol, and caveats.

All four agents use the same seeds, so each method sees the same initial soil moisture, reservoir level, and weather sequence for a given seed. Policies choose `argmax` actions; they do not sample actions during evaluation.

## Important interpretation

This test is an additional held-out evaluation of the **simulator**, not evidence of field performance in real farms. Confidence intervals are descriptive normal-approximation intervals; avoid claiming statistical significance solely from overlapping or non-overlapping intervals. Discuss paired differences alongside the mean and standard deviation. A2C and other methods should be compared on reward and agronomic outcomes together, not reward alone.
