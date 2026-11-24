"""Evaluate the four selected smart-irrigation RL checkpoints on shared unseen seeds.

Run from the project root:
    uv run python -m analysis.generalization_evaluation

By default, this evaluates 50 fresh seeds (20000--20049) using the same
seed list for DQN, REINFORCE, PPO, and A2C. Actions are greedy/deterministic,
matching the evaluation convention used by the training scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from environment.custom_env import SmartIrrigationEnv
from training.a2c_training import build_a2c_networks
from training.dqn_training import build_dqn_policy
from training.ppo_training import build_ppo_networks
from training.reinforce_training import build_policy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINTS = {
    "DQN": "models/dqn/dqn_experiment_04_gamma_095.pt",
    "REINFORCE": "models/pg/reinforce_experiment_01_default.pt",
    "PPO": "models/pg/ppo_experiment_01_default.pt",
    "A2C": "models/pg/a2c_experiment_03_lr_0006.pt",
}
ACTION_LABELS = {
    0: "none_0mm",
    1: "light_2mm",
    2: "moderate_5mm",
    3: "substantial_8mm",
    4: "heavy_12mm",
}


def _torch_load(path: Path) -> dict[str, Any]:
    """Load this project's trusted checkpoint across supported PyTorch versions."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # compatibility with older PyTorch releases
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dictionary in {path}")
    return checkpoint


def load_agent(algorithm: str, checkpoint_path: Path) -> Callable[[np.ndarray], int]:
    checkpoint = _torch_load(checkpoint_path)
    observation_dim = int(checkpoint.get("observation_dim", -1))
    action_count = int(checkpoint.get("action_count", -1))
    config = checkpoint.get("config", {})
    hidden_dim = int(config.get("hidden_dim", 128))

    if observation_dim != 10 or action_count != 5:
        raise ValueError(
            f"{algorithm} checkpoint expects observation_dim={observation_dim}, "
            f"action_count={action_count}; this environment expects 10 and 5."
        )

    if algorithm == "DQN":
        q_network, _actor = build_dqn_policy(observation_dim, action_count, hidden_dim)
        q_network.module.load_state_dict(checkpoint["model_state_dict"])
        model = q_network.module
    elif algorithm == "REINFORCE":
        model, _logits_module, _actor = build_policy(
            observation_dim, action_count, hidden_dim
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    elif algorithm == "PPO":
        model, _actor_module, _actor, _critic_net, _critic = build_ppo_networks(
            observation_dim, action_count, hidden_dim
        )
        model.load_state_dict(checkpoint["actor_state_dict"])
    elif algorithm == "A2C":
        model, _actor_module, _actor, _critic_net, _critic = build_a2c_networks(
            observation_dim, action_count, hidden_dim
        )
        model.load_state_dict(checkpoint["actor_state_dict"])
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    model.eval()

    def act(observation: np.ndarray) -> int:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32)
        with torch.no_grad():
            scores = model(obs_tensor)
            # For DQN these are Q-values; for PG methods these are policy logits.
            action = int(torch.argmax(scores).item())
        if not 0 <= action < action_count:
            raise RuntimeError(f"{algorithm} returned invalid action {action}")
        return action

    print(
        f"Loaded {algorithm}: {checkpoint_path.relative_to(PROJECT_ROOT)} "
        f"(hidden_dim={hidden_dim}, run={config.get('run_name', 'unknown')})"
    )
    return act


def evaluate_one(
    algorithm: str,
    act: Callable[[np.ndarray], int],
    seed: int,
) -> tuple[dict[str, Any], dict[int, int]]:
    env = SmartIrrigationEnv()
    action_counts = {action: 0 for action in ACTION_LABELS}
    try:
        observation, reset_info = env.reset(seed=seed)
        initial_soil = float(reset_info["soil_moisture_mm"])
        initial_reservoir = float(reset_info["reservoir_mm"])
        initial_health = float(env.crop_health)

        episode_return = 0.0
        water_stress_days = 0
        waterlogging_days = 0
        final_info: dict[str, Any] = {}
        episode_length = 0
        terminated = False
        truncated = False

        for _ in range(env.episode_days):
            action = act(observation)
            action_counts[action] += 1
            observation, reward, terminated, truncated, final_info = env.step(action)
            episode_return += float(reward)
            episode_length += 1
            if float(final_info.get("water_stress", 0.0)) > 0.05:
                water_stress_days += 1
            if float(final_info.get("waterlogging", 0.0)) > 0.05:
                waterlogging_days += 1
            if terminated or truncated:
                break

        if not final_info:
            raise RuntimeError(f"{algorithm}, seed={seed}: environment produced no steps")

        row: dict[str, Any] = {
            "algorithm": algorithm,
            "seed": seed,
            "episode_return": episode_return,
            "episode_length_days": episode_length,
            "yield_score": float(final_info.get("yield_score", 0.0)),
            "final_crop_health": float(final_info.get("crop_health", 0.0)),
            "total_irrigation_mm": float(final_info.get("total_irrigation_mm", 0.0)),
            "final_soil_moisture_mm": float(final_info.get("soil_moisture_mm", 0.0)),
            "remaining_reservoir_mm": float(final_info.get("reservoir_mm", 0.0)),
            "water_stress_days": water_stress_days,
            "waterlogging_days": waterlogging_days,
            "terminated_early": bool(terminated),
            "completed_season": bool(truncated),
            "initial_soil_moisture_mm": initial_soil,
            "initial_reservoir_mm": initial_reservoir,
            "initial_crop_health": initial_health,
        }
        return row, action_counts
    finally:
        env.close()


def _mean_std_ci(values: list[float] | np.ndarray) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    margin = 1.96 * std / math.sqrt(len(array)) if len(array) > 1 else 0.0
    return mean, std, mean - margin, mean + margin


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run shared-seed unseen-start generalization evaluation."
    )
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of common unseen seeds (default: 50).")
    parser.add_argument("--seed-start", type=int, default=20_000,
                        help="First unseen seed; defaults to 20000.")
    parser.add_argument("--output-dir", default="analysis/outputs",
                        help="Output directory relative to project root.")
    parser.add_argument("--dqn", default=DEFAULT_CHECKPOINTS["DQN"])
    parser.add_argument("--reinforce", default=DEFAULT_CHECKPOINTS["REINFORCE"])
    parser.add_argument("--ppo", default=DEFAULT_CHECKPOINTS["PPO"])
    parser.add_argument("--a2c", default=DEFAULT_CHECKPOINTS["A2C"])
    args = parser.parse_args()

    if args.episodes < 2:
        parser.error("--episodes must be at least 2 to estimate variability")

    torch.set_num_threads(1)
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    checkpoint_args = {
        "DQN": args.dqn,
        "REINFORCE": args.reinforce,
        "PPO": args.ppo,
        "A2C": args.a2c,
    }
    agents: dict[str, Callable[[np.ndarray], int]] = {}
    for algorithm, relative_path in checkpoint_args.items():
        path = Path(relative_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"Missing {algorithm} checkpoint: {path}")
        agents[algorithm] = load_agent(algorithm, path)

    episode_rows: list[dict[str, Any]] = []
    action_totals = {name: {action: 0 for action in ACTION_LABELS}
                     for name in agents}
    initial_conditions: dict[int, tuple[float, float, float]] = {}

    for algorithm, act in agents.items():
        print(f"\nEvaluating {algorithm} on {len(seeds)} unseen seeds...")
        for index, seed in enumerate(seeds, start=1):
            row, action_counts = evaluate_one(algorithm, act, seed)
            initial = (
                row["initial_soil_moisture_mm"],
                row["initial_reservoir_mm"],
                row["initial_crop_health"],
            )
            if seed in initial_conditions and not np.allclose(
                initial, initial_conditions[seed], rtol=0.0, atol=1e-7
            ):
                raise RuntimeError(f"Initial conditions differ across algorithms for seed {seed}")
            initial_conditions[seed] = initial
            episode_rows.append(row)
            for action, count in action_counts.items():
                action_totals[algorithm][action] += count
            if index == 1 or index % 10 == 0 or index == len(seeds):
                print(
                    f"  {index:>3}/{len(seeds)} | seed={seed} "
                    f"return={row['episode_return']:.3f} "
                    f"yield={row['yield_score']:.2f} "
                    f"health={row['final_crop_health']:.3f}"
                )

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_csv = output_dir / "generalization_episode_results.csv"
    write_csv(episode_csv, episode_rows)

    summary_rows: list[dict[str, Any]] = []
    for algorithm in agents:
        rows = [row for row in episode_rows if row["algorithm"] == algorithm]
        reward_mean, reward_std, reward_ci_low, reward_ci_high = _mean_std_ci(
            [row["episode_return"] for row in rows]
        )
        yield_mean, yield_std, yield_ci_low, yield_ci_high = _mean_std_ci(
            [row["yield_score"] for row in rows]
        )
        health_mean, health_std, _, _ = _mean_std_ci(
            [row["final_crop_health"] for row in rows]
        )
        irrigation_mean, irrigation_std, _, _ = _mean_std_ci(
            [row["total_irrigation_mm"] for row in rows]
        )
        summary: dict[str, Any] = {
            "algorithm": algorithm,
            "episodes": len(rows),
            "mean_eval_reward": reward_mean,
            "std_eval_reward": reward_std,
            "reward_ci95_low": reward_ci_low,
            "reward_ci95_high": reward_ci_high,
            "median_eval_reward": float(np.median([r["episode_return"] for r in rows])),
            "min_eval_reward": float(np.min([r["episode_return"] for r in rows])),
            "max_eval_reward": float(np.max([r["episode_return"] for r in rows])),
            "mean_yield_score": yield_mean,
            "std_yield_score": yield_std,
            "yield_ci95_low": yield_ci_low,
            "yield_ci95_high": yield_ci_high,
            "mean_final_crop_health": health_mean,
            "std_final_crop_health": health_std,
            "mean_irrigation_mm": irrigation_mean,
            "std_irrigation_mm": irrigation_std,
            "mean_water_stress_days": float(np.mean([r["water_stress_days"] for r in rows])),
            "mean_waterlogging_days": float(np.mean([r["waterlogging_days"] for r in rows])),
            "early_termination_rate": float(np.mean([r["terminated_early"] for r in rows])),
            "season_completion_rate": float(np.mean([r["completed_season"] for r in rows])),
            "action_0_none_count": action_totals[algorithm][0],
            "action_1_2mm_count": action_totals[algorithm][1],
            "action_2_5mm_count": action_totals[algorithm][2],
            "action_3_8mm_count": action_totals[algorithm][3],
            "action_4_12mm_count": action_totals[algorithm][4],
        }
        summary_rows.append(summary)

    summary_csv = output_dir / "generalization_summary.csv"
    write_csv(summary_csv, summary_rows)

    # Paired reward differences use the same unseen seed for both algorithms.
    by_algorithm_seed = {
        algorithm: {row["seed"]: float(row["episode_return"])
                    for row in episode_rows if row["algorithm"] == algorithm}
        for algorithm in agents
    }
    paired_rows: list[dict[str, Any]] = []
    reference = "DQN"
    for algorithm in ("REINFORCE", "PPO", "A2C"):
        differences = np.asarray([
            by_algorithm_seed[reference][seed] - by_algorithm_seed[algorithm][seed]
            for seed in seeds
        ], dtype=float)
        mean, std, ci_low, ci_high = _mean_std_ci(differences)
        paired_rows.append({
            "comparison": f"{reference} minus {algorithm}",
            "paired_seeds": len(seeds),
            "mean_reward_difference": mean,
            "std_paired_difference": std,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "interpretation": (
                "DQN higher on average; interval excludes zero"
                if ci_low > 0 else
                "DQN lower on average; interval excludes zero"
                if ci_high < 0 else
                "difference is uncertain; interval includes zero"
            ),
        })
    paired_csv = output_dir / "generalization_paired_comparisons.csv"
    write_csv(paired_csv, paired_rows)

    metadata = {
        "purpose": "Generalization to unseen randomized initial conditions and weather seeds",
        "episode_count_per_algorithm": len(seeds),
        "seed_start": args.seed_start,
        "seed_end_inclusive": seeds[-1],
        "shared_seed_protocol": True,
        "policy_action_selection": "Greedy deterministic argmax; same convention as training evaluation",
        "environment_episode_days": SmartIrrigationEnv.EPISODE_DAYS,
        "action_mapping_mm": {str(k): float(v) for k, v in enumerate(SmartIrrigationEnv.ACTION_WATER_MM)},
        "initial_soil_moisture_range_mm": [38.0, 68.0],
        "initial_reservoir_range_mm": [105.0, 160.0],
        "initial_crop_health": 0.95,
        "reward_ci_method": "Normal-approximation 95% CI: mean +/- 1.96 * sample SD / sqrt(n)",
        "paired_comparison_method": "Per-seed DQN minus comparison-agent return; same normal-approximation CI",
        "checkpoints": checkpoint_args,
        "notes": [
            "These seeds are separated from training seeds 7-126 and original evaluation seeds 10007-10016.",
            "The confidence intervals are descriptive estimates, not proof of broad real-world performance.",
            "The environment is a transparent stochastic simulation, not a calibrated field-crop model.",
        ],
    }
    metadata_path = output_dir / "generalization_protocol.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nGENERALIZATION SUMMARY (mean +/- SD; n={})".format(len(seeds)))
    print("{:<12} {:>10} {:>10} {:>10} {:>12} {:>10}".format(
        "Algorithm", "Reward", "SD", "Yield", "Final health", "Irrig. mm"
    ))
    for row in summary_rows:
        print("{:<12} {:>10.3f} {:>10.3f} {:>10.2f} {:>12.3f} {:>10.2f}".format(
            row["algorithm"], row["mean_eval_reward"], row["std_eval_reward"],
            row["mean_yield_score"], row["mean_final_crop_health"],
            row["mean_irrigation_mm"],
        ))
    print("\nPaired reward comparisons (DQN minus other methods):")
    for row in paired_rows:
        print(
            f"  {row['comparison']}: {row['mean_reward_difference']:.3f} "
            f"[95% CI {row['ci95_low']:.3f}, {row['ci95_high']:.3f}] — "
            f"{row['interpretation']}"
        )
    print("\nSaved:")
    print(f"  {episode_csv.relative_to(PROJECT_ROOT)}")
    print(f"  {summary_csv.relative_to(PROJECT_ROOT)}")
    print(f"  {paired_csv.relative_to(PROJECT_ROOT)}")
    print(f"  {metadata_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
