"""Run ten controlled REINFORCE hyperparameter experiments.

Place this file in training/reinforce_sweep.py beside reinforce_training.py.
All runs use the same environment, episode budget, training seed, and held-out
weather/start seeds for a fair method-specific comparison.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from training.reinforce_training import train_reinforce


EXPERIMENTS = [
    {
        "run_name": "reinforce_experiment_01_default",
    },
    {
        "run_name": "reinforce_experiment_02_lr_0005",
        "learning_rate": 5e-4,
    },
    {
        "run_name": "reinforce_experiment_03_lr_002",
        "learning_rate": 2e-3,
    },
    {
        "run_name": "reinforce_experiment_04_gamma_095",
        "gamma": 0.95,
    },
    {
        "run_name": "reinforce_experiment_05_gamma_0999",
        "gamma": 0.999,
    },
    {
        "run_name": "reinforce_experiment_06_entropy_0",
        "entropy_coefficient": 0.0,
    },
    {
        "run_name": "reinforce_experiment_07_entropy_003",
        "entropy_coefficient": 0.03,
    },
    {
        "run_name": "reinforce_experiment_08_hidden_64",
        "hidden_dim": 64,
    },
    {
        "run_name": "reinforce_experiment_09_full_return",
        "reward_to_go": False,
    },
    {
        "run_name": "reinforce_experiment_10_normalized_returns",
        "normalize_returns": True,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ten controlled REINFORCE hyperparameter experiments."
    )
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    project_root = Path(__file__).resolve().parents[1]
    summary_path = (
        project_root / "logs" / "reinforce" / "reinforce_sweep_summary.csv"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for index, experiment in enumerate(EXPERIMENTS, start=1):
        config = {
            "episodes": args.episodes,
            "seed": args.seed,
            **experiment,
        }
        changed = ", ".join(
            f"{key}={value}"
            for key, value in experiment.items()
            if key != "run_name"
        ) or "default configuration"

        print("\n" + "=" * 76, flush=True)
        print(
            f"Starting REINFORCE experiment {index:02d}/10: "
            f"{experiment['run_name']}",
            flush=True,
        )
        print(f"Changed hyperparameters: {changed}", flush=True)

        result = train_reinforce(config)
        rows.append(
            {
                "experiment": index,
                "run_name": experiment["run_name"],
                "episodes": args.episodes,
                "seed": args.seed,
                "changed_hyperparameter": changed,
                "final_20_mean_training_reward": result[
                    "final_20_mean_training_reward"
                ],
                "mean_recent_policy_loss": result["mean_recent_policy_loss"],
                "mean_recent_policy_entropy": result[
                    "mean_recent_policy_entropy"
                ],
                "mean_eval_reward": result["mean_eval_reward"],
                "std_eval_reward": result["std_eval_reward"],
                "mean_eval_yield_score": result["mean_eval_yield_score"],
                "mean_eval_final_health": result["mean_eval_final_health"],
                "mean_eval_irrigation_mm": result["mean_eval_irrigation_mm"],
                "metrics_csv": result["metrics_csv"],
                "model_path": result["model_path"],
            }
        )

        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print("\nAll ten REINFORCE experiments completed.", flush=True)
    print(f"Combined summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
