"""Run ten controlled A2C hyperparameter experiments.

Place this file in training/a2c_sweep.py beside a2c_training.py. Every run
uses the same environment, training seed, episode budget, and held-out
assessment seeds used by DQN, REINFORCE, and PPO.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from training.a2c_training import train_a2c


EXPERIMENTS = [
    {"run_name": "a2c_experiment_01_default"},
    {
        "run_name": "a2c_experiment_02_lr_0001",
        "learning_rate": 1e-4,
    },
    {
        "run_name": "a2c_experiment_03_lr_0006",
        "learning_rate": 6e-4,
    },
    {
        "run_name": "a2c_experiment_04_gamma_095",
        "gamma": 0.95,
    },
    {
        "run_name": "a2c_experiment_05_gamma_0999",
        "gamma": 0.999,
    },
    {
        "run_name": "a2c_experiment_06_gae_090",
        "gae_lambda": 0.90,
    },
    {
        "run_name": "a2c_experiment_07_gae_099",
        "gae_lambda": 0.99,
    },
    {
        "run_name": "a2c_experiment_08_entropy_0",
        "entropy_coefficient": 0.0,
    },
    {
        "run_name": "a2c_experiment_09_entropy_003",
        "entropy_coefficient": 0.03,
    },
    {
        "run_name": "a2c_experiment_10_rollout_8",
        "episodes_per_update": 8,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ten controlled A2C hyperparameter experiments."
    )
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    project_root = Path(__file__).resolve().parents[1]
    summary_path = project_root / "logs" / "a2c" / "a2c_sweep_summary.csv"
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
            f"Starting A2C experiment {index:02d}/10: {experiment['run_name']}",
            flush=True,
        )
        print(f"Changed hyperparameters: {changed}", flush=True)

        result = train_a2c(config)
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
                "mean_recent_critic_loss": result["mean_recent_critic_loss"],
                "mean_recent_policy_entropy": result[
                    "mean_recent_policy_entropy"
                ],
                "mean_eval_reward": result["mean_eval_reward"],
                "std_eval_reward": result["std_eval_reward"],
                "mean_eval_yield_score": result["mean_eval_yield_score"],
                "mean_eval_final_health": result["mean_eval_final_health"],
                "mean_eval_irrigation_mm": result["mean_eval_irrigation_mm"],
                "mean_eval_policy_entropy": result["mean_eval_policy_entropy"],
                "episode_metrics_csv": result["episode_metrics_csv"],
                "update_metrics_csv": result["update_metrics_csv"],
                "model_path": result["model_path"],
            }
        )

        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print("\nAll ten A2C experiments completed.", flush=True)
    print(f"Combined summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
