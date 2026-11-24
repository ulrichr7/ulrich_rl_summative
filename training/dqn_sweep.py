"""Run the nine remaining DQN hyperparameter experiments.

Place this file in training/dqn_sweep.py, beside dqn_training.py.
All runs use the same episode budget, seed, and held-out evaluation seeds.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from training.dqn_training import train_dqn


EXPERIMENTS = [
    {
        "run_name": "dqn_experiment_02_lr_0005",
        "learning_rate": 5e-4,
    },
    {
        "run_name": "dqn_experiment_03_lr_002",
        "learning_rate": 2e-3,
    },
    {
        "run_name": "dqn_experiment_04_gamma_095",
        "gamma": 0.95,
    },
    {
        "run_name": "dqn_experiment_05_gamma_0999",
        "gamma": 0.999,
    },
    {
        "run_name": "dqn_experiment_06_batch_32",
        "batch_size": 32,
    },
    {
        "run_name": "dqn_experiment_07_batch_128",
        "batch_size": 128,
    },
    {
        "run_name": "dqn_experiment_08_replay_10000",
        "replay_buffer_size": 10_000,
    },
    {
        "run_name": "dqn_experiment_09_target_tau_001",
        "target_tau": 0.01,
    },
    {
        "run_name": "dqn_experiment_10_epsilon_decay_3000",
        "epsilon_decay_steps": 3_000,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run nine controlled DQN hyperparameter experiments."
    )
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    project_root = Path(__file__).resolve().parents[1]
    summary_path = project_root / "logs" / "dqn" / "dqn_sweep_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for index, experiment in enumerate(EXPERIMENTS, start=2):
        config = {
            "episodes": args.episodes,
            "seed": args.seed,
            **experiment,
        }
        print("\n" + "=" * 72, flush=True)
        print(
            f"Starting DQN experiment {index:02d}/10: "
            f"{experiment['run_name']}",
            flush=True,
        )
        print(
            "Changed hyperparameters: "
            + ", ".join(
                f"{key}={value}" for key, value in experiment.items()
                if key != "run_name"
            ),
            flush=True,
        )

        result = train_dqn(config)
        rows.append(
            {
                "experiment": index,
                "run_name": experiment["run_name"],
                "episodes": args.episodes,
                "seed": args.seed,
                "changed_hyperparameter": ", ".join(
                    f"{key}={value}" for key, value in experiment.items()
                    if key != "run_name"
                ),
                "final_20_mean_training_reward": result[
                    "final_20_mean_training_reward"
                ],
                "mean_recent_training_loss": result[
                    "mean_recent_training_loss"
                ],
                "mean_eval_reward": result["mean_eval_reward"],
                "std_eval_reward": result["std_eval_reward"],
                "mean_eval_yield_score": result["mean_eval_yield_score"],
                "mean_eval_final_health": result["mean_eval_final_health"],
                "mean_eval_irrigation_mm": result[
                    "mean_eval_irrigation_mm"
                ],
                "metrics_csv": result["metrics_csv"],
                "model_path": result["model_path"],
            }
        )

        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print("\nAll nine additional DQN experiments completed.", flush=True)
    print(f"Combined summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
