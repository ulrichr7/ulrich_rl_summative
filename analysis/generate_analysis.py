"""Generate report-ready comparison tables and figures for the RL summative.

Run from the project root with:
    uv run python analysis/generate_analysis.py

Expected input:
  logs/dqn/dqn_sweep_summary.csv
  logs/reinforce/reinforce_sweep_summary.csv
  logs/ppo/ppo_sweep_summary.csv
  logs/a2c/a2c_sweep_summary.csv
  per-run episode_metrics.csv and (for PPO/A2C) update_metrics.csv

The script validates the 40 experiment rows, identifies each algorithm's best
configuration using mean evaluation reward, and generates figures/tables under
analysis/outputs/. It does not retrain models or modify experiment results.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
OUT = ROOT / "analysis" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

SUMMARY_FILES = {
    "DQN": LOGS / "dqn" / "dqn_sweep_summary.csv",
    "REINFORCE": LOGS / "reinforce" / "reinforce_sweep_summary.csv",
    "PPO": LOGS / "ppo" / "ppo_sweep_summary.csv",
    "A2C": LOGS / "a2c" / "a2c_sweep_summary.csv",
}


def first_existing_column(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def localize_path(value: object) -> Path | None:
    """Resolve a recorded absolute/relative path against this project root."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    raw = str(value).strip()
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    if candidate.is_file():
        return candidate
    # The experiment logs may store paths from the original Mac user directory.
    normalized = raw.replace("\\", "/")
    for marker in ("/logs/", "logs/"):
        if marker in normalized:
            suffix = normalized.split(marker, 1)[1]
            candidate = LOGS / suffix
            if candidate.is_file():
                return candidate
    # Model paths can also be recorded as absolute paths; metrics are expected
    # to live at logs/<algorithm>/<run_name>/episode_metrics.csv.
    return None


def resolve_metrics_path(row: pd.Series, algorithm: str, kind: str) -> Path:
    field_options = (
        ["metrics_csv", "episode_metrics_csv"]
        if kind == "episode"
        else ["update_metrics_csv"]
    )
    for field in field_options:
        path = localize_path(row.get(field))
        if path is not None:
            return path

    run_name = str(row["run_name"])
    leaf = "episode_metrics.csv" if kind == "episode" else "update_metrics.csv"
    candidate = LOGS / algorithm.lower() / run_name / leaf
    if candidate.is_file():
        return candidate

    # DQN Experiment 01 was originally stored separately; its corrected summary
    # row points to logs/dqn/dqn_experiment_01/episode_metrics.csv.
    raise FileNotFoundError(
        f"Could not find {kind} metrics for {algorithm}/{run_name}. "
        f"Tried summary path fields {field_options} and {candidate}."
    )


def load_summaries() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for algorithm, path in SUMMARY_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing summary file: {path}")
        df = pd.read_csv(path)
        if len(df) != 10:
            raise ValueError(
                f"{algorithm} has {len(df)} rows in {path}; expected 10. "
                "Check the experiment summary before plotting."
            )
        if "algorithm" in df.columns:
            df = df.drop(columns=["algorithm"])
        df.insert(0, "algorithm", algorithm)
        frames.append(df)
        print(f"{algorithm}: {len(df)} experiments — OK")

    master = pd.concat(frames, ignore_index=True)
    if len(master) != 40:
        raise ValueError(f"Expected 40 experiments, found {len(master)}")
    if master["run_name"].duplicated().any():
        duplicates = master.loc[master["run_name"].duplicated(), "run_name"].tolist()
        raise ValueError(f"Duplicate experiment run names: {duplicates}")
    return master.sort_values(["algorithm", "experiment"]).reset_index(drop=True)


def build_best_table(master: pd.DataFrame) -> pd.DataFrame:
    best = (
        master.sort_values("mean_eval_reward", ascending=False)
        .groupby("algorithm", as_index=False)
        .head(1)
        .sort_values("mean_eval_reward", ascending=False)
        .reset_index(drop=True)
    )
    return best


def rolling_mean(values: pd.Series, window: int = 10) -> pd.Series:
    return values.rolling(window=window, min_periods=1).mean()


def load_episode_frame(row: pd.Series) -> pd.DataFrame:
    algorithm = str(row["algorithm"])
    path = resolve_metrics_path(row, algorithm, "episode")
    df = pd.read_csv(path)
    episode_col = first_existing_column(df, ["episode", "episodes_completed", "step"])
    reward_col = first_existing_column(
        df, ["episode_reward", "reward", "total_reward", "mean_rollout_reward"]
    )
    if reward_col is None:
        raise ValueError(
            f"No reward column found in {path}. Available columns: {list(df.columns)}"
        )
    if episode_col is None:
        df = df.copy()
        df["episode"] = np.arange(1, len(df) + 1)
        episode_col = "episode"
    return df.rename(columns={episode_col: "episode", reward_col: "episode_reward"})


def load_update_frame(row: pd.Series) -> pd.DataFrame | None:
    algorithm = str(row["algorithm"])
    if algorithm not in {"PPO", "A2C"}:
        return None
    path = resolve_metrics_path(row, algorithm, "update")
    df = pd.read_csv(path)
    update_col = first_existing_column(df, ["episodes_completed", "update", "episode"])
    if update_col is None:
        df = df.copy()
        df["update"] = np.arange(1, len(df) + 1)
        update_col = "update"
    if update_col != "episodes_completed":
        df = df.rename(columns={update_col: "episodes_completed"})
    return df


def make_best_model_reward_figures(best: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, (_, row) in zip(axes.flat, best.iterrows()):
        df = load_episode_frame(row)
        rewards = pd.to_numeric(df["episode_reward"], errors="coerce")
        episodes = pd.to_numeric(df["episode"], errors="coerce")
        ax.plot(episodes, rewards, linewidth=0.8, alpha=0.35, label="Episode reward")
        ax.plot(episodes, rolling_mean(rewards, 10), linewidth=2, label="10-episode moving mean")
        ax.axhline(0, linewidth=0.7)
        ax.set_title(f"{row['algorithm']} — {row['run_name']}")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Episode reward")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Training reward trajectories for the best configuration of each algorithm", fontsize=14)
    fig.savefig(OUT / "best_models_training_rewards.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, (_, row) in zip(axes.flat, best.iterrows()):
        df = load_episode_frame(row)
        rewards = pd.to_numeric(df["episode_reward"], errors="coerce").fillna(0)
        episodes = pd.to_numeric(df["episode"], errors="coerce")
        ax.plot(episodes, rewards.cumsum(), linewidth=2)
        ax.set_title(str(row["algorithm"]))
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Cumulative training reward")
        ax.grid(True, alpha=0.25)
    fig.suptitle("Cumulative reward across training episodes — best configuration per algorithm", fontsize=14)
    fig.savefig(OUT / "best_models_cumulative_rewards.png", dpi=220)
    plt.close(fig)



def make_convergence_estimates(best: pd.DataFrame) -> None:
    """Estimate stable-performance entry using an explicit rolling-window rule.

    Operational rule: compute a 10-episode moving mean. The estimated
    stabilization episode is the first episode at which the moving mean stays
    within a tolerance of the final-20-episode mean for 10 consecutive points.
    Tolerance is max(10% of the absolute final-20 mean, 0.5 reward points).
    This is a reporting convention, not proof of mathematical convergence.
    """
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, (_, row) in zip(axes.flat, best.iterrows()):
        df = load_episode_frame(row)
        rewards = pd.to_numeric(df["episode_reward"], errors="coerce")
        episodes = pd.to_numeric(df["episode"], errors="coerce")
        smooth = rolling_mean(rewards, 10)
        final_mean = float(rewards.tail(20).mean())
        tolerance = max(0.10 * abs(final_mean), 0.5)
        within = (smooth - final_mean).abs() <= tolerance
        stable_episode = None
        for start in range(0, max(0, len(within) - 9)):
            if bool(within.iloc[start : start + 10].all()):
                stable_episode = int(episodes.iloc[start])
                break
        rows.append({
            "algorithm": row["algorithm"],
            "run_name": row["run_name"],
            "final_20_mean_training_reward": final_mean,
            "stability_tolerance": tolerance,
            "episodes_to_stability_estimate": stable_episode if stable_episode is not None else "not reached",
            "criterion": "10-episode moving mean within max(10% of |final-20 mean|, 0.5) for 10 consecutive episodes",
        })
        ax.plot(episodes, smooth, linewidth=1.8, label="10-episode moving mean")
        ax.axhline(final_mean, linestyle="--", linewidth=1.0, label="Final-20 mean")
        ax.fill_between(
            episodes, final_mean - tolerance, final_mean + tolerance,
            alpha=0.15, label="Stability band"
        )
        if stable_episode is not None:
            ax.axvline(stable_episode, linestyle=":", linewidth=1.0, label=f"Estimate: episode {stable_episode}")
        ax.set_title(str(row["algorithm"]))
        ax.set_xlabel("Training episode")
        ax.set_ylabel("10-episode mean reward")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Operational estimate of training stabilization (best configuration per algorithm)", fontsize=13)
    fig.savefig(OUT / "best_models_convergence_estimate.png", dpi=220)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / "episodes_to_stability_estimate.csv", index=False)

def make_evaluation_comparison(best: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    names = best["algorithm"].tolist()
    means = pd.to_numeric(best["mean_eval_reward"], errors="coerce").to_numpy()
    stds = pd.to_numeric(best["std_eval_reward"], errors="coerce").to_numpy()
    ax.bar(names, means, yerr=stds, capsize=5)
    ax.axhline(0, linewidth=0.8)
    ax.set_title("Best configuration evaluation reward by algorithm")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Mean evaluation reward (± standard deviation)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(OUT / "best_model_evaluation_comparison.png", dpi=220)
    plt.close(fig)


def make_dqn_loss(best: pd.DataFrame) -> None:
    row = best.loc[best["algorithm"] == "DQN"].iloc[0]
    df = load_episode_frame(row)
    loss_col = first_existing_column(
        df, ["loss", "td_loss", "training_loss", "mean_loss", "dqn_loss"]
    )
    if loss_col is None:
        print(
            "WARNING: Could not find DQN loss column. Available columns:",
            list(df.columns),
        )
        return
    values = pd.to_numeric(df[loss_col], errors="coerce")
    episodes = pd.to_numeric(df["episode"], errors="coerce")
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(episodes, values, alpha=0.35, linewidth=0.8, label="Recorded DQN loss")
    ax.plot(episodes, rolling_mean(values, 10), linewidth=2, label="10-episode moving mean")
    ax.set_title(f"DQN objective / TD loss — {row['run_name']}")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("DQN loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(OUT / "dqn_objective_loss.png", dpi=220)
    plt.close(fig)


def make_policy_entropy(best: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, algorithm in zip(axes, ["REINFORCE", "PPO", "A2C"]):
        row = best.loc[best["algorithm"] == algorithm].iloc[0]
        if algorithm == "REINFORCE":
            df = load_episode_frame(row)
            entropy_col = first_existing_column(df, ["policy_entropy", "entropy"])
            x_col = "episode"
            label = "Policy entropy"
            if entropy_col is None:
                print("WARNING: REINFORCE entropy column not found in", row["run_name"], list(df.columns))
                continue
            x = pd.to_numeric(df[x_col], errors="coerce")
            y = pd.to_numeric(df[entropy_col], errors="coerce")
        else:
            df = load_update_frame(row)
            if df is None:
                continue
            x = pd.to_numeric(df["episodes_completed"], errors="coerce")
            if algorithm == "A2C":
                entropy_col = first_existing_column(df, ["policy_entropy", "entropy"])
                if entropy_col is None:
                    print("WARNING: A2C policy entropy column not found. Available:", list(df.columns))
                    continue
                y = pd.to_numeric(df[entropy_col], errors="coerce")
                label = "Policy entropy"
            else:
                # PPO update_metrics records the entropy bonus already scaled by
                # entropy_coefficient. Recover raw entropy using the run config.
                entropy_col = first_existing_column(df, ["entropy_bonus"])
                if entropy_col is None:
                    print("WARNING: PPO entropy bonus column not found. Available:", list(df.columns))
                    continue
                config_path = LOGS / "ppo" / str(row["run_name"]) / "config.json"
                coeff = 0.01
                if config_path.is_file():
                    coeff = float(json.loads(config_path.read_text()).get("entropy_coefficient", 0.01))
                if coeff == 0:
                    # If the regularizer is disabled, its bonus is zero and does
                    # not reveal policy entropy. Plot the bonus honestly instead.
                    y = pd.to_numeric(df[entropy_col], errors="coerce")
                    label = "Entropy bonus (coefficient = 0; uninformative)"
                else:
                    y = pd.to_numeric(df[entropy_col], errors="coerce") / coeff
                    label = "Estimated policy entropy (entropy bonus / coefficient)"
        ax.plot(x, y, linewidth=1.8)
        ax.set_title(f"{algorithm} — {row['run_name']}")
        ax.set_xlabel("Training episodes completed")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Policy entropy during training for policy-gradient methods", fontsize=14)
    fig.savefig(OUT / "policy_gradient_entropy.png", dpi=220)
    plt.close(fig)


def make_ppo_diagnostics(best: pd.DataFrame) -> None:
    row = best.loc[best["algorithm"] == "PPO"].iloc[0]
    df = load_update_frame(row)
    if df is None:
        return
    x = pd.to_numeric(df["episodes_completed"], errors="coerce")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    for ax, col, title, ylabel in [
        (axes[0], "clip_fraction", "PPO clipping fraction", "Clip fraction"),
        (axes[1], "critic_loss", "PPO critic loss", "Critic loss"),
    ]:
        if col not in df.columns:
            ax.text(0.5, 0.5, f"Missing column: {col}", ha="center", va="center")
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        ax.plot(x, values, marker="o", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("Training episodes completed")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    fig.savefig(OUT / "ppo_clipping_and_critic_loss.png", dpi=220)
    plt.close(fig)


def make_a2c_critic(best: pd.DataFrame) -> None:
    row = best.loc[best["algorithm"] == "A2C"].iloc[0]
    df = load_update_frame(row)
    if df is None or "critic_loss" not in df.columns:
        return
    x = pd.to_numeric(df["episodes_completed"], errors="coerce")
    y = pd.to_numeric(df["critic_loss"], errors="coerce")
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax.plot(x, y, marker="o", linewidth=1.3)
    ax.set_title(f"A2C critic loss — {row['run_name']}")
    ax.set_xlabel("Training episodes completed")
    ax.set_ylabel("Critic loss")
    ax.grid(True, alpha=0.25)
    fig.savefig(OUT / "a2c_critic_loss.png", dpi=220)
    plt.close(fig)


def main() -> None:
    master = load_summaries()
    best = build_best_table(master)

    master.to_csv(OUT / "all_40_experiments.csv", index=False)
    best.to_csv(OUT / "best_model_comparison.csv", index=False)

    # Compact, report-friendly table with consistent column names.
    selected = [
        "algorithm", "experiment", "run_name", "changed_hyperparameter",
        "final_20_mean_training_reward", "mean_eval_reward", "std_eval_reward",
        "mean_eval_yield_score", "mean_eval_final_health", "mean_eval_irrigation_mm",
    ]
    best[selected].to_csv(OUT / "best_model_results_table.csv", index=False)

    make_best_model_reward_figures(best)
    make_convergence_estimates(best)
    make_evaluation_comparison(best)
    make_dqn_loss(best)
    make_policy_entropy(best)
    make_ppo_diagnostics(best)
    make_a2c_critic(best)

    lines = [
        "RL SUMMATIVE — ANALYSIS OUTPUTS",
        "================================",
        f"Total experiments validated: {len(master)}",
        "Experiments per algorithm: " + ", ".join(
            f"{name}={count}" for name, count in master.groupby("algorithm").size().items()
        ),
        "",
        "Best configuration per algorithm (ranked by mean evaluation reward):",
    ]
    for _, row in best.iterrows():
        lines.append(
            f"- {row['algorithm']}: {row['run_name']} | "
            f"eval reward={float(row['mean_eval_reward']):.4f} | "
            f"std={float(row['std_eval_reward']):.4f} | "
            f"yield={float(row['mean_eval_yield_score']):.2f} | "
            f"final health={float(row['mean_eval_final_health']):.4f} | "
            f"irrigation={float(row['mean_eval_irrigation_mm']):.2f} mm"
        )
    lines += [
        "",
        "Interpretation cautions:",
        "- Evaluation summaries currently use 10 fixed held-out seeds; treat ranking as preliminary.",
        "- Compare training reward curves separately from held-out evaluation reward.",
        "- Different policies can produce identical greedy action sequences on the same evaluation seeds.",
        "- Do not claim statistical significance from these 10 evaluation episodes.",
        "- The figures are generated from the recorded episode/update CSVs; no experiment values are fabricated.",
    ]
    (OUT / "analysis_notes.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nValidated {len(master)} experiments.")
    print(best[["algorithm", "run_name", "mean_eval_reward"]].to_string(index=False))
    print(f"\nReport tables and figures saved to: {OUT}")
    for path in sorted(OUT.iterdir()):
        print(" -", path.name)


if __name__ == "__main__":
    main()
