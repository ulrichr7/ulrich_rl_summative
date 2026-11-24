"""Train a TorchRL-backed REINFORCE policy in SmartIrrigationEnv.

The implementation uses TorchRL's TensorDictModule and ProbabilisticActor to
represent and sample a categorical stochastic policy. The update is the
Monte-Carlo REINFORCE policy-gradient objective with discounted reward-to-go,
optional episode-return EMA baseline, and an entropy bonus.

This intentionally uses the same Gymnasium environment, training seeds, and
held-out evaluation seed protocol as the DQN implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn
from torch.distributions import Categorical as TorchCategorical
from torchrl.data import Categorical as CategoricalSpec
from torchrl.modules import ProbabilisticActor

from environment.custom_env import SmartIrrigationEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ReinforceConfig:
    """Hyperparameters for one REINFORCE experiment."""

    episodes: int = 120
    learning_rate: float = 1e-3
    gamma: float = 0.99
    hidden_dim: int = 128
    entropy_coefficient: float = 0.01
    reward_to_go: bool = True
    normalize_returns: bool = False
    use_ema_baseline: bool = True
    baseline_decay: float = 0.90
    gradient_clip: float = 5.0
    evaluation_episodes: int = 10
    seed: int = 7
    run_name: str = "reinforce_pilot"

    @classmethod
    def from_mapping(cls, config: dict[str, Any] | None) -> "ReinforceConfig":
        if config is None:
            return cls()
        valid = {
            key: value
            for key, value in config.items()
            if key in cls.__dataclass_fields__
        }
        return cls(**valid)


def build_policy(observation_dim: int, action_count: int, hidden_dim: int):
    """Create the categorical stochastic policy using TorchRL modules."""
    policy_network = nn.Sequential(
        nn.Linear(observation_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, action_count),
    )
    logits_module = TensorDictModule(
        policy_network,
        in_keys=["observation"],
        out_keys=["logits"],
    )
    actor = ProbabilisticActor(
        logits_module,
        in_keys=["logits"],
        out_keys=["action"],
        spec=CategoricalSpec(n=action_count),
        distribution_class=TorchCategorical,
        return_log_prob=True,
        default_interaction_type="random",
    )
    return policy_network, logits_module, actor


def _discounted_returns(rewards: list[float], gamma: float, reward_to_go: bool) -> torch.Tensor:
    """Compute either per-step reward-to-go or the full episode return."""
    running_return = 0.0
    returns_reversed: list[float] = []
    for reward in reversed(rewards):
        running_return = float(reward) + gamma * running_return
        returns_reversed.append(running_return)
    returns = list(reversed(returns_reversed))
    if not reward_to_go and returns:
        returns = [returns[0]] * len(returns)
    return torch.tensor(returns, dtype=torch.float32)


def _sample_action(actor, observation: np.ndarray) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Sample an action and compute differentiable log-probability and entropy.

    TorchRL versions can expose a probabilistic actor's sampled log-probability
    under different keys (for example, ``sample_log_prob`` or
    ``action_log_prob``). To keep this implementation version-robust, recompute
    log_prob from the actor's original logits and sampled discrete action. This
    retains the gradient path through the logits for the REINFORCE update.
    """
    td = TensorDict(
        {"observation": torch.as_tensor(observation, dtype=torch.float32)},
        batch_size=[],
    )
    td = actor(td)

    logits = td.get("logits")
    sampled_action = td.get("action")
    if logits is None or sampled_action is None:
        raise RuntimeError(
            "TorchRL policy did not return the expected 'logits' and 'action' keys. "
            f"Available keys: {list(td.keys())}"
        )

    action_tensor = sampled_action.reshape(-1)[0].to(dtype=torch.long)
    distribution = TorchCategorical(logits=logits)
    log_probability = distribution.log_prob(action_tensor).reshape(())
    entropy = distribution.entropy().reshape(())
    return int(action_tensor.item()), log_probability, entropy


def _evaluate(policy_network: nn.Module, config: ReinforceConfig) -> dict[str, float]:
    """Evaluate a greedy policy on the same ten held-out seeds used by DQN."""
    eval_env = SmartIrrigationEnv()
    episode_returns: list[float] = []
    yield_scores: list[float] = []
    final_health: list[float] = []
    irrigation_totals: list[float] = []
    mean_entropies: list[float] = []

    try:
        policy_network.eval()
        with torch.no_grad():
            for episode in range(config.evaluation_episodes):
                observation, _ = eval_env.reset(
                    seed=10_000 + config.seed + episode
                )
                total_reward = 0.0
                entropy_values: list[float] = []
                final_info: dict[str, Any] = {}

                for _ in range(eval_env.episode_days):
                    obs_tensor = torch.as_tensor(observation, dtype=torch.float32)
                    logits = policy_network(obs_tensor)
                    distribution = TorchCategorical(logits=logits)
                    action = int(torch.argmax(logits).item())
                    entropy_values.append(float(distribution.entropy().item()))
                    observation, reward, terminated, truncated, final_info = eval_env.step(action)
                    total_reward += float(reward)
                    if terminated or truncated:
                        break

                episode_returns.append(total_reward)
                yield_scores.append(float(final_info.get("yield_score", 0.0)))
                final_health.append(float(final_info.get("crop_health", 0.0)))
                irrigation_totals.append(
                    float(final_info.get("total_irrigation_mm", 0.0))
                )
                mean_entropies.append(
                    float(np.mean(entropy_values)) if entropy_values else 0.0
                )
    finally:
        eval_env.close()
        policy_network.train()

    return {
        "mean_eval_reward": float(np.mean(episode_returns)),
        "std_eval_reward": float(np.std(episode_returns)),
        "mean_eval_yield_score": float(np.mean(yield_scores)),
        "mean_eval_final_health": float(np.mean(final_health)),
        "mean_eval_irrigation_mm": float(np.mean(irrigation_totals)),
        "mean_eval_policy_entropy": float(np.mean(mean_entropies)),
    }


def train_reinforce(
    config: dict[str, Any] | ReinforceConfig | None = None,
) -> dict[str, Any]:
    """Train one REINFORCE configuration and save metrics, checkpoint, and summary."""
    cfg = (
        config
        if isinstance(config, ReinforceConfig)
        else ReinforceConfig.from_mapping(config)
    )
    if cfg.episodes < 1:
        raise ValueError("episodes must be at least 1")
    if cfg.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 <= cfg.gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1")
    if not 0.0 <= cfg.baseline_decay < 1.0:
        raise ValueError("baseline_decay must be in [0, 1)")
    if cfg.hidden_dim < 1 or cfg.evaluation_episodes < 1:
        raise ValueError("hidden_dim and evaluation_episodes must be positive")

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cpu")

    env = SmartIrrigationEnv()
    observation_dim = int(env.observation_space.shape[0])
    action_count = int(env.action_space.n)
    policy_network, logits_module, actor = build_policy(
        observation_dim, action_count, cfg.hidden_dim
    )
    policy_network.to(device)
    logits_module.to(device)
    actor.to(device)

    optimizer = torch.optim.Adam(policy_network.parameters(), lr=cfg.learning_rate)

    log_dir = PROJECT_ROOT / "logs" / "reinforce" / cfg.run_name
    model_dir = PROJECT_ROOT / "models" / "pg"
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / "episode_metrics.csv"
    config_path = log_dir / "config.json"
    summary_path = log_dir / "summary.json"
    model_path = model_dir / f"{cfg.run_name}.pt"
    config_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    episode_rows: list[dict[str, float | int]] = []
    loss_history: list[float] = []
    entropy_history: list[float] = []
    running_baseline = 0.0

    try:
        for episode in range(cfg.episodes):
            observation, _ = env.reset(seed=cfg.seed + episode)
            episode_rewards: list[float] = []
            log_probabilities: list[torch.Tensor] = []
            entropies: list[torch.Tensor] = []
            final_info: dict[str, Any] = {}

            for _ in range(env.episode_days):
                action, log_probability, entropy = _sample_action(actor, observation)
                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    final_info,
                ) = env.step(action)

                log_probabilities.append(log_probability)
                entropies.append(entropy)
                episode_rewards.append(float(reward))
                observation = next_observation

                if terminated or truncated:
                    break

            returns = _discounted_returns(
                episode_rewards, cfg.gamma, cfg.reward_to_go
            ).to(device)
            policy_weights = returns.clone()
            if cfg.use_ema_baseline:
                policy_weights = policy_weights - running_baseline
            if cfg.normalize_returns and policy_weights.numel() > 1:
                policy_weights = (
                    policy_weights - policy_weights.mean()
                ) / (policy_weights.std(unbiased=False) + 1e-8)

            log_prob_tensor = torch.stack(log_probabilities).to(device)
            entropy_tensor = torch.stack(entropies).to(device)
            entropy_mean = entropy_tensor.mean()
            policy_loss = -(
                log_prob_tensor * policy_weights.detach()
            ).mean() - cfg.entropy_coefficient * entropy_mean

            optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                policy_network.parameters(), cfg.gradient_clip
            )
            optimizer.step()

            episode_reward = float(sum(episode_rewards))
            if cfg.use_ema_baseline:
                running_baseline = (
                    cfg.baseline_decay * running_baseline
                    + (1.0 - cfg.baseline_decay) * episode_reward
                )

            loss_value = float(policy_loss.detach().cpu().item())
            entropy_value = float(entropy_mean.detach().cpu().item())
            loss_history.append(loss_value)
            entropy_history.append(entropy_value)

            row: dict[str, float | int] = {
                "episode": episode + 1,
                "episode_reward": episode_reward,
                "policy_loss": loss_value,
                "policy_entropy": entropy_value,
                "gradient_norm": float(grad_norm.detach().cpu().item()),
                "discounted_return_start": float(returns[0].detach().cpu().item()),
                "yield_score": float(final_info.get("yield_score", 0.0)),
                "final_crop_health": float(final_info.get("crop_health", 0.0)),
                "total_irrigation_mm": float(
                    final_info.get("total_irrigation_mm", 0.0)
                ),
                "reservoir_remaining_mm": float(
                    final_info.get("reservoir_mm", 0.0)
                ),
            }
            episode_rows.append(row)

            print(
                f"REINFORCE episode {episode + 1:03d}/{cfg.episodes} | "
                f"reward={episode_reward:8.3f} | "
                f"yield={row['yield_score']:6.2f} | "
                f"health={row['final_crop_health']:.3f} | "
                f"entropy={entropy_value:.4f} | "
                f"loss={loss_value:.4f}",
                flush=True,
            )

        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0].keys()))
            writer.writeheader()
            writer.writerows(episode_rows)

        evaluation = _evaluate(policy_network, cfg)
        torch.save(
            {
                "model_state_dict": policy_network.state_dict(),
                "config": asdict(cfg),
                "evaluation": evaluation,
                "observation_dim": observation_dim,
                "action_count": action_count,
            },
            model_path,
        )

        summary = {
            "run_name": cfg.run_name,
            "episodes": cfg.episodes,
            "final_20_mean_training_reward": float(
                np.mean([row["episode_reward"] for row in episode_rows[-20:]])
            ),
            "mean_recent_policy_loss": float(np.mean(loss_history[-20:])),
            "mean_recent_policy_entropy": float(np.mean(entropy_history[-20:])),
            **evaluation,
            "metrics_csv": str(metrics_path),
            "model_path": str(model_path),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("\nREINFORCE evaluation:")
        print(json.dumps(summary, indent=2))
        return {
            "policy_network": policy_network,
            "metrics": episode_rows,
            **summary,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a TorchRL REINFORCE irrigation policy"
    )
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--run-name", default="reinforce_pilot")
    args = parser.parse_args()
    train_reinforce(
        {
            "episodes": args.episodes,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "entropy_coefficient": args.entropy_coefficient,
            "run_name": args.run_name,
        }
    )


if __name__ == "__main__":
    main()
