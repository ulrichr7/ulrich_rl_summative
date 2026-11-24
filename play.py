"""Run the best saved DQN irrigation agent in the custom environment.

Run from the project root:
    uv run python play.py

The Pygame window is enabled by default. Use --no-render for a terminal-only run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from environment.custom_env import SmartIrrigationEnv


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "dqn" / "dqn_experiment_04_gamma_095.pt"
ACTION_WATER_MM = (0.0, 2.0, 5.0, 8.0, 12.0)
ACTION_NAMES = (
    "no irrigation",
    "light irrigation",
    "moderate irrigation",
    "substantial irrigation",
    "heavy irrigation",
)


def load_dqn(model_path: Path) -> tuple[nn.Sequential, dict[str, Any]]:
    """Load the trained DQN network and its saved configuration."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"DQN checkpoint not found: {model_path}\n"
            "Check that the trained model is in models/dqn/ and that it was not excluded from Git."
        )

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("The checkpoint does not contain a 'model_state_dict'.")

    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        config = {}

    observation_dim = int(checkpoint.get("observation_dim", 10))
    action_count = int(checkpoint.get("action_count", 5))
    hidden_dim = int(config.get("hidden_dim", 128))

    if observation_dim != 10 or action_count != len(ACTION_WATER_MM):
        raise ValueError(
            "Checkpoint/environment mismatch: expected 10 observations and 5 actions, "
            f"but checkpoint has {observation_dim} observations and {action_count} actions."
        )

    # This is the same two-hidden-layer MLP used by training/dqn_training.py.
    q_network = nn.Sequential(
        nn.Linear(observation_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, action_count),
    )
    q_network.load_state_dict(checkpoint["model_state_dict"])
    q_network.eval()
    return q_network, config


def select_action(q_network: nn.Module, observation: np.ndarray) -> tuple[int, np.ndarray]:
    """Select the greedy action using the trained DQN Q-values."""
    observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        q_values = q_network(observation_tensor).squeeze(0)
    action = int(torch.argmax(q_values).item())
    return action, q_values.cpu().numpy()


def process_window_events(pygame: Any) -> bool:
    """Return False when the user closes the Pygame window or presses ESC."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play one season with the trained DQN smart-irrigation agent."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to a saved DQN checkpoint (default: best DQN checkpoint).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20000,
        help="Environment seed for this demonstration (default: 20000).",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Run in the terminal without opening the Pygame window.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=5.0,
        help="How long to keep the final Pygame screen open (default: 5 seconds).",
    )
    args = parser.parse_args()

    if args.hold_seconds < 0:
        parser.error("--hold-seconds cannot be negative")

    model_path = args.model
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    q_network, config = load_dqn(model_path)
    render_mode = None if args.no_render else "human"
    env = SmartIrrigationEnv(render_mode=render_mode)

    pygame = None
    if not args.no_render:
        import pygame as pygame_module

        pygame = pygame_module

    total_reward = 0.0
    steps = 0
    running = True
    final_info: dict[str, Any] = {}
    terminated = False
    truncated = False

    try:
        observation, reset_info = env.reset(seed=args.seed)

        print("SMART IRRIGATION - TRAINED DQN AGENT", flush=True)
        print("=" * 66, flush=True)
        print(f"Model: {model_path.relative_to(PROJECT_ROOT) if model_path.is_relative_to(PROJECT_ROOT) else model_path}", flush=True)
        print(f"Training run: {config.get('run_name', 'unknown')}", flush=True)
        print(f"Training episodes: {config.get('episodes', 'unknown')}", flush=True)
        print(f"Environment seed: {args.seed}", flush=True)
        print(f"Initial state: {reset_info}", flush=True)
        print("Policy: greedy action with the highest predicted Q-value", flush=True)
        print("=" * 66, flush=True)
        print(
            "Day | Action | Applied | Soil | Reservoir | Crop health | Rain | "
            "Reward | Total reward",
            flush=True,
        )

        if pygame is not None:
            env.render()
            running = process_window_events(pygame)

        while running and steps < env.episode_days:
            if pygame is not None:
                running = process_window_events(pygame)
                if not running:
                    break

            action, q_values = select_action(q_network, observation)
            next_observation, reward, terminated, truncated, final_info = env.step(action)
            total_reward += float(reward)
            steps += 1

            print(
                f"{final_info['day']:3d} | {action} ({ACTION_NAMES[action]}) | "
                f"{final_info['irrigation_mm']:5.1f} mm | "
                f"{final_info['soil_moisture_mm']:5.1f} mm | "
                f"{final_info['reservoir_mm']:6.1f} mm | "
                f"{final_info['crop_health'] * 100:6.1f}% | "
                f"{final_info['rainfall_mm']:4.1f} mm | "
                f"{reward:7.3f} | {total_reward:9.3f} | "
                f"Q={np.round(q_values, 2).tolist()}",
                flush=True,
            )

            observation = next_observation
            if pygame is not None:
                env.render()
                running = process_window_events(pygame)

            if terminated or truncated:
                break

        print("=" * 66, flush=True)
        if not running and steps < env.episode_days and not final_info:
            print("Playback stopped before the first action because the window was closed.")
        elif not running and steps < env.episode_days:
            print("Playback stopped by the user before the season ended.")
        else:
            print("Episode finished." if (terminated or truncated) else "Episode stopped.")

        print(f"Days completed: {steps}/{env.episode_days}", flush=True)
        print(f"Cumulative reward: {total_reward:.3f}", flush=True)
        if final_info:
            print(f"Final yield score: {final_info.get('yield_score', 0.0):.2f}", flush=True)
            print(f"Final crop health: {final_info.get('crop_health', 0.0) * 100:.1f}%", flush=True)
            print(
                f"Total irrigation: {final_info.get('total_irrigation_mm', 0.0):.1f} mm",
                flush=True,
            )
            print(
                f"Reservoir remaining: {final_info.get('reservoir_mm', 0.0):.1f} mm",
                flush=True,
            )

        # Keep the final visualization visible briefly for the screen recording.
        if pygame is not None and running and args.hold_seconds > 0:
            clock = pygame.time.Clock()
            elapsed = 0.0
            while running and elapsed < args.hold_seconds:
                running = process_window_events(pygame)
                clock.tick(30)
                elapsed += 1.0 / 30.0

    finally:
        env.close()


if __name__ == "__main__":
    main()
