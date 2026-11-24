"""Run the smart-irrigation environment with an optional live Pygame view."""

from __future__ import annotations

import argparse

import numpy as np

from environment.custom_env import SmartIrrigationEnv


def window_is_open(pygame) -> bool:
    """Process the Pygame event queue and detect user exit requests."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart Irrigation RL simulator")
    parser.add_argument("--render", action="store_true", help="show the live 2D farm window")
    parser.add_argument("--days", type=int, default=90, help="maximum simulated days")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    args = parser.parse_args()

    if args.days < 1:
        parser.error("--days must be at least 1")

    pygame = None
    if args.render:
        import pygame as pygame_module
        pygame = pygame_module

    env = SmartIrrigationEnv(render_mode="human" if args.render else None)
    rng = np.random.default_rng(args.seed)
    total_reward = 0.0
    running = True

    try:
        observation, info = env.reset(seed=args.seed)
        print("SMART IRRIGATION SIMULATOR")
        print("=" * 58)
        print(f"Observation shape: {observation.shape}")
        print(f"Initial state: {info}")
        print("Actions: 0=none, 1=2mm, 2=5mm, 3=8mm, 4=12mm")
        print("This is a random baseline, not a trained agent.")
        print("=" * 58)

        # Draw the initial farm state before taking the first action.
        if args.render:
            env.render()
            pygame.display.flip()
            running = window_is_open(pygame)

        for _ in range(args.days):
            if args.render:
                running = window_is_open(pygame)
                if not running:
                    break

            # Random policy is only a smoke-test baseline.
            action = int(rng.integers(0, env.action_space.n))
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            print(
                f"day={info['day']:02d} action={action} "
                f"irrigation={info['irrigation_mm']:.1f}mm "
                f"soil={info['soil_moisture_mm']:.1f}mm "
                f"health={info['crop_health']:.3f} "
                f"reservoir={info['reservoir_mm']:.1f}mm "
                f"reward={reward:.3f}",
                flush=True,
            )

            if args.render:
                env.render()
                # Process macOS window events so the window stays responsive.
                running = window_is_open(pygame)

            if terminated or truncated:
                print("Episode finished.")
                break

        print("=" * 58)
        print(f"Cumulative reward: {total_reward:.3f}")

        # Keep the completed visualization open until the user closes it.
        if args.render and running:
            print("Farm window is still open. Close it or press ESC to exit.")
            clock = pygame.time.Clock()
            while running:
                running = window_is_open(pygame)
                clock.tick(30)
    finally:
        env.close()


if __name__ == "__main__":
    main()
