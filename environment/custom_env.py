"""Custom, sequential maize-irrigation environment for the RL summative project.

The environment is a transparent simulation, not a calibrated agronomic crop model.
One step represents one day. The agent's choices alter persistent soil-water,
reservoir, crop-health, and cumulative-yield state.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SmartIrrigationEnv(gym.Env[np.ndarray, int]):
    """Manage irrigation for a single simulated maize plot over a 90-day season.

    Actions (discrete):
        0: no irrigation (0 mm)
        1: light irrigation (2 mm)
        2: moderate irrigation (5 mm)
        3: substantial irrigation (8 mm)
        4: heavy irrigation (12 mm)

    Observation vector (float32, each component in [0, 1]):
        0 soil moisture / 100 mm
        1 reservoir water / 160 mm
        2 crop health
        3 season progress (day / 90)
        4 previous day's rainfall / 20 mm
        5 temperature mapped from 10--40 C to 0--1
        6 consecutive water-stress days / 14, clipped to 1
        7 cumulative rainfall / 600 mm, clipped to 1
        8 previous action / 4
        9 current crop-stage index / 3

    Weather is sampled from a seeded stochastic seasonal process. This supports
    repeatable experiments and out-of-distribution evaluation with new seeds.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    ACTION_WATER_MM = np.asarray([0.0, 2.0, 5.0, 8.0, 12.0], dtype=np.float32)
    EPISODE_DAYS = 90
    SOIL_CAPACITY_MM = 100.0
    RESERVOIR_CAPACITY_MM = 160.0
    TARGET_MOISTURE = (58.0, 68.0, 72.0, 60.0)  # stage-specific target (% of capacity)

    def __init__(self, render_mode: str | None = None, episode_days: int = EPISODE_DAYS):
        super().__init__()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'")
        if episode_days < 20:
            raise ValueError("episode_days must be at least 20 to represent crop development")

        self.render_mode = render_mode
        self.episode_days = int(episode_days)
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=np.zeros(10, dtype=np.float32),
            high=np.ones(10, dtype=np.float32),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng()
        self.day = 0
        self.soil_moisture_mm = 0.0
        self.reservoir_mm = 0.0
        self.crop_health = 1.0
        self.cumulative_rain_mm = 0.0
        self.previous_rain_mm = 0.0
        self.temperature_c = 25.0
        self.stress_days = 0
        self.previous_action = 0
        self.yield_factor_sum = 0.0
        self.total_irrigation_mm = 0.0
        self._last_reward = 0.0
        self._last_info: dict[str, Any] = {}
        self._pygame = None
        self._screen = None
        self._clock = None

    def _stage(self) -> int:
        progress = self.day / self.episode_days
        return min(3, int(progress * 4))

    def _weather(self) -> tuple[float, float]:
        """Return daily rainfall (mm) and temperature (C), with seasonal variation."""
        seasonal_phase = 2.0 * np.pi * (self.day / self.episode_days)
        mean_temp = 25.0 + 3.0 * np.sin(seasonal_phase - 0.8)
        temperature = float(np.clip(self._rng.normal(mean_temp, 2.8), 15.0, 36.0))

        # Most days are dry/light-rain days; occasional wet days create uncertainty.
        rain_probability = 0.32 + 0.08 * np.sin(seasonal_phase)
        if self._rng.random() < rain_probability:
            rainfall = float(self._rng.gamma(shape=1.8, scale=4.0))
        else:
            rainfall = 0.0
        return min(rainfall, 20.0), temperature

    def _get_obs(self) -> np.ndarray:
        stage = self._stage()
        return np.asarray(
            [
                self.soil_moisture_mm / self.SOIL_CAPACITY_MM,
                self.reservoir_mm / self.RESERVOIR_CAPACITY_MM,
                self.crop_health,
                self.day / self.episode_days,
                self.previous_rain_mm / 20.0,
                (self.temperature_c - 10.0) / 30.0,
                min(self.stress_days / 14.0, 1.0),
                min(self.cumulative_rain_mm / 600.0, 1.0),
                self.previous_action / 4.0,
                stage / 3.0,
            ],
            dtype=np.float32,
        ).clip(0.0, 1.0)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        options = options or {}

        # Randomized initial conditions allow generalization tests on unseen starts.
        self.soil_moisture_mm = float(
            options.get("initial_soil_moisture_mm", self._rng.uniform(38.0, 68.0))
        )
        self.reservoir_mm = float(options.get("initial_reservoir_mm", self._rng.uniform(105, 160)))
        self.soil_moisture_mm = float(np.clip(self.soil_moisture_mm, 0.0, self.SOIL_CAPACITY_MM))
        self.reservoir_mm = float(np.clip(self.reservoir_mm, 0.0, self.RESERVOIR_CAPACITY_MM))

        self.day = 0
        self.crop_health = float(options.get("initial_crop_health", 0.95))
        self.crop_health = float(np.clip(self.crop_health, 0.5, 1.0))
        self.cumulative_rain_mm = 0.0
        self.previous_rain_mm = 0.0
        self.temperature_c = 25.0
        self.stress_days = 0
        self.previous_action = 0
        self.yield_factor_sum = 0.0
        self.total_irrigation_mm = 0.0
        self._last_reward = 0.0
        self._last_info = {}

        observation = self._get_obs()
        info = {
            "day": self.day,
            "crop_stage": self._stage(),
            "soil_moisture_mm": self.soil_moisture_mm,
            "reservoir_mm": self.reservoir_mm,
        }
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected an integer from 0 to 4")
        if self.day >= self.episode_days:
            raise RuntimeError("Episode is finished. Call reset() before calling step() again.")

        stage = self._stage()
        requested_water = float(self.ACTION_WATER_MM[int(action)])
        irrigation = min(requested_water, self.reservoir_mm)
        self.reservoir_mm -= irrigation
        self.total_irrigation_mm += irrigation

        rainfall, temperature = self._weather()
        self.previous_rain_mm = rainfall
        self.temperature_c = temperature
        self.cumulative_rain_mm += rainfall

        # Daily crop water demand rises around flowering and falls near maturity.
        stage_demand = (2.8, 3.5, 4.0, 2.7)[stage]
        evapotranspiration = stage_demand * (0.8 + max(temperature - 18.0, 0.0) / 30.0)
        water_before_drainage = self.soil_moisture_mm + rainfall + irrigation - evapotranspiration
        drainage = max(0.0, water_before_drainage - self.SOIL_CAPACITY_MM)
        self.soil_moisture_mm = float(
            np.clip(water_before_drainage - drainage, 0.0, self.SOIL_CAPACITY_MM)
        )

        target = self.TARGET_MOISTURE[stage]
        lower_healthy_bound = target - 18.0
        upper_healthy_bound = target + 20.0
        water_stress = float(np.clip((lower_healthy_bound - self.soil_moisture_mm) / 35.0, 0.0, 1.0))
        waterlogging = float(
            np.clip((self.soil_moisture_mm - upper_healthy_bound) / 25.0, 0.0, 1.0)
        )

        if water_stress > 0.05:
            self.stress_days += 1
        else:
            self.stress_days = max(0, self.stress_days - 1)

        # Health changes persist across days; prolonged stress has a larger effect.
        health_change = 0.004 - 0.020 * water_stress - 0.010 * waterlogging
        if self.stress_days >= 5:
            health_change -= 0.004
        self.crop_health = float(np.clip(self.crop_health + health_change, 0.0, 1.0))

        temperature_suitability = float(np.clip(1.0 - abs(temperature - 25.0) / 18.0, 0.2, 1.0))
        moisture_suitability = float(np.clip(1.0 - water_stress * 0.9 - waterlogging * 0.35, 0.0, 1.0))
        daily_yield_factor = moisture_suitability * temperature_suitability * self.crop_health
        self.yield_factor_sum += daily_yield_factor
        self.previous_action = int(action)
        self.day += 1

        # Reward: sustain productive crop conditions while conserving scarce water.
        reward = (
            0.12 * daily_yield_factor
            - 0.025 * irrigation
            - 0.18 * water_stress
            - 0.08 * waterlogging
        )

        terminated = self.crop_health <= 0.05
        truncated = self.day >= self.episode_days and not terminated
        yield_score = 100.0 * self.yield_factor_sum / max(self.day, 1)
        if terminated:
            reward -= 4.0
        if truncated:
            reward += 4.0 * (yield_score / 100.0) + 1.5 * self.crop_health

        self._last_reward = float(reward)
        info = {
            "day": self.day,
            "crop_stage": self._stage(),
            "rainfall_mm": rainfall,
            "temperature_c": temperature,
            "irrigation_mm": irrigation,
            "soil_moisture_mm": self.soil_moisture_mm,
            "reservoir_mm": self.reservoir_mm,
            "crop_health": self.crop_health,
            "water_stress": water_stress,
            "waterlogging": waterlogging,
            "daily_yield_factor": daily_yield_factor,
            "yield_score": yield_score,
            "total_irrigation_mm": self.total_irrigation_mm,
            "terminated_early": terminated,
        }
        self._last_info = info
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    def render(self) -> np.ndarray | None:
        """Render a simple 2D farm dashboard; install pygame via project dependencies."""
        if self.render_mode is None:
            return None
        import pygame

        if self._pygame is None:
            pygame.init()
            self._pygame = pygame
            self._screen = pygame.display.set_mode((900, 560))
            pygame.display.set_caption("Smart Irrigation RL Simulator")
            self._clock = pygame.time.Clock()

        screen = self._screen
        font = pygame.font.SysFont("Arial", 23)
        small = pygame.font.SysFont("Arial", 17)
        screen.fill((235, 242, 231))

        # Plot panel and crop rows.
        pygame.draw.rect(screen, (215, 231, 205), (35, 110, 535, 360), border_radius=18)
        moisture = self.soil_moisture_mm / self.SOIL_CAPACITY_MM
        crop_green = int(90 + 130 * self.crop_health)
        for row in range(5):
            for col in range(9):
                x, y = 65 + col * 54, 145 + row * 62
                pygame.draw.line(screen, (115, 91, 61), (x - 10, y + 20), (x + 24, y + 20), 4)
                pygame.draw.ellipse(screen, (40, crop_green, 65), (x, y - 15, 18, 30))

        def label(text: str, x: int, y: int, color: tuple[int, int, int] = (35, 45, 35)) -> None:
            screen.blit(font.render(text, True, color), (x, y))

        label("RWANDA MAIZE PLOT", 45, 35)
        label(f"Day: {self.day}/{self.episode_days}", 620, 112)
        label(f"Crop health: {self.crop_health * 100:.1f}%", 620, 158)
        label(f"Soil moisture: {self.soil_moisture_mm:.1f} mm", 620, 204)
        label(f"Water reservoir: {self.reservoir_mm:.1f} mm", 620, 250)
        label(f"Total irrigation: {self.total_irrigation_mm:.1f} mm", 620, 296)
        label(f"Previous rainfall: {self.previous_rain_mm:.1f} mm", 620, 342)
        label(f"Temperature: {self.temperature_c:.1f} C", 620, 388)
        label(f"Last reward: {self._last_reward:.3f}", 620, 434)
        pygame.draw.rect(screen, (190, 205, 180), (620, 490, 220, 18), border_radius=8)
        pygame.draw.rect(screen, (50, 145, 75), (620, 490, int(220 * moisture), 18), border_radius=8)
        screen.blit(small.render("Soil water level", True, (35, 45, 35)), (620, 465))

        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])
        if self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(screen).transpose(1, 0, 2)
        return None

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.quit()
            self._pygame = None
            self._screen = None
            self._clock = None
