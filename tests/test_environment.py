import numpy as np

from environment.custom_env import SmartIrrigationEnv


def test_reset_observation_is_valid():
    env = SmartIrrigationEnv()
    obs, info = env.reset(seed=123)
    assert env.observation_space.contains(obs)
    assert obs.shape == (10,)
    assert info["day"] == 0


def test_step_changes_persistent_state_and_returns_gymnasium_tuple():
    env = SmartIrrigationEnv()
    env.reset(seed=123)
    before_reservoir = env.reservoir_mm
    obs, reward, terminated, truncated, info = env.step(4)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert env.reservoir_mm < before_reservoir
    assert info["irrigation_mm"] > 0
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_seed_reproduces_weather_and_transition():
    env_a, env_b = SmartIrrigationEnv(), SmartIrrigationEnv()
    env_a.reset(seed=88)
    env_b.reset(seed=88)
    result_a = env_a.step(2)
    result_b = env_b.step(2)
    np.testing.assert_allclose(result_a[0], result_b[0])
    assert result_a[1] == result_b[1]
    assert result_a[4]["rainfall_mm"] == result_b[4]["rainfall_mm"]


def test_episode_truncates_at_configured_horizon():
    env = SmartIrrigationEnv(episode_days=20)
    env.reset(seed=2)
    for _ in range(20):
        _, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            break
    assert terminated or truncated


def test_invalid_action_is_rejected():
    env = SmartIrrigationEnv()
    env.reset(seed=1)
    try:
        env.step(8)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid action to raise ValueError")
