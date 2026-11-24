# Smart Irrigation Reinforcement Learning Project

This project compares four reinforcement learning methods for irrigation control in a simulated maize field: **DQN, REINFORCE, PPO and A2C**. Each agent makes one irrigation decision per day during a 90-day growing season.

The project is a simulation. It is not connected to real farm equipment, sensors or live weather services.

## Environment

The custom environment uses the Gymnasium interface. The agent receives a 10-value observation and chooses one of five discrete actions.

| Action | Irrigation amount |
|---:|---:|
| 0 | 0 mm |
| 1 | 2 mm |
| 2 | 5 mm |
| 3 | 8 mm |
| 4 | 12 mm |

The observation includes soil moisture, reservoir level, crop health, season progress, rainfall, temperature, water-stress history, cumulative rainfall, the previous action and crop stage. The values are normalized to the range `[0, 1]` and stored as `float32` values.

The reward encourages crop growth and penalizes irrigation use, water stress and waterlogging. Crop failure receives an extra penalty, while completing the season gives a harvest bonus.

## Requirements and setup

Use Python 3.11 and [uv](https://docs.astral.sh/uv/) to install the dependencies:

```bash
uv sync --extra dev
```

Run the environment tests:

```bash
uv run pytest
```

Run the basic simulator:

```bash
uv run main.py
```

`main.py --render` opens the environment visualization using a random-action smoke test. It does not load a trained model.

## Run the trained DQN agent

The best DQN checkpoint from the experiments is:

`models/dqn/dqn_experiment_04_gamma_095.pt`

Run the trained agent with the Pygame visualization:

```bash
uv run python play.py --seed 20000
```

Run the same episode in the terminal without the window:

```bash
uv run python play.py --no-render --seed 20000
```

The playback script loads the saved DQN checkpoint, selects the action with the highest predicted Q-value, and runs the 90-day episode. It prints the action, irrigation applied, soil moisture, reservoir level, crop health and reward for each day. The Pygame view shows the field during the episode. The final screen stays open for five seconds by default; press `Esc` or close the window to stop playback.

You can choose another checkpoint or change how long the final screen stays open:

```bash
uv run python play.py --model models/dqn/dqn_experiment_04_gamma_095.pt --seed 20000 --hold-seconds 10
```

## Training

Each algorithm has a separate training module. For a new 120-episode run, use:

```bash
uv run python -m training.dqn_training --episodes 120 --run-name dqn_new_run
uv run python -m training.reinforce_training --episodes 120 --run-name reinforce_new_run
uv run python -m training.ppo_training --episodes 120 --run-name ppo_new_run
uv run python -m training.a2c_training --episodes 120 --run-name a2c_new_run
```

Training metrics are saved under `logs/` and trained checkpoints are saved under `models/`.

### Hyperparameter sweeps

The sweep scripts run the configured hyperparameter experiments and save summary CSV files under `logs/`:

```bash
uv run python -m training.dqn_sweep --episodes 120
uv run python -m training.reinforce_sweep --episodes 120
uv run python -m training.ppo_sweep --episodes 120
uv run python -m training.a2c_sweep --episodes 120
```

The completed project contains ten experiment records for each algorithm, for a total of 40 experiments. The default DQN experiment was trained separately and included in the combined DQN summary.

## Analysis and generalization

After the four experiment summary CSV files are available, create the combined analysis tables and figures:

```bash
uv run python analysis/generate_analysis.py
```

The analysis outputs are saved in `analysis/outputs/`.

Evaluate the selected best checkpoints on 50 unseen environment seeds:

```bash
uv run python -m analysis.generalization_evaluation
```

The generalization results include episode-level results, a summary table, paired comparisons and the evaluation protocol. The four selected checkpoints are:

| Algorithm | Checkpoint |
|---|---|
| DQN | `models/dqn/dqn_experiment_04_gamma_095.pt` |
| REINFORCE | `models/pg/reinforce_experiment_01_default.pt` |
| PPO | `models/pg/ppo_experiment_01_default.pt` |
| A2C | `models/pg/a2c_experiment_03_lr_0006.pt` |

## Project structure

```text
smart-irrigation-rl-summative/
├── README.md
├── pyproject.toml
├── uv.lock
├── main.py
├── play.py
├── environment/
├── training/
├── analysis/
│   └── outputs/
├── models/
├── logs/
├── assets/
└── tests/
```

## Reproducibility and limitations

- The completed hyperparameter experiments used seed 7.
- The generalization evaluation used 50 unseen seeds (20000-20049), shared by all four methods.
- Keep `pyproject.toml` and `uv.lock` in the repository so the dependencies can be installed again.
- Keep the model checkpoints and experiment results needed to review the comparisons.
- Weather, water balance and crop responses are simulated. The results show performance in this environment and do not prove performance on a real farm.
