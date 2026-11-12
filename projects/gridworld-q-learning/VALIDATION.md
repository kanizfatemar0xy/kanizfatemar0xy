# Verification and Repair Notes

Project owner: **Kaniz Fatema Roxy**. The input was `rl_qlearning_Nov2025.py`, a 4×4 deterministic GridWorld Q-learning exercise.

## Confirmed corrections

- Misleading “5×5” comment replaced by a documented **4×4** environment with 16 states.
- Rewards documented correctly: goal +10, holes -10, otherwise -0.01 per step (step penalty added intentionally to discourage stalling).
- Terminal-state Q updates use `target = reward` (they do **not** bootstrap future Q values after reaching a hole or the goal).
- `numpy.random.default_rng(seed)` is used for reproducible random exploration, including fair random tie-breaking during training. Evaluation is deterministic.
- Inputs, invalid actions, terminal transitions, hyperparameters and rollout dimensions are validated.
- Exploratory training success is reported separately from final greedy evaluation; fixed-environment rollouts are not presented as independent samples.
- Independent breadth-first search verifies the six-move shortest safe route without being used by the learning algorithm.
- Added unit tests, CLI tests, a 30-seed reproducibility benchmark, a random-policy baseline, and optional training plots.

## Actual executed checks

- Python source compilation and **17 unit/integration tests passed**.
- Q-learning trained for **3,000 episodes per seed** on **30 independent seeds**: **30/30** final greedy policies reached the goal in **six moves**.
- One reproducible agent, seed `42`: greedy path `0 → 1 → 2 → 6 → 10 → 14 → 15` (six moves, no holes).
- Seed `42`: exploratory success in last 100 episodes = **96%**.
- Random policy using RNG seed `2025`: **29 goals in 2,000 episodes**, or **1.45%**.
- Training curve PNG and JSON evaluation report generated locally.

This benchmark measures reliability of training with different seeds for **one fixed deterministic map and one fixed start**, not performance across unseen environments. The random-policy denominator differs from the Q-learning seed denominator, so the figures are not a formal controlled comparison. No guarantee is made for arbitrary hyperparameters or unseen grids.

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python rl_qlearning.py --episodes 3000 --seed 42 --benchmark-seeds 30 --report-json outputs/evaluation.json
```

For the plot, additionally install `requirements-plot.txt` and pass `--plot outputs/training_curve.png`.