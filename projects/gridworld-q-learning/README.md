# GridWorld Q-Learning — A Reproducible Reinforcement Learning Agent

**Python · NumPy · Tabular reinforcement learning · Deterministic policy evaluation · Automated tests**

An educational reinforcement learning project that trains a tabular Q-learning agent to navigate a **4×4 GridWorld**, avoiding holes and reaching a goal. It extends my original `rl_qlearning_Nov2025.py` exercise with a correctly defined environment, reproducible exploration, terminal-safe temporal-difference updates, training diagnostics, automated tests, and independent shortest-path verification.

> **Scope:** A deliberately small, deterministic, fully observable environment. This is a learning and software-engineering demonstration, **not** evidence that the agent generalises to unseen maps or stochastic dynamics.

## Environment

```text
 S  .  .  .
 .  H  .  H
 .  .  .  H
 H  .  .  G
```

- **S:** Start, state `0`; **G:** goal, state `15`; **H:** terminal hole states `5, 7, 11, 12`.
- **Actions:** `0 = up`, `1 = down`, `2 = left`, `3 = right`.
- **Rewards:** goal `+10`, hole `-10`, every other move `-0.01` (including moves against walls).
- **Transitions:** deterministic; an attempted move through a boundary leaves the agent in its current state.
- **Episode end:** reaching a hole or goal, or reaching the configurable `max_steps` limit.

### What the agent learns

Q-learning updates action values from trial-and-error transitions, without receiving the optimal path or the environment's reward map as training input:

\[
Q(s,a) \leftarrow Q(s,a) + \alpha\bigl[r + \gamma(1-\mathrm{terminated}) \max_{a'}Q(s',a') - Q(s,a)\bigr].
\]

Training uses decaying epsilon-greedy exploration **with random tie-breaking** and one explicit NumPy random generator per training seed. An independent breadth-first search (BFS) computes the shortest safe path **only for verification**, not for training.

## Quick start

Requires Python **3.11+**.

```bash
python -m pip install -r requirements.txt
python rl_qlearning.py --seed 42 --episodes 3000 --benchmark-seeds 30
```

To produce a machine-readable result report:

```bash
python rl_qlearning.py --seed 42 --episodes 3000 --benchmark-seeds 30 --report-json outputs/evaluation.json
```

To plot training rewards and success fraction (optional):

```bash
python -m pip install -r requirements-plot.txt
python rl_qlearning.py --plot outputs/training_curve.png
```

To run all automated tests:

```bash
python -m unittest discover -s tests -v
```

## Measured results (fixed start, fixed map)

| Diagnostic | Result |
| --- | --- |
| **Independent training seeds** | **30/30** learned greedy policies reached the goal |
| **Shortest safe routes among those seeds** | **30/30**, each in **6 moves** |
| Greedy rollout for training seed 42 | `0 → 1 → 2 → 6 → 10 → 14 → 15` |
| Last 100 exploratory training episodes for seed 42 | 96/100 reached the goal |
| Uniform random-action baseline | 29/2,000 episodes reached the goal (1.45%), RNG seed 2025 |
| Automated offline tests | **17 passed** |

Results were generated with **3,000 training episodes per seed**, maximum **64 steps/episode**, learning rate **0.2**, discount **0.97**, starting epsilon **1.0**, minimum epsilon **0.05**, and epsilon decay multiplier **0.997**. See [`examples/evaluation_seed42.json`](examples/evaluation_seed42.json) and [`VALIDATION.md`](VALIDATION.md) for the reproducible recorded run.

**How to interpret the numbers:** The 30/30 figure measures **independently trained policies** on the **same deterministic, known map and fixed start**; it does not measure generalisation to other maps. The random-policy result uses episodes, not independent trained agents, so it is a baseline illustration, not a matched statistical comparison. Once a greedy policy is fixed, replaying the same deterministic episode does not create independent test samples. The shortest path is verified by BFS and need not be unique.

## Files

| File | Purpose |
| --- | --- |
| [`rl_qlearning.py`](rl_qlearning.py) | Environment, training, evaluation, BFS benchmark, CLI and optional training plot |
| [`tests/test_qlearning.py`](tests/test_qlearning.py) | Unit and reproducibility tests, including checks for terminal-state bootstrapping |
| [`tests/test_cli.py`](tests/test_cli.py) | CLI execution and JSON report integration checks |
| [`examples/evaluation_seed42.json`](examples/evaluation_seed42.json) | Saved, explicitly labelled evaluation and seed-sweep results |
| [`VALIDATION.md`](VALIDATION.md) | What was fixed and how it was tested |
| [`requirements.txt`](requirements.txt) | Core NumPy requirement |
| [`requirements-plot.txt`](requirements-plot.txt) | Optional Matplotlib requirement |

## Improvements over the original exercise

1. Corrected the environment description to **4×4** (the original comment incorrectly said 5×5).
2. Replaced inconsistent comments with the actual rewards and an explicit configurable step cost.
3. Updated the Bellman target to **avoid bootstrapping after terminal transitions**.
4. Added reproducible random seeds, a stable evaluator and random tie-breaking during training.
5. Added separate **training diagnostics vs. post-training evaluation**, avoiding an unverified “optimal policy” label.
6. Validated greedy paths against an independent shortest-safe-route BFS reference.
7. Added a multi-seed test, a seeded random-policy baseline, an optional training plot, JSON output, and tests.

## Limitations & next steps

This example uses only one deterministic grid and a fixed initial state. A learned arrow in a little-visited state is not necessarily optimal. The state space is small enough for a Q-table; more realistic tasks may require function approximation, randomized starts, stochastic transitions, held-out maps, and independent tuning/evaluation budgets. Those experiments have **not** been claimed as completed.

**Portfolio owner:** [Kaniz Fatema Roxy](https://github.com/kanizfatemar0xy) · [Professional profile](https://www.linkedin.com/in/kaniz-fatema-roxy)