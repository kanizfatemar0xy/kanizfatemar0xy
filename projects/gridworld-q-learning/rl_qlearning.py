"""Reproducible tabular Q-learning for a small deterministic GridWorld.

Environment (4 x 4):
    S . . .
    . H . H
    . . . H
    H . . G

This is an educational reinforcement learning example, not an OpenAI Gym game.
The agent receives the current state and learns solely from interactions. BFS is
used only after training as an independent shortest-path verification baseline.

Run: python rl_qlearning.py --seed 42 --episodes 3000 --benchmark-seeds 30
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

ACTION_LABELS = ("UP", "DOWN", "LEFT", "RIGHT")
ACTION_ARROWS = ("↑", "↓", "←", "→")


@dataclass(frozen=True)
class GridWorld:
    """Deterministic 4x4 grid; terminal cells are holes or the goal."""

    size: int = 4
    start_state: int = 0
    goal_state: int = 15
    hole_states: frozenset[int] = frozenset({5, 7, 11, 12})
    step_reward: float = -0.01
    goal_reward: float = 10.0
    hole_reward: float = -10.0

    def __post_init__(self) -> None:
        if self.size < 2:
            raise ValueError("size must be at least 2")
        if not (0 <= self.start_state < self.n_states and 0 <= self.goal_state < self.n_states):
            raise ValueError("start/goal outside the grid")
        if self.start_state == self.goal_state:
            raise ValueError("start and goal must be different")
        if any(not (0 <= h < self.n_states) for h in self.hole_states):
            raise ValueError("hole outside the grid")
        if self.start_state in self.hole_states or self.goal_state in self.hole_states:
            raise ValueError("start/goal cannot be a hole")

    @property
    def n_states(self) -> int:
        return self.size * self.size

    @property
    def n_actions(self) -> int:
        return len(ACTION_LABELS)

    def reset(self) -> int:
        return self.start_state

    def transition(self, state: int, action: int) -> tuple[int, float, bool]:
        """Return (next_state, reward, terminated); boundaries keep the position."""
        if not 0 <= state < self.n_states:
            raise ValueError(f"state outside grid: {state}")
        if action not in range(self.n_actions):
            raise ValueError(f"invalid action: {action}")
        if state == self.goal_state or state in self.hole_states:
            raise ValueError("cannot step from a terminal state; call reset()")

        row, col = divmod(state, self.size)
        if action == 0:
            row = max(0, row - 1)
        elif action == 1:
            row = min(self.size - 1, row + 1)
        elif action == 2:
            col = max(0, col - 1)
        else:
            col = min(self.size - 1, col + 1)
        next_state = row * self.size + col
        if next_state == self.goal_state:
            return next_state, self.goal_reward, True
        if next_state in self.hole_states:
            return next_state, self.hole_reward, True
        return next_state, self.step_reward, False


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 3000
    max_steps: int = 64
    learning_rate: float = 0.2
    discount: float = 0.97
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.997
    seed: int = 42

    def __post_init__(self) -> None:
        if self.episodes < 1 or self.max_steps < 1:
            raise ValueError("episodes and max_steps must be positive")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0 <= self.discount <= 1:
            raise ValueError("discount must be in [0, 1]")
        if not 0 <= self.epsilon_min <= self.epsilon_start <= 1:
            raise ValueError("epsilon bounds must be in [0, 1] and min <= start")
        if not 0 < self.epsilon_decay <= 1:
            raise ValueError("epsilon_decay must be in (0, 1]")


@dataclass
class TrainingResult:
    q_table: np.ndarray
    episode_rewards: list[float]
    episode_successes: list[int]
    episode_steps: list[int]
    episode_epsilons: list[float]


def greedy_action(q_values: np.ndarray) -> int:
    """Stable deterministic tie-break at evaluation time."""
    return int(np.argmax(q_values))


def epsilon_greedy(q_values: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    """Explore uniformly; break exploitation ties randomly to avoid index bias."""
    if rng.random() < epsilon:
        return int(rng.integers(len(q_values)))
    max_value = np.max(q_values)
    tied = np.flatnonzero(np.isclose(q_values, max_value, rtol=1e-12, atol=1e-12))
    return int(rng.choice(tied))


def td_target(reward: float, next_values: np.ndarray, terminated: bool, discount: float) -> float:
    """Terminal transitions must never bootstrap off a successor Q-value."""
    return float(reward if terminated else reward + discount * np.max(next_values))


def train(env: GridWorld, config: TrainingConfig) -> TrainingResult:
    """Model-free, off-policy one-step tabular Q-learning from the fixed start."""
    rng = np.random.default_rng(config.seed)
    q = np.zeros((env.n_states, env.n_actions), dtype=float)
    rewards: list[float] = []
    successes: list[int] = []
    lengths: list[int] = []
    epsilons: list[float] = []

    for episode in range(config.episodes):
        epsilon = max(config.epsilon_min, config.epsilon_start * config.epsilon_decay ** episode)
        state = env.reset()
        episode_reward = 0.0
        reached_goal = 0
        count = 0
        for step in range(config.max_steps):
            action = epsilon_greedy(q[state], epsilon, rng)
            next_state, reward, terminated = env.transition(state, action)
            target = td_target(reward, q[next_state], terminated, config.discount)
            q[state, action] += config.learning_rate * (target - q[state, action])
            episode_reward += reward
            count = step + 1
            state = next_state
            if terminated:
                reached_goal = int(state == env.goal_state)
                break
        rewards.append(float(episode_reward))
        successes.append(reached_goal)
        lengths.append(count)
        epsilons.append(float(epsilon))

    return TrainingResult(q, rewards, successes, lengths, epsilons)


def rollout(env: GridWorld, q_table: np.ndarray, max_steps: int = 64) -> dict:
    """Single deterministic greedy evaluation trajectory from the start state.

    Since transitions and tie-breaking are deterministic, replaying the same
    policy many times does not provide independent evaluation samples.
    """
    if q_table.shape != (env.n_states, env.n_actions):
        raise ValueError("Q-table dimensions do not match the environment")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    state = env.reset()
    path = [state]
    actions: list[int] = []
    total_reward = 0.0
    outcome = "timeout"
    for _ in range(max_steps):
        action = greedy_action(q_table[state])
        state, reward, terminated = env.transition(state, action)
        actions.append(action)
        path.append(state)
        total_reward += reward
        if terminated:
            outcome = "goal" if state == env.goal_state else "hole"
            break
    return {
        "outcome": outcome,
        "success": outcome == "goal",
        "steps": len(actions),
        "states": path,
        "actions": [ACTION_LABELS[a] for a in actions],
        "total_reward": round(total_reward, 6),
    }


def shortest_safe_path(env: GridWorld) -> list[int] | None:
    """Breadth-first search for an independently verifiable shortest safe route.

    BFS is never used for choosing actions or modifying the learned Q-table.
    """
    queue = deque([(env.start_state, [env.start_state])])
    visited = {env.start_state}
    while queue:
        state, path = queue.popleft()
        if state == env.goal_state:
            return path
        for action in range(env.n_actions):
            neighbor, _, _ = env.transition(state, action)
            if neighbor in env.hole_states or neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append((neighbor, path + [neighbor]))
    return None


def benchmark_training_seeds(env: GridWorld, config: TrainingConfig, n_seeds: int = 30) -> dict:
    """Evaluate independently trained policies across different RNG seeds."""
    if n_seeds < 1:
        raise ValueError("n_seeds must be positive")
    shortest = shortest_safe_path(env)
    optimal_steps = None if shortest is None else len(shortest) - 1
    attempts: list[dict] = []
    for seed in range(n_seeds):
        trained = train(env, TrainingConfig(**{**asdict(config), "seed": seed}))
        result = rollout(env, trained.q_table, max_steps=config.max_steps)
        attempts.append({"seed": seed, "success": result["success"], "steps": result["steps"], "outcome": result["outcome"]})
    successes = sum(item["success"] for item in attempts)
    optimal = sum(item["success"] and item["steps"] == optimal_steps for item in attempts)
    return {
        "training_seeds": n_seeds,
        "successful_policies": successes,
        "success_rate": successes / n_seeds,
        "shortest_path_policies": optimal,
        "shortest_path_rate": optimal / n_seeds,
        "shortest_path_steps": optimal_steps,
        "seed_results": attempts,
    }


def random_policy_baseline(env: GridWorld, episodes: int = 2000, max_steps: int = 64, seed: int = 2025) -> dict:
    """Independent, seeded random-action baseline from the fixed start state."""
    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes and max_steps must be positive")
    rng = np.random.default_rng(seed)
    outcomes = {"goal": 0, "hole": 0, "timeout": 0}
    for _ in range(episodes):
        state = env.reset()
        outcome = "timeout"
        for _ in range(max_steps):
            state, _, terminated = env.transition(state, int(rng.integers(env.n_actions)))
            if terminated:
                outcome = "goal" if state == env.goal_state else "hole"
                break
        outcomes[outcome] += 1
    return {"episodes": episodes, "seed": seed, "outcomes": outcomes,
            "goal_fraction": outcomes["goal"] / episodes}


def render_policy(env: GridWorld, q_table: np.ndarray) -> str:
    """Display S, H, G and greedy direction for other cells."""
    rows = []
    for row in range(env.size):
        cells = []
        for col in range(env.size):
            state = row * env.size + col
            if state == env.start_state:
                symbol = "S"
            elif state == env.goal_state:
                symbol = "G"
            elif state in env.hole_states:
                symbol = "H"
            else:
                symbol = ACTION_ARROWS[greedy_action(q_table[state])]
            cells.append(f" {symbol} ")
        rows.append("|" + "|".join(cells) + "|")
    return "\n".join(rows)


def save_learning_curve(result: TrainingResult, filename: str | Path, window: int = 100) -> None:
    """Optional PNG diagnostic of training rewards and successes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if window < 1:
        raise ValueError("window must be positive")
    rewards = np.asarray(result.episode_rewards, dtype=float)
    successes = np.asarray(result.episode_successes, dtype=float)
    weights = np.ones(min(window, len(rewards))) / min(window, len(rewards))
    reward_avg = np.convolve(rewards, weights, mode="valid")
    success_avg = np.convolve(successes, weights, mode="valid")
    x = np.arange(len(reward_avg)) + len(weights)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(x, reward_avg)
    axes[0].set_ylabel("Rolling mean reward")
    axes[0].set_title(f"Training diagnostics (moving average, window={len(weights)} episodes)")
    axes[0].grid(alpha=0.25)
    axes[1].plot(x, success_avg)
    axes[1].set_ylabel("Rolling goal fraction")
    axes[1].set_xlabel("Training episode")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproducible Q-learning on a deterministic 4x4 GridWorld")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the displayed training run")
    parser.add_argument("--episodes", type=int, default=3000, help="Q-learning episodes per training seed")
    parser.add_argument("--max-steps", type=int, default=64, help="Maximum actions per episode")
    parser.add_argument("--benchmark-seeds", type=int, default=30, help="Number of independent training seeds; 0 disables benchmark")
    parser.add_argument("--random-baseline-episodes", type=int, default=2000, help="Seeded random-policy baseline episodes; 0 disables it")
    parser.add_argument("--plot", type=Path, help="Save a training-curve PNG (requires matplotlib)")
    parser.add_argument("--report-json", type=Path, help="Save evaluation and benchmark as JSON")
    args = parser.parse_args(argv)
    if args.benchmark_seeds < 0 or args.random_baseline_episodes < 0:
        parser.error("benchmark and random-baseline episode counts must be nonnegative")
    if args.seed < 0 or args.episodes < 1 or args.max_steps < 1:
        parser.error("seed must be nonnegative; episodes and max-steps must be positive")

    env = GridWorld()
    config = TrainingConfig(episodes=args.episodes, max_steps=args.max_steps, seed=args.seed)
    trained = train(env, config)
    evaluation = rollout(env, trained.q_table, max_steps=config.max_steps)
    ideal = shortest_safe_path(env)
    print("GridWorld: deterministic 4x4, S=0, G=15, holes=5/7/11/12")
    print(f"Training: episodes={config.episodes}, seed={config.seed}, epsilon_start={config.epsilon_start}")
    print(f"Exploratory training success (last 100 episodes): {np.mean(trained.episode_successes[-100:]):.1%}")
    print(f"Greedy evaluation: {evaluation['outcome']}, steps={evaluation['steps']}, states={evaluation['states']}")
    print(f"Independent BFS shortest safe route: {ideal} (moves={len(ideal)-1 if ideal else 'unreachable'})")
    print("Greedy policy (unvisited cells are not guaranteed to be optimal):")
    print(render_policy(env, trained.q_table))

    report = {"environment": {"size": env.size, "start": env.start_state, "goal": env.goal_state, "holes": sorted(env.hole_states)}, "config": asdict(config), "greedy_evaluation": evaluation, "bfs_shortest_path": ideal, "training_last_100_success_fraction": float(np.mean(trained.episode_successes[-100:]))}
    if args.benchmark_seeds:
        benchmark = benchmark_training_seeds(env, config, args.benchmark_seeds)
        report["benchmark"] = benchmark
        print(f"Independent training seeds: {benchmark['successful_policies']}/{benchmark['training_seeds']} reach goal; {benchmark['shortest_path_policies']}/{benchmark['training_seeds']} match the shortest path")
    if args.random_baseline_episodes:
        baseline = random_policy_baseline(env, args.random_baseline_episodes, config.max_steps)
        report["random_policy_baseline"] = baseline
        print(f"Random-policy baseline: {baseline['outcomes']['goal']}/{baseline['episodes']} reach goal ({baseline['goal_fraction']:.1%}); seeded episodes, not independent training runs")
    if args.plot:
        save_learning_curve(trained, args.plot)
        print(f"Saved training plot: {args.plot}")
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved JSON report: {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())