"""Offline, deterministic correctness and reproducibility checks."""
import unittest

import numpy as np

from rl_qlearning import (
    GridWorld,
    TrainingConfig,
    benchmark_training_seeds,
    epsilon_greedy,
    greedy_action,
    render_policy,
    random_policy_baseline,
    rollout,
    shortest_safe_path,
    td_target,
    train,
)


class TestGridWorld(unittest.TestCase):
    def setUp(self):
        self.env = GridWorld()

    def test_geometry(self):
        self.assertEqual(self.env.n_states, 16)
        self.assertEqual(self.env.n_actions, 4)
        self.assertEqual(self.env.reset(), 0)

    def test_boundary_actions_remain_in_cell(self):
        for action in (0, 2):
            self.assertEqual(self.env.transition(0, action), (0, -0.01, False))
        self.assertEqual(self.env.transition(3, 3), (3, -0.01, False))
        self.assertEqual(self.env.transition(8, 2), (8, -0.01, False))

    def test_destination_and_terminal_rewards(self):
        self.assertEqual(self.env.transition(4, 3), (5, -10.0, True))
        self.assertEqual(self.env.transition(14, 3), (15, 10.0, True))
        self.assertEqual(self.env.transition(0, 1), (4, -0.01, False))

    def test_terminal_transitions_and_invalid_inputs_rejected(self):
        for state in (5, 15):
            with self.assertRaises(ValueError):
                self.env.transition(state, 0)
        with self.assertRaises(ValueError):
            self.env.transition(16, 0)
        with self.assertRaises(ValueError):
            self.env.transition(0, 4)
        with self.assertRaises(ValueError):
            self.env.transition(-1, 1)

    def test_environment_validation(self):
        with self.assertRaises(ValueError):
            GridWorld(goal_state=5)
        with self.assertRaises(ValueError):
            GridWorld(size=1)
        with self.assertRaises(ValueError):
            GridWorld(hole_states=frozenset({0}))

    def test_bfs_independent_shortest_safe_path(self):
        path = shortest_safe_path(self.env)
        self.assertIsNotNone(path)
        self.assertEqual(len(path) - 1, 6)
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], 15)
        self.assertFalse(set(path).intersection(self.env.hole_states))
        for a, b in zip(path, path[1:]):
            self.assertTrue(any(self.env.transition(a, act)[0] == b for act in range(4)))


class TestLearning(unittest.TestCase):
    def setUp(self):
        self.env = GridWorld()
        self.cfg = TrainingConfig(episodes=3000, seed=42)

    def test_terminal_td_does_not_bootstrap(self):
        future_q = np.asarray([50., 60., 70., 80.])
        self.assertEqual(td_target(-10., future_q, True, 0.97), -10.)
        self.assertEqual(td_target(10., future_q, True, 0.97), 10.)
        self.assertAlmostEqual(td_target(-0.01, future_q, False, 0.97), 77.59)

    def test_epsilon_greedy_explores_all_actions(self):
        rng = np.random.default_rng(4)
        moves = {epsilon_greedy(np.array([0., 0., 0., 0.]), 1.0, rng) for _ in range(100)}
        self.assertEqual(moves, {0, 1, 2, 3})
        self.assertEqual(greedy_action(np.array([0., 2., 2., 1.])), 1)

    def test_training_configuration_validation(self):
        for kwargs in ({"episodes": 0}, {"max_steps": 0}, {"learning_rate": -0.01}, {"discount": 1.1}, {"epsilon_min": 1.1}, {"epsilon_decay": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrainingConfig(**kwargs)

    def test_reproducibility(self):
        one = train(self.env, self.cfg)
        two = train(self.env, self.cfg)
        np.testing.assert_array_equal(one.q_table, two.q_table)
        self.assertEqual(one.episode_successes, two.episode_successes)
        self.assertEqual(one.episode_rewards, two.episode_rewards)
        self.assertEqual(len(one.episode_rewards), self.cfg.episodes)

    def test_greedy_rollout_safe_and_shortest_seed_42(self):
        agent = train(self.env, self.cfg)
        result = rollout(self.env, agent.q_table)
        self.assertEqual(result["outcome"], "goal")
        self.assertEqual(result["steps"], 6)
        self.assertFalse(set(result["states"]).intersection(self.env.hole_states))
        self.assertIn("S", render_policy(self.env, agent.q_table))
        self.assertIn("G", render_policy(self.env, agent.q_table))

    def test_rollout_timeout_on_untrained_q_table(self):
        result = rollout(self.env, np.zeros((16, 4)), max_steps=3)
        self.assertEqual(result["outcome"], "timeout")
        self.assertFalse(result["success"])
        self.assertEqual(result["states"], [0, 0, 0, 0])

    def test_bad_q_table_shape_rejected(self):
        with self.assertRaises(ValueError):
            rollout(self.env, np.zeros((15, 4)))
        with self.assertRaises(ValueError):
            rollout(self.env, np.zeros((16, 4)), max_steps=0)

    def test_seeded_random_baseline_is_reproducible(self):
        one = random_policy_baseline(self.env, episodes=200, max_steps=30, seed=2025)
        two = random_policy_baseline(self.env, episodes=200, max_steps=30, seed=2025)
        self.assertEqual(one, two)
        self.assertEqual(sum(one["outcomes"].values()), 200)
        self.assertGreaterEqual(one["goal_fraction"], 0)
        self.assertLessEqual(one["goal_fraction"], 1)

    def test_seed_sweep_reliability(self):
        benchmark = benchmark_training_seeds(self.env, self.cfg, n_seeds=10)
        self.assertEqual(benchmark["training_seeds"], 10)
        self.assertEqual(benchmark["successful_policies"], 10)
        self.assertEqual(benchmark["shortest_path_policies"], 10)
        self.assertEqual(benchmark["shortest_path_steps"], 6)


if __name__ == "__main__":
    unittest.main()