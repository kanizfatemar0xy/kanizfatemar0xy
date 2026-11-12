"""Check that a new user can run the CLI and produce machine-readable output."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TestCLI(unittest.TestCase):
    def test_cli_generates_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_file = Path(temp_dir) / "report.json"
            result = subprocess.run(
                [sys.executable, str(ROOT / "rl_qlearning.py"), "--episodes", "150",
                 "--seed", "42", "--benchmark-seeds", "0",
                 "--random-baseline-episodes", "50", "--report-json", str(report_file)],
                cwd=ROOT, capture_output=True, text=True, check=True,
            )
            self.assertIn("Greedy evaluation:", result.stdout)
            data = json.loads(report_file.read_text(encoding="utf-8"))
            self.assertEqual(data["config"]["episodes"], 150)
            self.assertEqual(data["random_policy_baseline"]["episodes"], 50)
            self.assertEqual(data["bfs_shortest_path"][0], 0)
            self.assertEqual(data["bfs_shortest_path"][-1], 15)

    def test_cli_rejects_zero_episodes(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "rl_qlearning.py"), "--episodes", "0"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("episodes and max-steps must be positive", result.stderr)


if __name__ == "__main__":
    unittest.main()