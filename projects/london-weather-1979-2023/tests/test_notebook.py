"""Portfolio notebook quality checks. Do not execute real data here."""
import ast
from pathlib import Path
import unittest
import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "DataScienceReport_Fixed.ipynb"


class TestNotebookStructure(unittest.TestCase):
    def test_notebook_is_valid_and_code_parses(self):
        nb = nbformat.read(NOTEBOOK, as_version=4)
        nbformat.validate(nb)
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        self.assertGreaterEqual(len(code_cells), 10)
        for cell in code_cells:
            ast.parse(cell.source)
            self.assertEqual(cell.outputs, [], "no stale outputs in publication")
            self.assertIsNone(cell.execution_count)

    def test_no_original_data_leakage_patterns(self):
        nb = nbformat.read(NOTEBOOK, as_version=4)
        code_text = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
        self.assertNotIn("drive.mount", code_text)
        self.assertNotIn("train_test_split", code_text)
        self.assertNotIn("cross_val_score", code_text)
        self.assertNotIn("google.colab", code_text)

    def test_metadata_project_files(self):
        for name in ("weather_project.py", "README.md", "requirements.txt", "data/README.md"):
            self.assertTrue((ROOT / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()