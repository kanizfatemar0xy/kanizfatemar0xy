"""Full notebook execution on synthetic data; NOT scientific model verification."""
import runpy
import tempfile
from pathlib import Path
from shutil import copy2
import nbformat
import sys
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
fixture = runpy.run_path(str(root / 'tests/test_weather_project.py'))['raw_fixture']
with tempfile.TemporaryDirectory(prefix='london_weather_smoke_') as folder:
    home = Path(folder)
    copy2(root / 'DataScienceReport_Fixed.ipynb', home / 'DataScienceReport_Fixed.ipynb')
    copy2(root / 'weather_project.py', home / 'weather_project.py')
    (home / 'data').mkdir()
    synthetic = fixture('1979-01-01', 16436)
    synthetic.to_csv(home / 'data/london_weather_data_1979_to_2023.csv', index=False)
    nb = nbformat.read(home/'DataScienceReport_Fixed.ipynb', as_version=4)
    print('Executing notebook on 16,436 SYNTHETIC rows (not actual observations)',flush=True)
    client = NotebookClient(nb, timeout=180, kernel_name='python3',
                            resources={'metadata':{'path':str(home)}}, allow_errors=False)
    result = client.execute()
    count = sum(c.cell_type=='code' for c in result.cells)
    assert all(o.output_type!='error' for c in result.cells if c.cell_type=='code' for o in c.outputs)
    print(f'PASS: executed all {count} code cells and {len(result.cells)} total notebook cells.',flush=True)
    print('No synthetic notebook outputs or CSV are retained in the deliverables.',flush=True)