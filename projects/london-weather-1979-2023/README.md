# London Weather Analysis & Next-Day Forecasting (1979–2023)

**Python · pandas · NumPy · scikit-learn · Matplotlib · Jupyter**

A reproducible data science portfolio project analysing **daily Heathrow/London weather observations** from 1979–2023, then evaluating simple **next-day sunshine and rainfall prediction** methods using a chronological holdout. This is a fully refactored version of a historical Jupyter notebook, with unit and integration tests, explicit variable units, data quality checks, and documented limitations.

[View the notebook](DataScienceReport_Fixed.ipynb) · [Source code](weather_project.py) · [Unit tests](tests/test_weather_project.py) · [Methodology and test report](REVIEW_AND_TEST_REPORT.md)

## Questions explored

1. Seasonality of temperatures, including June 1979–1990.
2. Complete-decade temperatures, annual temperature and precipitation.
3. Monthly climatology for temperature, rainfall and sunshine.
4. Comparison of monthly rainfall totals: 1979–2018 versus 2019–2022.
5. Daily rainfall associations with humidity, cloud cover, pressure and temperature (2019–2022).
6. Next-day sunshine and precipitation forecasts using only past weather observations.

## Data and provenance

**Original observations are not included in this ZIP or repository.** The notebook automatically downloads and caches `data/london_weather_data_1979_to_2023.csv` from [pc1991/London-Weather-Data-From-1979-To-2023](https://github.com/pc1991/London-Weather-Data-From-1979-To-2023) when run with an internet connection. That public repository attributes its data to the **European Climate Assessment & Dataset (ECA&D)**, Heathrow station ID 1860, and describes 16,436 daily records covering 1979-01-01 to 2023-12-31.

The dataset is **one station, not a London-wide average**. Numeric values arrive in mixed original units: `TG/TX/TN` in 0.1°C, `RR` in 0.1 mm, `SS` in 0.1 hours, `PP` in 0.1 hPa; `HU`, `QQ`, `CC` and `SD` are not scaled. Each observation has a quality flag (0 valid, 1 suspect, 9 missing). The project filters suspect/missing measurements, divides only the tenths-unit columns once, and does not invent observations to fill the descriptive plots.

**Use terms:** ECA&D data policy says their observations are for non-commercial research and education. Before any redistribution or commercial use, consult the [original ECA&D data policy](https://www.ecad.eu/documents/ECAD_datapolicy.pdf). See [data/README.md](data/README.md) for source details and attribution.

## Run locally

You need **Python 3.11+**, an internet connection for the first data download, and a Jupyter-capable Python environment.

```bash
# Optional, recommended:
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_*.py' -v
jupyter lab DataScienceReport_Fixed.ipynb
```

In Jupyter select **Run → Restart Kernel and Run All Cells**. The notebook will download the original CSV into `data/` if needed. To work **offline**, download [the CSV yourself](https://raw.githubusercontent.com/pc1991/London-Weather-Data-From-1979-To-2023/main/london_weather_data_1979_to_2023.csv) and save it to `data/london_weather_data_1979_to_2023.csv` before running.

**Important:** The notebook depends on `weather_project.py`; opening the `.ipynb` by itself without the accompanying project folder will not work. Use the ZIP package as a unit.

## Modelling and validation

| Stage | Design |
| --- | --- |
| Forecast target | `RR[t]` or `SS[t]` for a day **t** |
| Available predictors | Day-of-year seasonality and measurements from **t−1 or earlier** |
| Training period | 1979–2018 |
| Validation period | 2019–2020 (choose lowest MAE method) |
| Held-out test | 2021–2023 (evaluated after selection) |
| Baselines | Previous day and training-only monthly climatology |
| Candidate models | Regularised linear regression (Ridge), Random Forest |
| Missing predictors | Median imputer inside a training-only scikit-learn pipeline |
| Metrics | MAE, RMSE, R² — including negative R² if observed |

This is a **historical backtest** based on true prior-day observations, not a production forecasting service. Test metrics and plots will be produced **only when the original CSV is run**. No unverified accuracy or forecasting-skill numbers are claimed here.

## File guide

```text
DataScienceReport_Fixed.ipynb     Corrected, documented analysis notebook
weather_project.py               Data quality, analysis and forecast functions
data/README.md                   Source URL, citation and usage restrictions
requirements.txt                 Python dependencies
README.md                        Project overview and reproduction guide
REVIEW_AND_TEST_REPORT.md        Changes, test evidence and limitations
tests/test_weather_project.py     Unit/regression tests using synthetic data
tests/test_notebook.py           Notebook syntax and quality checks
tests/run_notebook_smoke.py      Runs all notebook cells on SYNTHETIC inputs
.github/workflows/test.yml       Offline GitHub Actions test configuration
```

## What the tests prove — and what they do not

**Automated offline tests** verify unit conversion, quality flags, date integrity, input leakage prevention, time splits, summary completeness and both modelling code paths. A **full notebook smoke test** executes every code cell with generated synthetic data, confirming executable integration. **Neither test proves actual Heathrow historical results**: they require downloading and running the authentic CSV. The original saved outputs were discarded because their repeated unit conversion invalidated them.

## References

- [Dataset mirror and metadata](https://github.com/pc1991/London-Weather-Data-From-1979-To-2023)
- [ECA&D data dictionary](https://www.ecad.eu/dailydata/datadictionaryelement.php)
- [ECA&D data policy](https://www.ecad.eu/documents/ECAD_datapolicy.pdf)
- Klein Tank, A. M. G., et al. (2002). *Daily dataset of 20th-century surface air temperature and precipitation series for the European Climate Assessment*. International Journal of Climatology, **22**, 1441–1453.

---

**Portfolio intent:** Data Analyst · Junior Data Scientist · Research Assistant — with accurate methods, clear code and reproducibility rather than unsupported model claims.