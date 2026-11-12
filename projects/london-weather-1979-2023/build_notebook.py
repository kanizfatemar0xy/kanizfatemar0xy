"""Generate the corrected notebook. Run from project root."""
from pathlib import Path
import nbformat as nbf
from textwrap import dedent

cells=[]
def md(text):
    cells.append(nbf.v4.new_markdown_cell(dedent(text).strip()))
def code(text):
    cells.append(nbf.v4.new_code_cell(dedent(text).strip()))

md('''
# London Weather Analysis & Next-Day Forecasting (1979–2023)

**Heathrow station observations · Python · pandas · Matplotlib · scikit-learn**

> **Corrected and reproducible portfolio notebook** based on the original `DataScienceReport.ipynb` (33 cells). The repeated scaling, Google Drive dependency, uncontrolled median imputation, duplicate analysis, random time-series splits, same-day forecast leakage and obsolete saved outputs have been removed.

**Station-level scope:** This is Heathrow station weather, not an average of every location in London. The historical comparison is descriptive, not evidence of a causal climate trend or a validated operational weather forecast.
''')
md('''
## Research questions

1. **Seasonality:** How do mean temperatures vary across the calendar year, including June 1979–1990?
2. **Decadal and annual changes:** What do 1980–2019 decade summaries and 1979–2023 annual temperature and rainfall records show?
3. **Monthly patterns:** How do average monthly temperatures, rainfall and sunshine differ?
4. **Recent rainfall:** How do the monthly rainfall totals in 2019–2022 compare descriptively with 1979–2018?
5. **Weather relationships:** How is daily rainfall associated with temperature, humidity, cloud cover and pressure in 2019–2022?
6. **Prediction:** Can we predict **next-day** sunshine and rainfall using only prior-day information, outperforming simple historical baselines on a held-out 2021–2023 period?

The notebook is ordered so each code cell depends only on earlier cells. To reproduce the results, choose **Kernel → Restart & Run All**.
''')
code('''
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from weather_project import (
    DATA_URL, load_raw_weather, clean_weather_data, quality_report,
    monthly_summary, annual_summary, decadal_temperature,
    rainfall_period_comparison, next_day_frame, split_chronologically,
    evaluate_forecast, regression_metrics,
)

pd.set_option('display.max_columns', 25)
print('pandas', pd.__version__)
''')
md('''
## Data source and variable definitions

The expected file is **`data/london_weather_data_1979_to_2023.csv`**. When missing, the loader downloads and caches the published CSV from [pc1991's London Weather repository](https://github.com/pc1991/London-Weather-Data-From-1979-To-2023), which attributes the observations to [ECA&D](https://www.ecad.eu/).

| Field | Description | Source unit | Clean unit |
|---|---|---|---|
| `DATE` | Observation date | YYYYMMDD | Date |
| `TX`, `TN`, `TG` | Maximum, minimum, mean temperature | 0.1 °C | °C |
| `SS` | Sunshine duration | 0.1 hours | hours |
| `RR` | Daily precipitation | 0.1 mm | mm |
| `PP` | Atmospheric pressure | 0.1 hPa | hPa |
| `QQ` | Global radiation | W/m² | unchanged |
| `HU` | Relative humidity | % | unchanged |
| `CC` | Cloud cover | oktas | unchanged |
| `SD` | Snow depth | cm | unchanged |

**Quality control:** Matching `Q_` flags are decoded: `0` = valid; `1` = suspect; `9` = missing. Only valid readings are analysed. **Units are converted exactly once** and missing/suspect observations stay missing in climate summaries. A model may impute *features* using **training-set-only medians**, but target observations are never fabricated.

**Terms:** ECA&D's public policy limits use to non-commercial research and education. Verify rights before redistributing the raw CSV or commercialising the data. This project links to, but does not bundle, the entire dataset.
''')
code('''
raw = load_raw_weather(Path('data') / 'london_weather_data_1979_to_2023.csv')
weather = clean_weather_data(raw)
print('Raw rows:', len(raw), 'Columns:', len(raw.columns))
print('Date range:', weather['DATE'].min().date(), 'to', weather['DATE'].max().date())
print('Duplicate dates:', weather['DATE'].duplicated().sum())

# The verified upstream copy has 16,436 rows, 1979-01-01 to 2023-12-31.
if len(weather) != 16436 or weather['DATE'].min() != pd.Timestamp('1979-01-01') or weather['DATE'].max() != pd.Timestamp('2023-12-31'):
    print('WARNING: This dataset differs in size/date coverage from the verified source; review provenance.')

assert weather['RR'].dropna().ge(0).all(), 'Invalid negative rainfall after quality checks'
assert weather['SS'].dropna().between(0, 24).all(), 'Sunshine duration outside 0–24 hours'
assert weather['DATE'].is_monotonic_increasing
print('\\nFirst rows, after one-time unit conversion:')
display(weather[['DATE', 'TG', 'TX', 'TN', 'RR', 'SS', 'PP', 'HU', 'CC']].head())
''')
md('''
## Data quality audit

The source quality flags are important: removing suspect readings changes completeness, especially for mean and maximum temperature. A global median fill would artificially smooth historical weather and bias rainfall totals, so **this notebook never fills missing descriptive observations**.

Annual and monthly summaries mask statistics when fewer than **90% of expected calendar days** have valid measurements. This is a documented analysis choice, not a universal scientific threshold.
''')
code('''
quality = quality_report(raw, weather)
print('Quality flags and available observations by variable:')
display(quality)
monthly = monthly_summary(weather, minimum_coverage=0.90)
annual = annual_summary(weather, minimum_coverage=0.90)
print('\\nYears with incomplete temperature measurements (annual temp masked):')
display(annual.loc[annual['mean_temp'].isna(), ['temp_days', 'expected_days']])
''')
md('''
## RQ1 — Calendar-month seasonality and June 1979–1990

Monthly means are calculated from valid **daily mean temperature** (`TG`, °C). This reveals seasonal patterns but does not by itself establish a warming trend.
''')
code('''
seasonality = weather.groupby('month')['TG'].mean()
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(seasonality.index, seasonality.values, marker='o')
ax.set(title='Heathrow: mean temperature by calendar month (1979–2023)', xlabel='Month', ylabel='Mean temperature (°C)')
ax.set_xticks(range(1, 13)); ax.grid(alpha=0.3)
plt.show()

early_june = (weather.loc[weather['month'].eq(6) & weather['year'].between(1979, 1990)]
              .groupby('year')['TG'].mean())
print('June mean temperature (1979–1990):')
display(early_june.to_frame('June mean (°C)'))
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(early_june.index, early_june.values, marker='o')
ax.set(title='June mean temperature by year, 1979–1990', xlabel='Year', ylabel='Mean temperature (°C)')
ax.grid(alpha=0.3)
plt.show()
''')
md('''
## RQ2 — Decadal temperatures and annual rainfall

The four *complete* decades 1980–1989 through 2010–2019 are compared without overlapping years. The partial 1979 and 2020–2023 periods are not treated as complete decades. Differences are descriptive; for climate-change attribution, station history and statistical modelling would be required.
''')
code('''
decades = decadal_temperature(weather)
display(decades.round(2))
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(decades['decade'], decades['mean_temp_c'], marker='o')
ax.set(title='Mean temperature across complete decades', xlabel='Decade', ylabel='Mean temperature (°C)')
ax.grid(alpha=0.3)
plt.show()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(annual.index, annual['mean_temp'], marker='.', markersize=5)
ax.set(title='Annual mean temperature (quality and coverage checked)', xlabel='Year', ylabel='Mean temperature (°C)')
ax.grid(alpha=0.3)
plt.show()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(annual.index, annual['total_rain'])
ax.set(title='Total annual precipitation (only adequately covered years)', xlabel='Year', ylabel='Precipitation (mm)')
ax.grid(alpha=0.3)
plt.show()
print('Ten wettest observed years with sufficient coverage:')
display(annual['total_rain'].dropna().nlargest(10).to_frame('Rainfall (mm)'))
''')
md('''
## RQ3 — Monthly climatology

Calendar-month averages are derived from **monthly aggregates** (mean temperature and monthly total precipitation), not by summing all years or by accidentally dividing the stored values multiple times.
''')
code('''
climate = monthly.groupby('month').agg(
    mean_temp_c=('temp_mean','mean'),
    mean_max_temp_c=('temp_max_mean','mean'),
    mean_monthly_rain_mm=('rain_total','mean'),
    mean_monthly_sun_hours=('sunshine_total','mean'),
)
display(climate.round(2))
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(climate.index,climate['mean_temp_c'],marker='o',label='Mean daily temp')
ax.plot(climate.index,climate['mean_max_temp_c'],marker='o',label='Mean daily maximum temp')
ax.set(title='Typical monthly temperature at Heathrow', xlabel='Calendar month',ylabel='°C')
ax.set_xticks(range(1,13));ax.legend();ax.grid(alpha=0.3)
plt.show()
fig,ax=plt.subplots(figsize=(9,4))
ax.bar(climate.index,climate['mean_monthly_rain_mm'])
ax.set(title='Average total rainfall by calendar month',xlabel='Calendar month',ylabel='Mean monthly rainfall (mm)')
ax.set_xticks(range(1,13))
plt.show()
''')
md('''
## RQ4 — Was monthly rainfall in 2019–2022 different?

Compare the mean of **monthly rainfall totals** for 1979–2018 with 2019–2022. This is descriptive: four recent years are not enough for claims of statistical significance or attribution. Missing and suspect rainfall readings are excluded from totals.
''')
code('''
rain_period = rainfall_period_comparison(monthly)
display(rain_period.round(2))
x=np.arange(1,13)
fig,ax=plt.subplots(figsize=(10,4))
ax.plot(x,rain_period['baseline_1979_2018_mm'],marker='o',label='1979–2018 baseline')
ax.plot(x,rain_period['recent_2019_2022_mm'],marker='o',label='2019–2022')
ax.set(title='Average monthly precipitation total: two periods',xlabel='Calendar month',ylabel='Monthly rainfall (mm)')
ax.set_xticks(x);ax.legend();ax.grid(alpha=0.3)
plt.show()
''')
md('''
## RQ5 — Rainfall associations, 2019–2022

The next table shows **pairwise Pearson correlations** between same-day rainfall and other observed weather variables. Pairwise missing observations are omitted. These associations do **not** imply causation and are **not used as future-known predictors** in the forecasting section.
''')
code('''
recent = weather.loc[weather['year'].between(2019,2022), ['DATE','RR','TG','HU','CC','PP']].copy()
correlations = recent[['RR','TG','HU','CC','PP']].corr()
display(correlations.round(3))
print('Number of valid rainy-day observations:',recent['RR'].notna().sum())
fig,ax=plt.subplots(figsize=(8,4))
paired=recent[['RR','HU']].dropna()
ax.scatter(paired['HU'],paired['RR'],alpha=.2,s=12)
ax.set(title='Daily rainfall vs relative humidity (2019–2022)',xlabel='Relative humidity (%)',ylabel='Daily rain (mm)')
ax.grid(alpha=0.2)
plt.show()
''')
md('''
## RQ6 — Predict the next day's sunshine and rainfall

**Forecast origin:** for each date *t*, the target is observed sunshine `SS[t]` or rainfall `RR[t]`, while every weather-related predictor uses measurements from **t−1 or earlier**. The day-of-year sine/cosine encodings are known in advance. No feature uses same-day rainfall, sunshine, temperature, humidity, pressure or cloud cover.

**Time partition:** training 1979–2018, validation 2019–2020, **untouched test 2021–2023**. Select the model using **validation MAE only**, then refit the selected learner on train+validation and evaluate on the test period once. The reported comparison also includes simple *yesterday's value* and training-only monthly climatology baselines. Medians for missing input features are learned only from the training partition through scikit-learn pipelines.

**Interpretation:** These are historical one-step-ahead backtests with the true previous day's observations supplied each day—not multi-day forecast simulations. The models are not operational weather forecasts.
''')
code('''
sun_frame = next_day_frame(weather, 'SS')
rain_frame = next_day_frame(weather, 'RR')
for target, frame in [('Sunshine',sun_frame),('Rainfall',rain_frame)]:
    train, validation, test = split_chronologically(frame)
    print(f'{target}: train {len(train):,} through {train.index.max().date()} | '
          f'validation {len(validation):,} | test {len(test):,} through {test.index.max().date()}')
''')
code('''
print('NEXT-DAY SUNSHINE (SS)')
sun_val, sun_test, sun_winner = evaluate_forecast(sun_frame, target='SS')
print('\\n2019–2020 validation methods (sorted by MAE):')
display(sun_val.round(3))
fig,ax=plt.subplots(figsize=(11,4))
recent_test=sun_test.iloc[-180:]
ax.plot(recent_test.index,recent_test['actual'],label='Observed',lw=1.0)
ax.plot(recent_test.index,recent_test['predicted'],label=f'Predicted: {sun_winner}',lw=1.0)
ax.set(title='Next-day sunshine, final 180 test days',xlabel='Date',ylabel='Sunshine duration (hours)')
ax.legend();ax.grid(alpha=0.25)
plt.show()
''')
code('''
print('NEXT-DAY RAINFALL (RR)')
rain_val, rain_test, rain_winner = evaluate_forecast(rain_frame, target='RR')
print('\\n2019–2020 validation methods (sorted by MAE):')
display(rain_val.round(3))
fig,ax=plt.subplots(figsize=(11,4))
recent_test=rain_test.iloc[-180:]
ax.plot(recent_test.index,recent_test['actual'],label='Observed',lw=1.0)
ax.plot(recent_test.index,recent_test['predicted'],label=f'Predicted: {rain_winner}',lw=1.0)
ax.set(title='Next-day rainfall, final 180 test days',xlabel='Date',ylabel='Daily rainfall (mm)')
ax.legend();ax.grid(alpha=0.25)
plt.show()
''')
md('''
## Findings, limitations and next steps

Interpret charts and metrics **after running this notebook with the original dataset**. The provided notebook intentionally contains **no old saved outputs**: those were generated from an earlier notebook with repeatedly scaled values and inconsistent validation and should not be treated as reliable findings.

- **Source scope:** Heathrow station, not all London. Observational data are affected by quality flags, possible station/instrument changes, and sampling decisions.
- **Historical comparisons:** descriptive only. No statistical significance test, uncertainty interval or formal attribution is claimed.
- **Missing values:** quality-flagged observations are excluded from descriptive summaries; only model input features use training-only median imputation.
- **Forecasting:** test metrics depend on the original 1979–2023 dataset and your local environment. Compare to simple baselines, including persistence; a negative R² is possible and should be reported accurately.
- **Future improvements:** stronger rolling-origin validation, uncertainty intervals, station homogeneity checks, additional weather predictors, forecast calibration, and independent external-station tests.

### Reproduce from a terminal

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
jupyter lab DataScienceReport_Fixed.ipynb
```

## References and attribution

- Data mirror and metadata: https://github.com/pc1991/London-Weather-Data-From-1979-To-2023
- Source data project: https://www.ecad.eu/ — European Climate Assessment & Dataset
- ECA&D data policy: https://www.ecad.eu/documents/ECAD_datapolicy.pdf
- Klein Tank, A. M. G. et al. (2002). *Daily dataset of 20th-century surface air temperature and precipitation series for the European Climate Assessment.* International Journal of Climatology 22, 1441–1453.

**Repository status:** Refactored source code and offline synthetic tests are supplied. Full real-data results should be published only after rerunning on the authentic source CSV.
''')

nb=nbf.v4.new_notebook(cells=cells,metadata={
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python'},
})
nbf.validate(nb)
nbf.write(nb,Path('DataScienceReport_Fixed.ipynb'))
print('Notebook created:',len(nb.cells),'cells,',sum(c.cell_type=='code' for c in nb.cells),'code cells')