"""Reproducible, quality-aware Heathrow (London) weather analysis 1979–2023.

ECA&D station observations: data and quality codes documented in README.
These functions distinguish descriptive same-day analysis from next-day forecasting.
"""
from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

DATA_FILENAME = "london_weather_data_1979_to_2023.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/pc1991/"
    "London-Weather-Data-From-1979-To-2023/main/"
    + DATA_FILENAME
)
# Units documented by ECA&D and the public dataset repository.
# QQ is radiation in W/m², HU a percent, CC cloud cover in oktas, SD snow in cm.
TENTHS = ("TX", "TN", "TG", "SS", "RR", "PP")
UNSCALED = ("QQ", "HU", "CC", "SD")
WEATHER_COLUMNS = TENTHS + UNSCALED
RAW_COLUMNS = ("DATE",) + tuple(x for v in WEATHER_COLUMNS for x in (v, "Q_" + v))
MODEL_INPUTS = ("TG", "TX", "TN", "RR", "SS", "PP", "HU", "CC", "QQ")
MODEL_FEATURES = tuple(v + "_lag1" for v in MODEL_INPUTS) + (
    "TG_prior7_mean", "RR_prior7_sum", "SS_prior7_mean", "doy_sin", "doy_cos",
)


def load_raw_weather(path: str | Path | None = None, *, download: bool = True) -> pd.DataFrame:
    """Load the original CSV; cache once locally. Never silently use a made-up dataset."""
    file_path = Path(path) if path is not None else Path("data") / DATA_FILENAME
    if not file_path.is_file():
        if not download:
            raise FileNotFoundError(f"Dataset not found: {file_path}. Download {DATA_URL}")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            request = Request(DATA_URL, headers={"User-Agent": "research-notebook/1.0"})
            with urlopen(request, timeout=35) as response:
                contents = response.read()
            if not contents.startswith(b"DATE,TX,Q_TX") or len(contents) < 100_000:
                raise ValueError("Downloaded file is not the expected London weather CSV")
            file_path.write_bytes(contents)
        except Exception as exc:
            raise RuntimeError(
                "Dataset could not be downloaded. Download the original CSV from\n"
                + DATA_URL + "\nand save it to " + str(file_path)
            ) from exc
    result = pd.read_csv(file_path, low_memory=False, na_values=["-9999", -9999])
    if len(result) < 10:
        raise ValueError("CSV is empty or unexpectedly short")
    return result


def clean_weather_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert units EXACTLY ONCE; Q=0 valid, 1 suspect, 9 missing.

    Does not impute missing data. Raw input is untouched. Quality flags retained
    for a transparent audit. Running this function on cleaned data is rejected.
    """
    if raw.attrs.get("units_converted") is True:
        raise ValueError("Data have already been converted to physical units")
    absent = set(RAW_COLUMNS) - set(raw.columns)
    if absent:
        raise ValueError("Missing expected source columns: " + ", ".join(sorted(absent)))
    out = raw.copy(deep=True)
    out["DATE"] = pd.to_datetime(out["DATE"].astype(str), format="%Y%m%d", errors="coerce")
    if out["DATE"].isna().any():
        raise ValueError("Invalid source DATE value(s), expected YYYYMMDD")
    if out["DATE"].duplicated().any():
        raise ValueError("Duplicate daily DATE values in source file")
    out = out.sort_values("DATE").reset_index(drop=True)
    for col in WEATHER_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace(-9999, np.nan)
        valid = pd.to_numeric(out["Q_" + col], errors="coerce").eq(0)
        out.loc[~valid, col] = np.nan
        if col in TENTHS:
            out[col] = out[col] / 10.0
    # Physically impossible negative precipitation/sunshine is considered missing.
    for col in ("RR", "SS"):
        out.loc[out[col] < 0, col] = np.nan
    out["year"] = out["DATE"].dt.year
    out["month"] = out["DATE"].dt.month
    out["day"] = out["DATE"].dt.day
    out.attrs["units_converted"] = True
    return out


def quality_report(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Summarise quality codes, usable records and source missingness by variable."""
    rows = []
    for col in WEATHER_COLUMNS:
        q = pd.to_numeric(raw["Q_" + col], errors="coerce")
        rows.append({
            "variable": col, "valid_flag": int(q.eq(0).sum()),
            "suspect_flag": int(q.eq(1).sum()),
            "missing_flag": int(q.eq(9).sum()),
            "usable_days": int(cleaned[col].notna().sum()),
            "excluded_or_missing": int(cleaned[col].isna().sum()),
        })
    return pd.DataFrame(rows).set_index("variable")


def monthly_summary(df: pd.DataFrame, minimum_coverage: float = 0.90) -> pd.DataFrame:
    """Monthly aggregates; mask statistics when fewer than 90% days are valid."""
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be between 0 and 1")
    data = df.set_index("DATE").sort_index()
    group = data.resample("MS")
    result = group.agg(temp_mean=("TG", "mean"), temp_max_mean=("TX", "mean"),
                       rain_total=("RR", lambda x: x.sum(min_count=1)),
                       sunshine_total=("SS", lambda x: x.sum(min_count=1)),
                       temp_days=("TG", "count"), rain_days=("RR", "count"),
                       sun_days=("SS", "count"), max_days=("TX", "count"))
    result["days_in_month"] = result.index.days_in_month
    for value, count in (("temp_mean", "temp_days"), ("temp_max_mean", "max_days"),
                         ("rain_total", "rain_days"), ("sunshine_total", "sun_days")):
        result.loc[result[count] / result["days_in_month"] < minimum_coverage, value] = np.nan
    result["year"] = result.index.year
    result["month"] = result.index.month
    return result


def annual_summary(df: pd.DataFrame, minimum_coverage: float = 0.90) -> pd.DataFrame:
    """Annual means/totals with explicit completeness checks."""
    data = df.set_index("DATE").sort_index()
    group = data.resample("YS")
    result = group.agg(mean_temp=("TG", "mean"), total_rain=("RR", lambda x: x.sum(min_count=1)),
                       total_sun=("SS", lambda x: x.sum(min_count=1)),
                       temp_days=("TG", "count"), rain_days=("RR", "count"),
                       sun_days=("SS", "count"), max_days=("TX", "count"))
    result["year"] = result.index.year
    expected = result["year"].apply(lambda y: 366 if pd.Timestamp(f"{y}-12-31").is_leap_year else 365)
    # A truncated year is never silently interpreted as a complete total.
    result["expected_days"] = expected.to_numpy()
    for value, count in (("mean_temp", "temp_days"), ("total_rain", "rain_days"),
                         ("total_sun", "sun_days")):
        result.loc[result[count] / result["expected_days"] < minimum_coverage, value] = np.nan
    return result.set_index("year")


def decadal_temperature(df: pd.DataFrame) -> pd.DataFrame:
    """Hottest and coldest calendar month for four complete decades, 1980–2019."""
    records=[]
    for start in (1980, 1990, 2000, 2010):
        subset=df[df["year"].between(start, start+9)]
        by_month=subset.groupby("month")["TG"].mean()
        if by_month.empty or by_month.isna().all():
            continue
        records.append({"decade": f"{start}–{start+9}",
                        "mean_temp_c": subset["TG"].mean(),
                        "warmest_month": pd.Timestamp(2000,int(by_month.idxmax()),1).strftime("%B"),
                        "coolest_month": pd.Timestamp(2000,int(by_month.idxmin()),1).strftime("%B")})
    result=pd.DataFrame(records)
    if not result.empty:
        result["change_from_previous_c"] = result["mean_temp_c"].diff()
    return result


def rainfall_period_comparison(monthly: pd.DataFrame) -> pd.DataFrame:
    """Compare monthly precipitation totals in 1979–2018 vs 2019–2022 (descriptive)."""
    base = monthly[monthly["year"].between(1979,2018)].groupby("month")["rain_total"].mean()
    recent = monthly[monthly["year"].between(2019,2022)].groupby("month")["rain_total"].mean()
    table = pd.DataFrame({"baseline_1979_2018_mm":base,"recent_2019_2022_mm":recent})
    table["difference_mm"] = table["recent_2019_2022_mm"] - table["baseline_1979_2018_mm"]
    return table


def next_day_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """For date t, features use observations no later than t-1.

    Calendar seasonality for t is known in advance. Missing dates are explicitly
    reindexed so lag1 always means yesterday rather than previous available row.
    """
    if target not in ("RR", "SS"):
        raise ValueError("target must be RR (rain) or SS (sunshine)")
    ordered = df.set_index("DATE").sort_index().asfreq("D")
    features = pd.DataFrame(index=ordered.index)
    for col in MODEL_INPUTS:
        features[col + "_lag1"] = ordered[col].shift(1)
    features["TG_prior7_mean"] = ordered["TG"].shift(1).rolling(7,min_periods=5).mean()
    features["RR_prior7_sum"] = ordered["RR"].shift(1).rolling(7,min_periods=7).sum()
    features["SS_prior7_mean"] = ordered["SS"].shift(1).rolling(7,min_periods=7).mean()
    day = features.index.dayofyear.to_numpy()
    features["doy_sin"] = np.sin(2 * np.pi * day / 365.2425)
    features["doy_cos"] = np.cos(2 * np.pi * day / 365.2425)
    features["target"] = ordered[target]
    features["baseline_yesterday"] = features[target + "_lag1"]
    features["year"] = features.index.year
    features["month"] = features.index.month
    # Known target values are required, but missing FEATURES remain for train-only imputation.
    return features.dropna(subset=["target"]).copy()


def split_chronologically(frame: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Training: ≤2018; validation: 2019–2020; untouched test: 2021–2023."""
    train = frame.loc[frame["year"]<=2018].copy()
    val = frame.loc[frame["year"].between(2019,2020)].copy()
    test = frame.loc[frame["year"].between(2021,2023)].copy()
    if min(len(train),len(val),len(test)) < 10:
        raise ValueError("Need at least 10 valid observations in each time partition")
    if not train.index.max() < val.index.min() < test.index.min():
        raise AssertionError("Time partitions overlap or are unordered")
    return train,val,test


def regression_metrics(y_true, prediction) -> dict[str,float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    yt=np.asarray(y_true,dtype=float)
    yp=np.asarray(prediction,dtype=float)
    if not np.isfinite(yt).all() or not np.isfinite(yp).all() or len(yt)!=len(yp):
        raise ValueError("Non-finite observations or mismatched predictions")
    return {"MAE":float(mean_absolute_error(yt,yp)),
            "RMSE":float(np.sqrt(mean_squared_error(yt,yp))),
            "R2":float(r2_score(yt,yp))}


def evaluate_forecast(frame: pd.DataFrame, *, target: str,
                      n_estimators: int = 80) -> tuple[pd.DataFrame,pd.DataFrame,str]:
    """Choose one method by VALIDATION MAE, refit using train+val, score test once.

    Includes persistence and a training-only seasonal (monthly) mean benchmark.
    These are next-day historical backtests, not operational forecasts.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if target not in ("RR", "SS"):
        raise ValueError("Unsupported target")
    train,val,test = split_chronologically(frame)
    feat=list(MODEL_FEATURES)
    models={
        "Ridge":lambda:Pipeline([("impute",SimpleImputer(strategy="median")),
                                  ("scale",StandardScaler()),("regressor",Ridge(alpha=10))]),
        "Random Forest":lambda:Pipeline([("impute",SimpleImputer(strategy="median")),
                                          ("regressor",RandomForestRegressor(
                                              n_estimators=n_estimators,max_depth=12,
                                              min_samples_leaf=4,random_state=42,n_jobs=2))]),
    }
    def seasonal(train_df, scored_df):
        baseline=train_df.groupby("month")["target"].mean()
        fallback=float(train_df["target"].mean())
        return scored_df["month"].map(baseline).fillna(fallback).to_numpy()

    seasonal_val=seasonal(train,val)
    predictions={
        "Monthly climatology":seasonal_val,
        "Yesterday (persistence)":val["baseline_yesterday"].fillna(pd.Series(seasonal_val,index=val.index)).to_numpy(),
    }
    for name, builder in models.items():
        fitted=builder().fit(train[feat],train["target"])
        predictions[name]=fitted.predict(val[feat])
    rows=[]
    for name, pred in predictions.items():
        rows.append({"method":name,**regression_metrics(val["target"],np.clip(pred,0,None))})
    validation=pd.DataFrame(rows).sort_values(["MAE","method"]).reset_index(drop=True)
    winner=str(validation.iloc[0]["method"])
    combined=pd.concat([train,val]).sort_index()
    seasonal_test=seasonal(combined,test)
    if winner=="Monthly climatology":
        final_pred=seasonal_test
    elif winner=="Yesterday (persistence)":
        final_pred=test["baseline_yesterday"].fillna(pd.Series(seasonal_test,index=test.index)).to_numpy()
    else:
        final_model=models[winner]().fit(combined[feat],combined["target"])
        final_pred=final_model.predict(test[feat])
    final_pred=np.clip(final_pred,0,24 if target=="SS" else None)
    test_baseline=test["baseline_yesterday"].fillna(pd.Series(seasonal_test,index=test.index)).to_numpy()
    comparison=pd.DataFrame({"actual":test["target"].to_numpy(),"predicted":final_pred,
                             "persistence_baseline":np.clip(test_baseline,0,None)},index=test.index)
    comparison.index.name="DATE"
    # Add only final-held-out test results; no selection or retraining based on these values.
    test_metrics=regression_metrics(comparison["actual"],comparison["predicted"])
    baseline_metrics=regression_metrics(comparison["actual"],comparison["persistence_baseline"])
    print(f"{target} next-day forecast | selected by 2019–2020 validation MAE: {winner}")
    print("2021–2023 TEST selected method:", {k:round(v,3) for k,v in test_metrics.items()})
    print("2021–2023 TEST yesterday baseline:", {k:round(v,3) for k,v in baseline_metrics.items()})
    return validation,comparison,winner