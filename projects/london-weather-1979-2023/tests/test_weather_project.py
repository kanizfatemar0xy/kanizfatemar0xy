"""Offline tests: synthetic and small hand-checked source-format weather rows."""
import unittest
import numpy as np
import pandas as pd

from weather_project import (
    TENTHS, WEATHER_COLUMNS, clean_weather_data, quality_report,
    monthly_summary, annual_summary, rainfall_period_comparison,
    decadal_temperature, next_day_frame, split_chronologically,
    regression_metrics, evaluate_forecast,
)


def raw_fixture(start="2018-01-01", periods=7, *, seed=2026):
    rng=np.random.default_rng(seed)
    days=pd.date_range(start,periods=periods,freq="D")
    t=np.arange(periods)
    # Values follow the upstream raw unit conventions (integers in tenths).
    values={"TX":120+5*np.sin(t/365*2*np.pi)*10, "TN":50+3*np.cos(t/365*2*np.pi)*10,
            "TG":80+10*np.sin(t/365*2*np.pi)*10, "SS":45+20*np.cos(t/365*2*np.pi),
            "RR":rng.integers(0,80,size=periods), "PP":10130+90*np.sin(t/17),
            "QQ":80+80*np.maximum(0,np.cos(t/30)),"HU":70+5*np.cos(t/17),
            "CC":4+3*np.sin(t/10),"SD":np.zeros(periods)}
    data={"DATE":days.strftime("%Y%m%d").astype(int)}
    for name in WEATHER_COLUMNS:
        data[name]=np.rint(values[name]).astype(float)
        data["Q_"+name]=np.zeros(periods,dtype=int)
    return pd.DataFrame(data)


class TestWeatherCleaning(unittest.TestCase):
    def test_unit_conversion_and_date(self):
        src=raw_fixture(periods=4)
        src.loc[0,"TG"]=123; src.loc[0,"RR"]=18; src.loc[0,"SS"]=70
        src.loc[0,"PP"]=10190; src.loc[0,"QQ"]=120; src.loc[0,"HU"]=83
        out=clean_weather_data(src)
        self.assertEqual(str(out.loc[0,"DATE"].date()),"2018-01-01")
        for key,expected in {"TG":12.3,"RR":1.8,"SS":7.,"PP":1019.,"QQ":120,"HU":83}.items():
            self.assertAlmostEqual(out.loc[0,key],expected,msg=key)
        self.assertEqual(src.loc[0,"TG"],123)

    def test_quality_flags_exclude_suspect_and_missing(self):
        src=raw_fixture(periods=5)
        src.loc[1,"Q_TG"]=1
        src.loc[2,"Q_TG"]=9
        src.loc[3,"RR"]=-9999
        src.loc[4,"Q_PP"]=1
        out=clean_weather_data(src)
        self.assertTrue(pd.isna(out.loc[1,"TG"]))
        self.assertTrue(pd.isna(out.loc[2,"TG"]))
        self.assertTrue(pd.isna(out.loc[3,"RR"]))
        self.assertTrue(pd.isna(out.loc[4,"PP"]))
        self.assertEqual(quality_report(src,out).loc["TG","suspect_flag"],1)
        self.assertEqual(quality_report(src,out).loc["TG","missing_flag"],1)

    def test_prevent_double_scaling(self):
        out=clean_weather_data(raw_fixture())
        with self.assertRaisesRegex(ValueError,"already"):
            clean_weather_data(out)

    def test_date_duplicate_rejected(self):
        src=raw_fixture()
        src.loc[1,"DATE"]=src.loc[0,"DATE"]
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            clean_weather_data(src)

    def test_missing_column_rejected(self):
        with self.assertRaisesRegex(ValueError,"Missing"):
            clean_weather_data(raw_fixture().drop(columns=["Q_TG"]))

    def test_invalid_date_rejected(self):
        src=raw_fixture();src.loc[0,"DATE"]=20181344
        with self.assertRaisesRegex(ValueError,"Invalid"):
            clean_weather_data(src)

    def test_monthly_completeness_mask(self):
        src=raw_fixture("2020-01-01",31)
        src.loc[:7,"Q_RR"]=9
        clean=clean_weather_data(src)
        m=monthly_summary(clean,0.9)
        self.assertTrue(pd.isna(m.loc["2020-01-01","rain_total"]))
        self.assertEqual(int(m.loc["2020-01-01","rain_days"]),23)

    def test_year_completeness_mask(self):
        src=raw_fixture("2021-01-01",365)
        src.loc[:100,"Q_RR"]=9
        annual=annual_summary(clean_weather_data(src))
        self.assertTrue(pd.isna(annual.loc[2021,"total_rain"]))
        self.assertEqual(int(annual.loc[2021,"expected_days"]),365)

    def test_annual_coverage_leap_year(self):
        src=raw_fixture("2020-01-01",366)
        annual=annual_summary(clean_weather_data(src))
        self.assertEqual(int(annual.loc[2020,"expected_days"]),366)

    def test_decades_and_periods(self):
        src=raw_fixture("1979-01-01",45*365+11)
        df=clean_weather_data(src)
        dec=decadal_temperature(df)
        self.assertEqual(len(dec),4)
        rain=rainfall_period_comparison(monthly_summary(df))
        self.assertEqual(len(rain),12)

    def test_forecasting_no_current_day_feature_leakage(self):
        src=raw_fixture("2018-01-01",11)
        cleaned=clean_weather_data(src)
        f=next_day_frame(cleaned,"RR")
        date=pd.Timestamp("2018-01-04")
        self.assertAlmostEqual(f.loc[date,"TG_lag1"],cleaned.loc[2,"TG"])
        self.assertAlmostEqual(f.loc[date,"RR_lag1"],cleaned.loc[2,"RR"])
        self.assertAlmostEqual(f.loc[date,"target"],cleaned.loc[3,"RR"])
        # Change same-day raw value; prediction features as-of yesterday must not change.
        src.loc[3,"RR"]=5000
        changed=next_day_frame(clean_weather_data(src),"RR")
        self.assertTrue(np.allclose(f.loc[date,list(next_day_frame(cleaned,"RR").columns[:14])].to_numpy(dtype=float),
                                    changed.loc[date,list(next_day_frame(cleaned,"RR").columns[:14])].to_numpy(dtype=float),equal_nan=True))
        self.assertNotEqual(f.loc[date,"target"],changed.loc[date,"target"])

    def test_missing_days_do_not_create_false_yesterday(self):
        df=clean_weather_data(raw_fixture("2018-01-01",8))
        df=df[df["DATE"]!=pd.Timestamp("2018-01-03")]
        f=next_day_frame(df,"SS")
        self.assertTrue(pd.isna(f.loc["2018-01-04","SS_lag1"]))

    def test_time_split_nonoverlap(self):
        df=clean_weather_data(raw_fixture("2017-01-01",7*365))
        f=next_day_frame(df,"SS")
        train,val,test=split_chronologically(f)
        self.assertLess(train.index.max(),val.index.min())
        self.assertLess(val.index.max(),test.index.min())
        self.assertEqual(int(test["year"].min()),2021)

    def test_metrics_no_nan(self):
        s=regression_metrics([0,1,2],[0,1,2])
        self.assertEqual(s["MAE"],0)
        with self.assertRaises(ValueError):
            regression_metrics([1,2],[1,np.nan])

    def test_pipeline_smoke_both_targets(self):
        df=clean_weather_data(raw_fixture("2016-01-01",365*8))
        for target in ("SS","RR"):
            frame=next_day_frame(df,target)
            table,preds,chosen=evaluate_forecast(frame,target=target,n_estimators=6)
            self.assertEqual(len(table),4)
            self.assertEqual(preds.index.min().year,2021)
            self.assertTrue((preds["predicted"]>=0).all())
            self.assertIn(chosen,set(table["method"]))


if __name__=="__main__":
    unittest.main()