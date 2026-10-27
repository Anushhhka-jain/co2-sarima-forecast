"""Shared helpers for the CO2 time series project.

Used by the notebook, train.py and app.py so the data loading and
tests are written once.
"""
from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX

NOAA_URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "co2_mm_mlo.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "co2_monthly.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "sarima_co2.pkl"
MODEL_INFO_PATH = MODEL_DIR / "model_info.json"
SEASONAL_PERIOD = 12  # monthly data, yearly cycle


def read_noaa_raw(source=RAW_PATH) -> pd.DataFrame:
    """Read the NOAA monthly CSV (local path or URL) into a DataFrame
    indexed by the first day of each month."""
    df = pd.read_csv(source, comment="#")
    df.columns = [c.strip() for c in df.columns]
    df.index = pd.to_datetime(
        pd.DataFrame({"year": df["year"], "month": df["month"], "day": 1})
    )
    df.index.name = "date"
    return df.asfreq("MS")


def load_co2(source=RAW_PATH) -> pd.Series:
    """Return the cleaned monthly CO2 series (ppm) named 'co2'.

    NOAA marks missing values with negative numbers (e.g. -99.99), so any
    non-positive value is treated as missing and filled by time interpolation.
    """
    s = read_noaa_raw(source)["average"].rename("co2")
    s = s.where(s > 0)
    if s.isna().any():
        s = s.interpolate(method="time")
    return s


def adf_report(series: pd.Series, name: str = "") -> pd.Series:
    """Augmented Dickey-Fuller test. H0: the series has a unit root (non-stationary)."""
    stat, pvalue, lags, nobs, crit, _ = adfuller(series.dropna(), autolag="AIC")
    return pd.Series(
        {
            "ADF statistic": round(stat, 4),
            "p-value": round(pvalue, 4),
            "lags used": lags,
            "observations": nobs,
            "critical 1%": round(crit["1%"], 4),
            "critical 5%": round(crit["5%"], 4),
            "verdict": "Stationary" if pvalue < 0.05 else "Non-stationary",
        },
        name=name or series.name,
    )


def kpss_report(series: pd.Series, name: str = "") -> pd.Series:
    """KPSS test (cross-check). H0: the series IS stationary - the opposite of ADF."""
    with warnings.catch_warnings():
        # statsmodels warns when the p-value is outside its lookup table (0.01-0.1)
        warnings.simplefilter("ignore")
        stat, pvalue, lags, crit = kpss(series.dropna(), regression="c", nlags="auto")
    return pd.Series(
        {
            "KPSS statistic": round(stat, 4),
            "p-value": round(pvalue, 4),
            "lags used": lags,
            "critical 5%": crit["5%"],
            "verdict": "Stationary" if pvalue > 0.05 else "Non-stationary",
        },
        name=name or series.name,
    )


def fit_sarima(series: pd.Series, order, seasonal_order):
    """Fit a SARIMA model and return the statsmodels results object."""
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(disp=False)


def save_model(results, path=MODEL_PATH):
    """Save a fitted SARIMA compactly: its orders, fitted coefficients and the data it was fitted on.

    The full statsmodels results object is ~90 MB (it keeps every internal matrix), too big for
    GitHub. These few values are all that is needed to rebuild the exact same model.
    Only plain Python types are stored, so the file loads with any pandas/statsmodels version.
    """
    y = results.model.data.orig_endog
    bundle = {
        "order": tuple(results.model.order),
        "seasonal_order": tuple(results.model.seasonal_order),
        "params": {name: float(v) for name, v in results.params.items()},
        "dates": [d.strftime("%Y-%m-%d") for d in y.index],
        "values": [float(v) for v in y.values],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load_model(path=MODEL_PATH):
    """Rebuild a saved SARIMA. The Kalman filter is re-run with the saved coefficients
    (no re-estimation), so forecasts are identical to the original fitted model."""
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    y = pd.Series(bundle["values"], index=pd.DatetimeIndex(bundle["dates"], freq="MS"), name="co2")
    model = SARIMAX(y, order=bundle["order"], seasonal_order=bundle["seasonal_order"])
    params = np.array([bundle["params"][name] for name in model.param_names])
    return model.filter(params)


def evaluate(actual, predicted) -> dict:
    """MAE, RMSE and MAPE (%) between two aligned series/arrays."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    err = actual - predicted
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE (%)": float(np.mean(np.abs(err / actual)) * 100),
    }
