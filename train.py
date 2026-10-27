"""Step 11 (part 1): train the final SARIMA model and save it for the Streamlit app.

Usage:
    python train.py            # use the saved data in data/raw/
    python train.py --refresh  # download the latest NOAA data first

The model order was chosen in notebooks/CO2_Forecasting.ipynb (Step 10):
lowest BIC of 36 candidates, white-noise residuals and the best test accuracy.
"""
import argparse
import json
import urllib.request
from datetime import datetime, timezone

from statsmodels.stats.diagnostic import acorr_ljungbox

from src.ts_utils import (MODEL_INFO_PATH, MODEL_PATH, NOAA_URL, PROCESSED_PATH, RAW_PATH,
                          evaluate, fit_sarima, load_co2, load_model, save_model)

ORDER = (0, 1, 1)
SEASONAL_ORDER = (0, 1, 1, 12)
TEST_MONTHS = 36


def download_data():
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(NOAA_URL, RAW_PATH)
    print(f"Downloaded latest data to {RAW_PATH}")


def main(refresh: bool = False):
    if refresh or not RAW_PATH.exists():
        download_data()

    y = load_co2(RAW_PATH)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    y.to_csv(PROCESSED_PATH)
    print(f"Data: {len(y)} months, {y.index.min():%b %Y} – {y.index.max():%b %Y}")

    # 1) Hold-out check: train on all but the last 36 months, forecast them
    train, test = y.iloc[:-TEST_MONTHS], y.iloc[-TEST_MONTHS:]
    holdout = fit_sarima(train, ORDER, SEASONAL_ORDER)
    forecast = holdout.get_forecast(TEST_MONTHS)
    ci = forecast.conf_int(alpha=0.05)
    metrics = evaluate(test, forecast.predicted_mean)
    metrics["95% interval coverage (%)"] = float(
        ((test >= ci.iloc[:, 0]) & (test <= ci.iloc[:, 1])).mean() * 100
    )
    print("Hold-out accuracy:", {k: round(v, 3) for k, v in metrics.items()})

    # 2) Final model on the full series
    model = fit_sarima(y, ORDER, SEASONAL_ORDER)
    lb = acorr_ljungbox(model.resid.iloc[13:], lags=[12, 24])  # skip start-up residuals

    save_model(model, MODEL_PATH)
    reloaded = load_model(MODEL_PATH)
    gap = (reloaded.forecast(24) - model.forecast(24)).abs().max()
    assert gap < 1e-8, f"Reloaded model gives different forecasts (max diff {gap})"

    info = {
        "model": "SARIMA(0,1,1)(0,1,1)12",
        "order": list(ORDER),
        "seasonal_order": list(SEASONAL_ORDER),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "log_likelihood": float(model.llf),
        "params": {k: float(v) for k, v in model.params.items()},
        "pvalues": {k: float(v) for k, v in model.pvalues.items()},
        "ljung_box_pvalues": {f"lag {k}": float(v) for k, v in lb["lb_pvalue"].items()},
        "n_obs": int(len(y)),
        "data_start": f"{y.index.min():%Y-%m}",
        "data_end": f"{y.index.max():%Y-%m}",
        "holdout": {
            "months": TEST_MONTHS,
            "train_end": f"{train.index.max():%Y-%m}",
            **{k: float(v) for k, v in metrics.items()},
        },
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    MODEL_INFO_PATH.write_text(json.dumps(info, indent=2))
    print(f"Saved model to {MODEL_PATH} and summary to {MODEL_INFO_PATH}")
    print(f"AIC = {model.aic:.2f}, BIC = {model.bic:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true", help="download the latest NOAA data first")
    main(parser.parse_args().refresh)
