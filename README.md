# Forecasting Atmospheric CO₂ at Mauna Loa with SARIMA

**Applied Time Series (ATS) — course end mini project**

Monthly CO₂ concentration measured at the Mauna Loa Observatory (Hawaii) from March 1958 to
September 2026 (823 months) is analysed with the Box–Jenkins method, modelled with a seasonal ARIMA
model, and deployed as an interactive Streamlit web app.

**Live app:** _add your Streamlit Cloud link here after deploying (see below)_

## Key results

| | |
|---|---|
| Data | NOAA GML monthly mean CO₂, Mauna Loa — 823 months, no missing values in the target |
| Stationarity | Original series non-stationary (ADF p = 1.00); stationary after d = 1 and D = 1 (ADF p < 0.001, KPSS agrees) |
| Seasonality | Strong 12-month cycle (STL seasonal strength 0.98): peak in May, lowest in Sep–Oct |
| Model | **SARIMA(0,1,1)(0,1,1)₁₂** — suggested by ACF/PACF and lowest BIC of 36 candidates |
| Coefficients | θ₁ = −0.38, Θ₁ = −0.86 (both p < 0.001) |
| Residuals | White noise (Ljung–Box p > 0.05 at lags 6–36) |
| Test accuracy (last 36 months) | MAE 0.92 ppm, RMSE 1.01 ppm, **MAPE 0.21 %**, 35 of 36 months inside the 95 % interval |
| Baselines | Seasonal naive MAPE 1.42 %, seasonal naive + drift 0.31 %, Holt–Winters 0.20 % |
| Forecast | Monthly CO₂ first reaches 450 ppm around April 2034 if the current trend continues |

## Project structure

```
ATS/
├── notebooks/CO2_Forecasting.ipynb   # the full analysis, steps 1–10, with explanations
├── src/
│   ├── ts_utils.py                   # data loading, ADF/KPSS reports, model fit/save/load, metrics
│   └── plot_style.py                 # shared chart style for the notebook figures
├── train.py                          # step 11a: fit the final model, check it on a hold-out, save it
├── app.py                            # step 11b: Streamlit web app
├── data/raw/co2_mm_mlo.csv           # NOAA data as downloaded
├── data/processed/co2_monthly.csv    # cleaned series (date, co2)
├── models/sarima_co2.pkl             # saved model (orders + coefficients + data, 18 KB)
├── models/model_info.json            # AIC, BIC, coefficients, hold-out accuracy
├── reports/figures/                  # all 17 notebook figures as PNG (for the report / PPT)
├── reports/model_selection.csv       # AIC/BIC of all 36 candidate models
├── requirements.txt                  # app + training packages
└── requirements-notebook.txt         # adds Jupyter + matplotlib for the notebook
```

## Where each guideline step is

| Step | Notebook section |
|---|---|
| 1. Data collection | Step 1 |
| 2. EDA — line graph, scatter plot, histogram, QQ plot, descriptive statistics | Step 2.1 – 2.5 |
| 3. Missing values | Step 3 (+ 3.1 demo on weekly data with real gaps) |
| 4. Stationarity, ADF test | Step 4 (ADF + KPSS) |
| 5. Transformations | Step 5 (log check, differencing) |
| 6. Seasonality, decomposition | Step 6 (classical + STL) |
| 7. ACF, PACF | Step 7 |
| 8. AR / MA / ARMA / ARIMA / SARIMA | Step 8 |
| 9. Lags | Step 9 |
| 10. AIC and BIC | Step 10 (grid search, diagnostics, test-set evaluation) |
| 11. Deployment | Step 11, `train.py`, `app.py` |

## How to run

```bash
# 1. create and activate a virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate

# 2. install packages
pip install -r requirements-notebook.txt

# 3. run the analysis notebook (or open it in VS Code / Jupyter)
jupyter notebook notebooks/CO2_Forecasting.ipynb

# 4. train and save the final model (add --refresh to download the newest NOAA data first)
python train.py

# 5. start the web app -> opens http://localhost:8501
streamlit run app.py
```

Tested with Python 3.14, pandas 3.1, statsmodels 0.15, streamlit 1.66, plotly 7.1.

## The web app

- **Forecast:** choose a horizon of 1–120 months and an 80 % or 95 % prediction interval. It shows
  the latest measurement, the forecast value, the recent growth rate, and when 450 ppm is reached.
  Includes an interactive chart, a monthly forecast table with CSV download, and yearly averages.
- **Data & seasonality:** STL decomposition, the average seasonal effect by month, descriptive statistics,
  and the raw data.
- **Model details:** model equation, coefficients, AIC/BIC, hold-out accuracy, Ljung–Box test, residual
  plots, and the model-selection table.
- A sidebar switch downloads the newest NOAA data and re-fits the model on the fly.

## Deploying online (free, Streamlit Community Cloud)

1. Create a GitHub repository and push this folder. `.venv` is excluded by `.gitignore`. The data, model
   and reports are small enough to commit.
2. Go to <https://share.streamlit.io>, sign in with GitHub, click **Create app**.
3. Select the repository, branch `main`, and main file `app.py`. In *Advanced settings*, pick Python 3.12
   or newer.
4. Click **Deploy**. Streamlit installs `requirements.txt` and gives you a public link. Paste it at the
   top of this README and into the report.

## Data source

NOAA Global Monitoring Laboratory, *Trends in Atmospheric Carbon Dioxide*, Mauna Loa monthly means,
<https://gml.noaa.gov/ccgg/trends/>. Data before May 1974 are from C. D. Keeling, Scripps Institution of
Oceanography.
