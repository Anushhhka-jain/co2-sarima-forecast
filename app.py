"""Step 11 (part 2): Streamlit web app for the CO2 SARIMA forecast.

Run locally:  streamlit run app.py
"""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf

from src.ts_utils import (MODEL_INFO_PATH, MODEL_PATH, NOAA_URL, PROJECT_ROOT, RAW_PATH,
                          fit_sarima, load_co2, load_model)

ORDER, SEASONAL_ORDER = (0, 1, 1), (0, 1, 1, 12)
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

st.set_page_config(page_title="CO₂ Forecast · SARIMA", page_icon="🌍", layout="wide")

# Chart colours: same hues in both themes, stepped for the light or dark background
PALETTE = {
    "light": {"history": "#2a78d6", "forecast": "#eb6834", "third": "#1baf7a", "tint": "#86b6ef",
              "neg": "#e34948", "grid": "#e1e0d9", "axis": "#c3c2b7", "muted": "#898781", "band_alpha": 0.15},
    "dark": {"history": "#3987e5", "forecast": "#d95926", "third": "#199e70", "tint": "#1c5cab",
             "neg": "#e66767", "grid": "#2c2c2a", "axis": "#383835", "muted": "#898781", "band_alpha": 0.3},
}
C = PALETTE["dark" if st.context.theme.type == "dark" else "light"]


def rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def style(fig, height=420, y_title="CO₂ (ppm)"):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, traceorder="normal"),
    )
    fig.update_xaxes(showgrid=False, linecolor=C["axis"])
    fig.update_yaxes(gridcolor=C["grid"], zeroline=False, title_text=y_title)
    return fig


# ------------------------------------------------------------------ data & model
@st.cache_resource(show_spinner="Loading the trained model…")
def saved_model():
    if MODEL_PATH.exists():
        return load_model(MODEL_PATH)
    # no saved model yet (e.g. train.py was not run) - fit one from the bundled data
    return fit_sarima(load_co2(RAW_PATH if RAW_PATH.exists() else NOAA_URL), ORDER, SEASONAL_ORDER)


@st.cache_resource(ttl=24 * 3600, show_spinner="Downloading the latest NOAA data and re-fitting the model…")
def live_model():
    return fit_sarima(load_co2(NOAA_URL), ORDER, SEASONAL_ORDER)


@st.cache_data
def model_info():
    return json.loads(MODEL_INFO_PATH.read_text()) if MODEL_INFO_PATH.exists() else None


@st.cache_data
def decompose(y: pd.Series):
    return STL(y, period=12, robust=True).fit()


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Forecast settings")
    horizon = st.slider("Forecast horizon (months)", 1, 120, 36)
    level = st.radio("Prediction interval", ["80%", "95%"], index=1, horizontal=True)
    use_live = st.toggle("Use the latest NOAA data", value=False,
                         help="Downloads the newest monthly values from NOAA and re-fits the model.")

model, source_note = saved_model(), "saved model"
if use_live:
    try:
        model, source_note = live_model(), "re-fitted on the latest NOAA data"
    except Exception as exc:  # network problems should not break the app
        st.sidebar.warning(f"Could not download new data ({exc.__class__.__name__}). Using the saved model.")

y = model.model.data.orig_endog.rename("co2")
last_year = int(y.index.max().year)

with st.sidebar:
    start_year = st.slider("Show history from", int(y.index.min().year), last_year - 1, 2010)
    st.divider()
    st.caption(
        "**Data:** NOAA Global Monitoring Laboratory, monthly mean CO₂ at Mauna Loa Observatory, Hawaii "
        f"([CSV]({NOAA_URL})).\n\n**Model:** SARIMA(0,1,1)(0,1,1)₁₂ — chosen by ACF/PACF analysis and the "
        "lowest BIC of 36 candidates.\n\nApplied Time Series mini project."
    )

alpha = 0.05 if level == "95%" else 0.20
forecast = model.get_forecast(horizon)
mean = forecast.predicted_mean
ci = forecast.conf_int(alpha=alpha)
lower, upper = ci.iloc[:, 0], ci.iloc[:, 1]
long_run = model.get_forecast(240).predicted_mean
crossing = long_run[long_run >= 450].index.min()

# ------------------------------------------------------------------ header
st.title("Atmospheric CO₂ forecast — Mauna Loa")
st.caption(f"SARIMA(0,1,1)(0,1,1)₁₂ · {source_note} · data {y.index.min():%b %Y} – {y.index.max():%b %Y} "
           f"({len(y)} months)")

tab_forecast, tab_eda, tab_data, tab_model = st.tabs(
    ["Forecast", "Exploratory analysis", "Data & seasonality", "Model details"])

# ------------------------------------------------------------------ forecast tab
with tab_forecast:
    yoy = y.iloc[-1] - y.iloc[-13]
    growth = y.iloc[-12:].mean() - y.iloc[-24:-12].mean()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Latest measurement ({y.index[-1]:%b %Y})", f"{y.iloc[-1]:.2f} ppm",
              f"{yoy:+.2f} ppm vs {y.index[-13]:%b %Y}", delta_color="inverse")
    k2.metric(f"Forecast for {mean.index[-1]:%b %Y}", f"{mean.iloc[-1]:.2f} ppm",
              f"{level} range {lower.iloc[-1]:.1f} – {upper.iloc[-1]:.1f}", delta_color="off",
              delta_arrow="off")
    k3.metric("Yearly growth rate (last 12 months)", f"{growth:.2f} ppm/yr",
              help="Average of the last 12 months minus the average of the 12 months before.")
    k4.metric("450 ppm first reached (forecast)",
              f"{crossing:%b %Y}" if pd.notna(crossing) else "after 20 years",
              help="First month the forecast reaches 450 ppm, if the current trend continues.")

    hist = y[y.index.year >= start_year]
    # start the forecast line at the last observation so the two lines connect
    fx = [y.index[-1], *mean.index]
    fy = [y.iloc[-1], *mean.values]
    band = np.column_stack([[y.iloc[-1], *lower.values], [y.iloc[-1], *upper.values]])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist, name="Observed", line=dict(color=C["history"], width=2),
                             hovertemplate="%{y:.2f} ppm<extra>Observed</extra>"))
    fig.add_trace(go.Scatter(x=fx, y=band[:, 1], line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=fx, y=band[:, 0], line=dict(width=0), fill="tonexty",
                             fillcolor=rgba(C["forecast"], C["band_alpha"]), hoverinfo="skip",
                             name=f"{level} prediction interval"))
    fig.add_trace(go.Scatter(x=fx, y=fy, name="Forecast", line=dict(color=C["forecast"], width=2),
                             customdata=band,
                             hovertemplate="%{y:.2f} ppm (range %{customdata[0]:.1f} – %{customdata[1]:.1f})"
                                           "<extra>Forecast</extra>"))
    if band.max() >= 447:
        fig.add_hline(y=450, line=dict(color=C["muted"], width=1),
                      annotation_text="450 ppm", annotation_position="top left")
    st.plotly_chart(style(fig, height=460), width="stretch")

    table = pd.DataFrame({
        "Month": mean.index.strftime("%b %Y"),
        "Forecast (ppm)": mean.values.round(2),
        f"Lower {level} (ppm)": lower.values.round(2),
        f"Upper {level} (ppm)": upper.values.round(2),
    })
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Monthly forecast")
        st.dataframe(table, hide_index=True, height=320, width="stretch", column_config={
            col: st.column_config.NumberColumn(format="%.2f") for col in table.columns[1:]})
        st.download_button("Download forecast (CSV)", table.to_csv(index=False).encode(),
                           file_name=f"co2_forecast_{horizon}_months.csv", mime="text/csv")
    with right:
        st.subheader("Yearly averages")
        combined = pd.concat([y, mean])
        yearly = combined.groupby(combined.index.year).agg(["mean", "size"])
        yearly = yearly[(yearly["size"] == 12) & (yearly.index >= last_year - 1)]
        yearly_table = pd.DataFrame({
            "Year": yearly.index.astype(str),
            "Average CO₂ (ppm)": yearly["mean"].round(2).values,
            "Source": ["observed" if yr < last_year else "includes forecast" for yr in yearly.index],
        })
        if len(yearly_table) > 1:
            st.dataframe(yearly_table, hide_index=True, height=320, width="stretch", column_config={
                "Average CO₂ (ppm)": st.column_config.NumberColumn(format="%.2f")})
        else:
            st.info("Increase the horizon to see complete forecast years.")

# ------------------------------------------------------------------ exploratory analysis tab
def table_height(rows: int) -> int:
    """Pixel height that shows every row of an st.dataframe without scrolling."""
    return (rows + 1) * 35 + 3


def p_text(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


with tab_eda:
    change = y.diff().dropna()  # month-to-month change
    both = [("CO₂ levels", y, "ppm"), ("Month-to-month change", change, "ppm change")]
    dots = dict(color=C["history"], size=5, opacity=0.5)

    # --- line graph
    st.subheader("Line graph")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(f"**Monthly CO₂, {y.index.min():%Y}–{y.index.max():%Y}**")
        full = go.Figure(go.Scatter(x=y.index, y=y, name="CO₂", line=dict(color=C["history"], width=2),
                                    hovertemplate="%{y:.2f} ppm<extra></extra>"))
        st.plotly_chart(style(full, height=340), width="stretch")
    with c2:
        recent = y[y.index > y.index.max() - pd.DateOffset(years=5)]
        st.markdown("**Last 5 years: the yearly up-and-down cycle**")
        zoom = go.Figure(go.Scatter(x=recent.index, y=recent, mode="lines+markers", name="CO₂",
                                    line=dict(color=C["history"], width=2), marker=dict(size=6),
                                    hovertemplate="%{y:.2f} ppm<extra></extra>"))
        st.plotly_chart(style(zoom, height=340), width="stretch")
    st.caption(f"A steady upward trend, from {y.iloc[0]:.0f} ppm in {y.index[0]:%Y} to {y.iloc[-1]:.0f} ppm in "
               f"{y.index[-1]:%Y}, that bends slightly upwards (the rise is speeding up). A regular yearly cycle "
               "sits on top: CO₂ peaks around May and is lowest in September–October. No sudden jumps or outliers.")

    # --- scatter plots
    st.subheader("Scatter plots")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**CO₂ vs time, with fitted trends**")
        t = np.arange(len(y))
        total_ss = np.sum((y - y.mean()) ** 2)
        r2 = {}
        trend = go.Figure(go.Scatter(x=y.index, y=y, mode="markers", name="Monthly value", marker=dots,
                                     hovertemplate="%{x|%b %Y}: %{y:.2f} ppm<extra></extra>"))
        for name, degree, color in [("Linear", 1, C["third"]), ("Quadratic", 2, C["forecast"])]:
            fit = np.polyval(np.polyfit(t, y, degree), t)
            r2[name] = 1 - np.sum((y - fit) ** 2) / total_ss
            trend.add_trace(go.Scatter(x=y.index, y=fit, name=f"{name} trend", line=dict(color=color, width=2),
                                       hovertemplate="%{x|%b %Y}: %{y:.2f} ppm<extra>" + name + " trend</extra>"))
        st.plotly_chart(style(trend, height=360).update_layout(hovermode="closest"), width="stretch")
        st.caption(f"R²: linear {r2['Linear']:.4f}, quadratic {r2['Quadratic']:.4f}. The curve fits better than "
                   "a straight line, so the trend is accelerating.")
    for col, lag in [(c2, 1), (c3, 12)]:
        with col:
            st.markdown(f"**Lag-{lag} plot: y(t) vs y(t−{lag})**")
            pairs = pd.DataFrame({"before": y.shift(lag), "now": y}).dropna()
            earlier = f"{lag} month{'s' if lag > 1 else ''} earlier"
            lag_fig = go.Figure(go.Scatter(
                x=pairs["before"], y=pairs["now"], mode="markers", marker=dots,
                customdata=pairs.index.strftime("%b %Y"),
                hovertemplate="%{customdata}: %{y:.2f} ppm<br>" + earlier + ": %{x:.2f} ppm<extra></extra>"))
            style(lag_fig, height=360, y_title="CO₂ at t (ppm)").update_layout(hovermode="closest")
            lag_fig.update_xaxes(title_text=f"CO₂ at t−{lag} (ppm)")
            st.plotly_chart(lag_fig, width="stretch")
            st.caption(f"Correlation = {y.autocorr(lag):.4f}")
    st.caption("In the lag plots the points lie almost on a straight line: each month is very strongly correlated "
               "with the month before and with the same month last year. Time series models use this memory.")

    # --- histograms
    st.subheader("Histograms")
    for col, (name, s, unit) in zip(st.columns(2), both):
        with col:
            st.markdown(f"**{name}**")
            xs = np.linspace(s.min(), s.max(), 300)
            hist = go.Figure(go.Histogram(x=s, nbinsx=30, histnorm="probability density", name="Histogram",
                                          marker_color=C["tint"],
                                          hovertemplate="%{x} " + unit + "<br>density %{y:.4f}<extra></extra>"))
            for label, curve, color in [("Density (KDE)", stats.gaussian_kde(s)(xs), C["history"]),
                                        ("Normal curve", stats.norm.pdf(xs, s.mean(), s.std()), C["forecast"])]:
                hist.add_trace(go.Scatter(x=xs, y=curve, name=label, line=dict(color=color, width=2),
                                          hovertemplate="%{x:.2f} " + unit + ": %{y:.4f}<extra>" + label + "</extra>"))
            style(hist, height=340, y_title="Density").update_layout(hovermode="closest", bargap=0.05)
            hist.update_xaxes(title_text=unit)
            st.plotly_chart(hist, width="stretch")
    st.caption("The levels are not bell-shaped: they spread over a wide range and lean to the right (more months "
               "at the low values from the early, slower-growing decades). Because of the trend they are not draws "
               "from one distribution, a sign of non-stationarity. The monthly changes have two humps, one for the "
               "months when CO₂ rises and one for when it falls: the seasonal cycle again.")

    # --- QQ plots
    st.subheader("QQ plots")
    for col, (name, s, unit) in zip(st.columns(2), both):
        with col:
            st.markdown(f"**{name}**")
            (theory, ordered), _ = stats.probplot(s, dist="norm")
            ends = theory[[0, -1]]
            qq = go.Figure()
            qq.add_trace(go.Scatter(x=theory, y=ordered, mode="markers", name="Data", marker=dots,
                                    hovertemplate="theoretical %{x:.2f}<br>observed %{y:.2f} " + unit
                                                  + "<extra></extra>"))
            # same standardised reference line as statsmodels qqplot(line="s") in the notebook
            qq.add_trace(go.Scatter(x=ends, y=s.mean() + s.std() * ends, mode="lines",
                                    name="Normal reference line", line=dict(color=C["forecast"], width=2),
                                    hoverinfo="skip"))
            style(qq, height=340, y_title=f"Observed quantiles ({unit})").update_layout(hovermode="closest")
            qq.update_xaxes(title_text="Theoretical quantiles (standard normal)")
            st.plotly_chart(qq, width="stretch")

    rows = []
    for name, s, _ in both:
        shapiro_p, jb_p = stats.shapiro(s).pvalue, stats.jarque_bera(s).pvalue
        rows.append({"Series": name, "Shapiro–Wilk p-value": p_text(shapiro_p), "Jarque–Bera p-value": p_text(jb_p),
                     "Skewness": round(s.skew(), 3), "Excess kurtosis": round(s.kurt(), 3),
                     "Normally distributed?": "Yes" if min(shapiro_p, jb_p) > 0.05 else "No"})
    st.markdown("**Normality tests**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Points bending away from the line at both ends, and p-values below 0.05, mean the data are not "
               "normally distributed. That is expected for a trending, seasonal series: the normality assumption "
               "applies to the model's residuals, not to the raw data.")

    # --- descriptive statistics
    st.subheader("Descriptive statistics")
    counts = y.groupby(y.index.year).size()
    annual = y.groupby(y.index.year).mean()[counts == 12]  # complete calendar years only
    growth = annual.diff()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Monthly series**")
        summary = pd.DataFrame({
            "Statistic": ["Observations", "Period", "Mean", "Median", "Standard deviation", "Minimum",
                          "First quartile (25%)", "Third quartile (75%)", "Maximum", "Range",
                          "Coefficient of variation", "Skewness", "Excess kurtosis",
                          "Average yearly increase, last 10 years"],
            "Value": [f"{len(y)} months", f"{y.index.min():%b %Y} – {y.index.max():%b %Y}",
                      f"{y.mean():.2f} ppm", f"{y.median():.2f} ppm", f"{y.std():.2f} ppm",
                      f"{y.min():.2f} ppm ({y.idxmin():%b %Y})", f"{y.quantile(0.25):.2f} ppm",
                      f"{y.quantile(0.75):.2f} ppm", f"{y.max():.2f} ppm ({y.idxmax():%b %Y})",
                      f"{y.max() - y.min():.2f} ppm", f"{y.std() / y.mean() * 100:.2f} %", f"{y.skew():.3f}",
                      f"{y.kurt():.3f}", f"{growth.iloc[-10:].mean():.2f} ppm/yr"],
        })
        st.dataframe(summary, hide_index=True, width="stretch", height=table_height(len(summary)))
    with c2:
        st.markdown("**By decade** (complete calendar years only)")
        decade = annual.index // 10 * 10
        decades = []
        for start, years in annual.groupby(decade):
            label = f"{start}s"
            if len(years) < 10:  # partial decade: say which years it covers
                first, last = years.index.min(), years.index.max()
                label += f" ({first})" if first == last else f" ({first}–{last})"
            mean_growth = growth.loc[years.index].mean()
            decades.append({"Decade": label, "Mean CO₂ (ppm)": f"{years.mean():.2f}",
                            "Average yearly increase (ppm/yr)": "—" if pd.isna(mean_growth) else f"{mean_growth:.2f}"})
        st.dataframe(pd.DataFrame(decades), hide_index=True, width="stretch", height=table_height(len(decades)))
        early, late = growth.loc[1960:1969].mean(), growth.iloc[-10:].mean()
        st.caption(f"CO₂ grew by {early:.2f} ppm a year in the 1960s and by {late:.2f} ppm a year over the last "
                   f"10 years, about {late / early:.1f} times faster.")

    # --- ACF and PACF
    st.subheader("ACF and PACF")
    st.caption("**ACF:** correlation of the series with its own past values at each lag. **PACF:** the same "
               "correlation after removing the effect of the shorter lags. Bars outside the shaded band "
               "(±1.96/√n, 95% confidence) are statistically significant.")
    nlags = 48
    differenced = y.diff().diff(12).dropna()  # d = 1 and D = 1, as used by the model
    acf_diff = acf(differenced, nlags=nlags)
    row_notes = {
        "Original series": "The ACF starts near 1 and decays very slowly: the signature of a trending, "
                           "non-stationary series, so it has to be differenced first.",
        "After differencing (d = 1, D = 1)": (
            f"The ACF has a negative spike at lag 1 ({acf_diff[1]:.2f}) and a larger one at lag 12 "
            f"({acf_diff[12]:.2f}), then cuts off (the small spike at lag 11 is a side-effect of combining a "
            "lag-1 and a lag-12 term). The PACF tails off at lags 1–3 and at the seasonal lags "
            "12, 24, 36. ACF cutting off while PACF tails off means moving-average terms: MA(1) and seasonal "
            "MA(1), which gives SARIMA(0,1,1)(0,1,1)₁₂."),
    }
    for label, s in [("Original series", y), ("After differencing (d = 1, D = 1)", differenced)]:
        bound = 1.96 / np.sqrt(len(s))
        values = {"ACF": acf_diff if s is differenced else acf(s, nlags=nlags),
                  "PACF": pacf(s, nlags=nlags, method="ywm")}
        for col, (name, corr) in zip(st.columns(2), values.items()):
            with col:
                st.markdown(f"**{name}: {label[0].lower() + label[1:]}**")
                corr_fig = go.Figure(go.Bar(x=np.arange(1, nlags + 1), y=corr[1:], marker_color=C["history"],
                                            width=0.5, hovertemplate="lag %{x}: %{y:.3f}<extra></extra>"))
                corr_fig.add_hrect(y0=-bound, y1=bound, fillcolor=C["muted"], opacity=0.2, line_width=0,
                                   layer="below")
                corr_fig.add_hline(y=0, line=dict(color=C["axis"], width=1))
                style(corr_fig, height=300, y_title=name).update_layout(hovermode="closest")
                corr_fig.update_xaxes(title_text="Lag (months)", dtick=6)
                st.plotly_chart(corr_fig, width="stretch")
        st.caption(row_notes[label])

# ------------------------------------------------------------------ data tab
with tab_data:
    stl = decompose(y)
    st.subheader("STL decomposition: trend + seasonal + residual")
    dec = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Trend", "Seasonal", "Residual"))
    for row, (comp, name) in enumerate([(stl.trend, "Trend"), (stl.seasonal, "Seasonal"),
                                        (stl.resid, "Residual")], start=1):
        dec.add_trace(go.Scatter(x=comp.index, y=comp, name=name, line=dict(color=C["history"], width=1.5),
                                 hovertemplate="%{y:.2f} ppm<extra>" + name + "</extra>"), row=row, col=1)
    dec.update_layout(showlegend=False)
    style(dec, height=560, y_title="ppm")
    dec.update_annotations(x=0, xanchor="left", font_size=13)
    st.plotly_chart(dec, width="stretch")

    st.subheader("Average seasonal effect by month")
    profile = stl.seasonal.groupby(stl.seasonal.index.month).mean()
    bars = go.Figure(go.Bar(
        x=MONTHS, y=profile.values,
        marker_color=[C["history"] if v >= 0 else C["neg"] for v in profile.values],
        hovertemplate="%{x}: %{y:+.2f} ppm<extra></extra>", width=0.55,
    ))
    st.plotly_chart(style(bars, height=340, y_title="ppm above / below trend"), width="stretch")
    st.caption("CO₂ peaks in May and is lowest in September–October: Northern Hemisphere plants absorb "
               "CO₂ while growing in summer and release it when leaves decay. Descriptive statistics are in the "
               "Exploratory analysis tab.")

    with st.expander("Show the monthly data"):
        data_table = y.reset_index()
        data_table.columns = ["Month", "CO₂ (ppm)"]
        data_table["Month"] = data_table["Month"].dt.strftime("%Y-%m")
        st.dataframe(data_table, hide_index=True, height=300, width="stretch")
        st.download_button("Download data (CSV)", data_table.to_csv(index=False).encode(),
                           file_name="co2_monthly.csv", mime="text/csv")

# ------------------------------------------------------------------ model tab
with tab_model:
    info = model_info()
    st.subheader("Model")
    st.latex(r"(1 - B)(1 - B^{12})\,y_t = (1 + \theta_1 B)(1 + \Theta_1 B^{12})\,\varepsilon_t")
    st.caption("d = 1 removes the trend, D = 1 (lag 12) removes the yearly cycle; θ₁ is the non-seasonal and "
               "Θ₁ the seasonal moving-average coefficient. B is the backshift operator (B·yₜ = yₜ₋₁).")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AIC", f"{model.aic:.1f}")
    m2.metric("BIC", f"{model.bic:.1f}")
    if info:
        m3.metric("Hold-out MAPE (36 months)", f"{info['holdout']['MAPE (%)']:.3f} %")
        m4.metric("Hold-out RMSE", f"{info['holdout']['RMSE']:.3f} ppm")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Coefficients")
        names = {"ma.L1": "θ₁ (MA, lag 1)", "ma.S.L12": "Θ₁ (seasonal MA, lag 12)", "sigma2": "σ² (noise variance)"}
        coef = pd.DataFrame({
            "Term": [names.get(n, n) for n in model.params.index],
            "Estimate": model.params.values.round(4),
            "Std error": model.bse.values.round(4),
            "p-value": ["< 0.001" if p < 0.001 else f"{p:.3f}" for p in model.pvalues.values],
        })
        st.dataframe(coef, hide_index=True, width="stretch")

        st.subheader("Ljung–Box test on residuals")
        lb = acorr_ljungbox(model.resid.iloc[13:], lags=[6, 12, 24, 36])
        st.dataframe(pd.DataFrame({
            "Lag": lb.index, "Q statistic": lb["lb_stat"].round(2).values,
            "p-value": lb["lb_pvalue"].round(3).values,
            "White noise?": np.where(lb["lb_pvalue"] > 0.05, "Yes", "No"),
        }), hide_index=True, width="stretch")
        st.caption("p > 0.05 means no autocorrelation is left in the residuals — the model captured the structure.")
    with c2:
        st.subheader("Residual autocorrelation")
        resid = model.resid.iloc[13:]
        r_acf = acf(resid, nlags=36)[1:]
        bound = 1.96 / np.sqrt(len(resid))
        racf = go.Figure(go.Bar(x=np.arange(1, 37), y=r_acf, marker_color=C["history"], width=0.5,
                                hovertemplate="lag %{x}: %{y:.3f}<extra></extra>"))
        for b in (bound, -bound):
            racf.add_hline(y=b, line=dict(color=C["muted"], width=1, dash="dot"))
        style(racf, height=300, y_title="ACF")
        racf.update_layout(hovermode="closest")
        racf.update_xaxes(title_text="Lag (months)", dtick=6)
        st.plotly_chart(racf, width="stretch")

        st.subheader("Residuals over time")
        rfig = go.Figure(go.Scatter(x=resid.index, y=resid, mode="markers",
                                    marker=dict(color=C["history"], size=4),
                                    hovertemplate="%{x|%b %Y}: %{y:+.2f} ppm<extra></extra>"))
        style(rfig, height=260, y_title="ppm")
        rfig.update_layout(hovermode="closest")
        st.plotly_chart(rfig, width="stretch")

    selection_path = PROJECT_ROOT / "reports" / "model_selection.csv"
    if selection_path.exists():
        st.subheader("Model selection: 36 candidates ranked by BIC")
        sel = pd.read_csv(selection_path).sort_values("BIC").head(10)
        st.dataframe(sel.round(2), hide_index=True, width="stretch")
