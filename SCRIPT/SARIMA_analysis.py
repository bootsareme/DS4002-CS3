"""
This script fits SARIMA models to historical CO2 per capita data for each
case-study country. Per the analysis plan:
  - Train on data up to 2015
  - Test on 2016–2022
  - Evaluate with RMSE and AIC
  - Run residual diagnostics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings, os, itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys

warnings.filterwarnings("ignore")

COUNTRIES = ["USA", "China", "India", "Germany", "Brazil", "South Africa"]
TRAIN_END = 2010  # train up through this year
TEST_START = 2011
TEST_END = 2022
DATA_START = 1900  # use 1900+ for all countries
FORECAST_HORIZON = 10  # forecast 10 years beyond 2022

OUTPUT_DIR = "/Users/sihewang/PycharmProjects/ds4002-Project-2-CO2vsGDP-Study/OUTPUT"
co2_raw = pd.read_csv("/Users/sihewang/PycharmProjects/ds4002-Project-2-CO2vsGDP-Study/DATA/co2_pcap_cons.csv", encoding="utf-8-sig")

def extract_series(df, country_name, start, end):
    # There are some leading NaNs for China's data so we're dropping them
    row = df[df["name"] == country_name]
    year_cols = [str(y) for y in range(start, end + 1)]
    available = [c for c in year_cols if c in df.columns]
    vals = pd.to_numeric(row[available].values.flatten(), errors="coerce")
    idx = pd.date_range(start=str(start), periods=len(available), freq="YS")
    ts = pd.Series(vals, index=idx, name="co2_per_capita")
    first_valid = ts.first_valid_index()
    if first_valid is not None:
        ts = ts.loc[first_valid:]
    ts = ts.ffill()
    return ts

def find_best_arima_order(train_data, max_p=4, max_d=2, max_q=4):
    """
    Alternate method to replace pmdarima because it doesn't work for me.
    Grid search over ARIMA(p,d,q) orders and return the one with lowest AIC.
    This replaces auto_arima from pmdarima so no extra dependency is needed.
    """
    # Determine d using ADF test
    best_d = 0
    temp = train_data.copy()
    for d in range(max_d + 1):
        adf_p = adfuller(temp.dropna(), autolag="AIC")[1]
        if adf_p < 0.05:
            best_d = d
            break
        temp = temp.diff().dropna()
        best_d = d + 1
    best_d = min(best_d, max_d)

    best_aic = np.inf
    best_order = (0, best_d, 0)

    # Grid search over p and q with fixed d
    for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
        if p == 0 and q == 0:
            # Still try this (random walk if d>0)
            pass
        try:
            model = SARIMAX(train_data, order=(p, best_d, q),
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            fit = model.fit(disp=False, maxiter=200)
            if fit.aic < best_aic:
                best_aic = fit.aic
                best_order = (p, best_d, q)
        except Exception:
            continue

    return best_order


# Stationary testing
for country in COUNTRIES:
    ts = extract_series(co2_raw, country, DATA_START, TEST_END)
    adf_stat, p_val, _, _, crit, _ = adfuller(ts.dropna(), autolag="AIC")
    status = "Stationary" if p_val < 0.05 else "Non-stationary -> differencing needed"
    print(f"  {country:15s}  ADF={adf_stat:+.3f}  p={p_val:.4f}  [{status}]")

    ts_diff = ts.diff().dropna()
    adf2, p2, *_ = adfuller(ts_diff, autolag="AIC")
    d1_status = "Stationary after d=1" if p2 < 0.05 else "Still non-stationary"
    print(f"  {'':15s}  After d=1: ADF={adf2:+.3f}  p={p2:.4f}  [{d1_status}]")
print()

# SARIMA fitting
results = {}
all_forecasts = {}

fig_scatter, axes_sc = plt.subplots(2, 3, figsize=(18, 11))
axes_sc = axes_sc.flatten()

fig_diag_all, axes_diag = plt.subplots(6, 3, figsize=(20, 28))

for i, country in enumerate(COUNTRIES):
    print(f"\n-- {country} --")

    ts_full = extract_series(co2_raw, country, DATA_START, TEST_END)
    train = ts_full[ts_full.index.year <= TRAIN_END]
    test = ts_full[ts_full.index.year >= TEST_START]

    print(f"  Train: {train.index[0].year}-{train.index[-1].year} ({len(train)} obs)")
    print(f"  Test:  {test.index[0].year}-{test.index[-1].year} ({len(test)} obs)")
    order = find_best_arima_order(train)
    print(f"  Best order: ARIMA{order}")
#________________Fit on training data______________
    model = SARIMAX(train, order=order, enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)

    aic = fit.aic
    print(f"  AIC: {aic:.2f}")

    # Forecast on test period
    forecast_obj = fit.get_forecast(steps=len(test))
    forecast_mean = forecast_obj.predicted_mean
    forecast_ci = forecast_obj.conf_int(alpha=0.05)

    forecast_mean.index = test.index
    forecast_ci.index = test.index

    rmse = np.sqrt(mean_squared_error(test, forecast_mean))
    mae = mean_absolute_error(test, forecast_mean)
    print(f"  Test RMSE: {rmse:.4f} tonnes")
    print(f"  Test MAE:  {mae:.4f} tonnes")

    # Extended forecast beyond 2022
    extended_model = SARIMAX(ts_full, order=order,
                             enforce_stationarity=False, enforce_invertibility=False)
    extended_fit = extended_model.fit(disp=False)
    future_fc = extended_fit.get_forecast(steps=FORECAST_HORIZON)
    future_mean = future_fc.predicted_mean
    future_ci = future_fc.conf_int(alpha=0.05)

    # Store results
    results[country] = {
        "order": order,
        "AIC": round(aic, 2), "RMSE": round(rmse, 4), "MAE": round(mae, 4),
        "train_size": len(train), "test_size": len(test),
    }
    all_forecasts[country] = {
        "train": train, "test": test,
        "test_forecast": forecast_mean, "test_ci": forecast_ci,
        "future_mean": future_mean, "future_ci": future_ci,
        "fit": fit,
    }

    # Plot
    ax = axes_sc[i]
    ax.plot(train.index, train, color="steelblue", label="Train", linewidth=1)
    ax.plot(test.index, test, color="black", label="Actual (test)", linewidth=2)
    ax.plot(test.index, forecast_mean, color="red", linestyle="--",
            label="Forecast", linewidth=2)
    ax.fill_between(test.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1],
                    color="red", alpha=0.15, label="95% CI")
    ax.plot(future_mean.index, future_mean, color="darkorange", linestyle=":",
            linewidth=2, label="Future forecast")
    ax.fill_between(future_ci.index, future_ci.iloc[:, 0], future_ci.iloc[:, 1],
                    color="orange", alpha=0.1)
    ax.set_title(f"{country}\nARIMA{order}, RMSE={rmse:.3f}", fontsize=11)
    ax.set_ylabel("CO2 per capita (t)")
    ax.legend(fontsize=7, loc="upper left")
    ax.axvline(pd.Timestamp(f"{TRAIN_END}-01-01"), color="gray",
               linestyle=":", alpha=0.6)

    # Residual Diagnostic
    residuals = fit.resid

    axes_diag[i, 0].plot(residuals, color="steelblue", linewidth=0.8)
    axes_diag[i, 0].axhline(0, color="red", linestyle="--", linewidth=0.8)
    axes_diag[i, 0].set_title(f"{country} - Residuals", fontsize=10)
    axes_diag[i, 0].set_ylabel("Residual")

    axes_diag[i, 1].hist(residuals, bins=30, density=True, alpha=0.6,
                         color="steelblue", edgecolor="white")
    axes_diag[i, 1].set_title(f"{country} - Residual Distribution", fontsize=10)

    plot_acf(residuals, ax=axes_diag[i, 2], lags=30, alpha=0.05)
    axes_diag[i, 2].set_title(f"{country} - Residual ACF", fontsize=10)

fig_scatter.suptitle("SARIMA Baseline - CO2 per Capita: Actual vs Forecast",
                     fontsize=14, y=1.01)
fig_scatter.tight_layout()
fig_scatter.savefig(os.path.join(OUTPUT_DIR, "sarima_forecasts.png"),
                    dpi=200, bbox_inches="tight")
plt.close(fig_scatter)

fig_diag_all.suptitle("SARIMA Residual Diagnostics", fontsize=14, y=1.005)
fig_diag_all.tight_layout()
fig_diag_all.savefig(os.path.join(OUTPUT_DIR, "sarima_residual_diagnostics.png"),
                     dpi=200, bbox_inches="tight")
plt.close(fig_diag_all)

# Print summary table
summary_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Country"})
summary_df["order"] = summary_df["order"].astype(str)
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(OUTPUT_DIR, "sarima_summary.csv"), index=False)

print("Outputs saved:")
print("  - sarima_summary.csv")
print("  - sarima_forecasts.png")
print("  - sarima_residual_diagnostics.png")