"""
ARIMAX Forecasting (GDP as exogenous variable)
This script fits ARIMAX models using the same (p,d,q) orders from the
SARIMA baseline, but adds GDP per capita as an external regressor.
The goal is to compare ARIMAX vs SARIMA to quantify GDP's contribution
to predicting CO2 emissions.

Same as SARIMA:
  - Train on data up to 2015
  - Test on 2016–2022
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings, os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

COUNTRIES = ["USA", "China", "India", "Germany", "Brazil", "South Africa"]
TRAIN_END = 2010  # train up through this year
TEST_START = 2011
TEST_END = 2022
DATA_START = 1900  # use 1900+ for all countries
FORECAST_HORIZON = 10  # forecast 10 years beyond 2022

OUTPUT_DIR = "/Users/sihewang/PycharmProjects/ds4002-Project-2-CO2vsGDP-Study/OUTPUT"
co2_raw = pd.read_csv("/Users/sihewang/PycharmProjects/ds4002-Project-2-CO2vsGDP-Study/DATA/co2_pcap_cons.csv", encoding="utf-8-sig")
gdp_raw = pd.read_csv("/Users/sihewang/PycharmProjects/ds4002-Project-2-CO2vsGDP-Study/DATA/gdp_pcap.csv", encoding="utf-8-sig")

SARIMA_ORDERS = {
    "USA":          (0, 1, 4),
    "China":        (2, 2, 1),
    "India":        (1, 2, 1),
    "Germany":      (2, 1, 2),
    "Brazil":       (2, 1, 0),
    "South Africa": (1, 1, 1),
}

# Same as SARIMA
def extract_series(df, country_name, value_name, start, end):
    row = df[df["name"] == country_name]
    year_cols = [str(y) for y in range(start, end + 1)]
    available = [c for c in year_cols if c in df.columns]
    vals = pd.to_numeric(row[available].values.flatten(), errors="coerce")
    idx = pd.date_range(start=str(start), periods=len(available), freq="YS")
    ts = pd.Series(vals, index=idx, name=value_name)
    first_valid = ts.first_valid_index()
    if first_valid is not None:
        ts = ts.loc[first_valid:]
    ts = ts.ffill()
    return ts

# Load SARIMA results
sarima_summary = pd.read_csv(os.path.join(OUTPUT_DIR, "sarima_summary.csv"))
sarima_lookup = sarima_summary.set_index("Country")[["RMSE", "MAE", "AIC"]].to_dict("index")

# ARIMAX Modeling
results = []
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for i, country in enumerate(COUNTRIES):
    print(f"\n-- {country} --")
    co2 = extract_series(co2_raw, country, "co2", DATA_START, TEST_END)
    gdp = extract_series(gdp_raw, country, "gdp", DATA_START, TEST_END)

    # Align indices (in case one starts later)
    common_idx = co2.index.intersection(gdp.index)
    co2 = co2.loc[common_idx]
    gdp = gdp.loc[common_idx]

    # Train/test split
    co2_train = co2[co2.index.year <= TRAIN_END]
    co2_test = co2[co2.index.year >= TEST_START]
    gdp_train = gdp[gdp.index.year <= TRAIN_END]
    gdp_test = gdp[gdp.index.year >= TEST_START]

    order = SARIMA_ORDERS[country]
    print(f"  Order (from SARIMA): ARIMA{order}")
    print(f"  Train: {co2_train.index[0].year}-{co2_train.index[-1].year} ({len(co2_train)} obs)")
    print(f"  Test:  {co2_test.index[0].year}-{co2_test.index[-1].year} ({len(co2_test)} obs)")

    # Standardize GDP to avoid numerical instability (large scale differences)
    gdp_mean = gdp_train.mean()
    gdp_std = gdp_train.std()
    gdp_train_scaled = (gdp_train - gdp_mean) / gdp_std
    gdp_test_scaled = (gdp_test - gdp_mean) / gdp_std  # use train stats

    # Fit ARIMAX (SARIMAX with exog=GDP)
    model = SARIMAX(co2_train, exog=gdp_train_scaled, order=order,
                    enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)

    aic = fit.aic
    # The exog parameter is indexed as 0 (integer) in statsmodels
    gdp_coef = fit.params.iloc[0]  # first param is the exog coefficient
    gdp_pval = fit.pvalues.iloc[0]

    print(f"  AIC: {aic:.2f}")
    sig = "significant" if gdp_pval < 0.05 else "NOT significant"
    print(f"  GDP coefficient: {gdp_coef:.6f} (p={gdp_pval:.4f}, {sig})")

    # Forecast on test period (must provide future GDP values)
    forecast_obj = fit.get_forecast(steps=len(co2_test), exog=gdp_test_scaled.values.reshape(-1, 1))
    forecast_mean = forecast_obj.predicted_mean
    forecast_ci = forecast_obj.conf_int(alpha=0.05)

    forecast_mean.index = co2_test.index
    forecast_ci.index = co2_test.index

    rmse = np.sqrt(mean_squared_error(co2_test, forecast_mean))
    mae = mean_absolute_error(co2_test, forecast_mean)
    print(f"  Test RMSE: {rmse:.4f} tonnes")
    print(f"  Test MAE:  {mae:.4f} tonnes")

    # Compare with SARIMA baseline
    sarima_rmse = sarima_lookup[country]["RMSE"]
    sarima_aic = sarima_lookup[country]["AIC"]
    rmse_change = ((rmse - sarima_rmse) / sarima_rmse) * 100
    aic_change = aic - sarima_aic
    print(f"  vs SARIMA:  RMSE {rmse_change:+.1f}%,  AIC {aic_change:+.2f}")

    results.append({
        "Country": country,
        "order": str(order),
        "ARIMAX_AIC": round(aic, 2),
        "ARIMAX_RMSE": round(rmse, 4),
        "ARIMAX_MAE": round(mae, 4),
        "SARIMA_AIC": round(sarima_aic, 2),
        "SARIMA_RMSE": round(sarima_rmse, 4),
        "AIC_change": round(aic_change, 2),
        "RMSE_change_pct": round(rmse_change, 1),
        "GDP_coef": round(float(gdp_coef), 8),
        "GDP_pval": round(float(gdp_pval), 4),
    })

    # ── PLOT ──
    ax = axes[i]
    ax.plot(co2_train.index, co2_train, color="steelblue", label="Train", linewidth=1)
    ax.plot(co2_test.index, co2_test, color="black", label="Actual (test)", linewidth=2)
    ax.plot(co2_test.index, forecast_mean, color="red", linestyle="--",
            label="ARIMAX forecast", linewidth=2)
    ax.fill_between(co2_test.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1],
                    color="red", alpha=0.15, label="95% CI")
    ax.axvline(pd.Timestamp(f"{TRAIN_END}-01-01"), color="gray", linestyle=":", alpha=0.6)

    rmse_delta = f"{rmse_change:+.1f}%"
    ax.set_title(f"{country}\nARIMAX{order}, RMSE={rmse:.3f} ({rmse_delta} vs SARIMA)", fontsize=11)
    ax.set_ylabel("CO2 per capita (t)")
    ax.legend(fontsize=7, loc="upper left")

fig.suptitle("ARIMAX (with GDP) — CO2 per Capita: Actual vs Forecast", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "arimax_forecasts.png"), dpi=200, bbox_inches="tight")
plt.close()

# ── 4. COMPARISON CHART: SARIMA vs ARIMAX RMSE ───────────────────────
results_df = pd.DataFrame(results)

fig2, ax2 = plt.subplots(figsize=(10, 6))
x = np.arange(len(results_df))
width = 0.35

bars1 = ax2.bar(x - width / 2, results_df["SARIMA_RMSE"], width,
                label="SARIMA (CO2 only)", color="steelblue", alpha=0.8)
bars2 = ax2.bar(x + width / 2, results_df["ARIMAX_RMSE"], width,
                label="ARIMAX (CO2 + GDP)", color="coral", alpha=0.8)

ax2.set_xlabel("Country")
ax2.set_ylabel("Test RMSE (tonnes CO2/capita)")
ax2.set_title("SARIMA vs ARIMAX — Test Set RMSE Comparison")
ax2.set_xticks(x)
ax2.set_xticklabels(results_df["Country"], rotation=15)
ax2.legend()

# Add percentage labels on ARIMAX bars
for j, (bar, pct) in enumerate(zip(bars2, results_df["RMSE_change_pct"])):
    color = "green" if pct < 0 else "red"
    ax2.annotate(f"{pct:+.1f}%", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                 xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9,
                 color=color, fontweight="bold")

fig2.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, "sarima_vs_arimax_comparison.png"), dpi=200, bbox_inches="tight")
plt.close()

# ── 5. SUMMARY TABLE ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("COMPARISON TABLE: SARIMA vs ARIMAX")
print("=" * 70)

display_cols = ["Country", "order", "SARIMA_RMSE", "ARIMAX_RMSE",
                "RMSE_change_pct", "SARIMA_AIC", "ARIMAX_AIC",
                "GDP_coef", "GDP_pval"]
print(results_df[display_cols].to_string(index=False))

results_df.to_csv(os.path.join(OUTPUT_DIR, "arimax_vs_sarima_comparison.csv"), index=False)

print("\n" + "-" * 55)
print("Outputs saved to /mnt/user-data/outputs/:")
print("  - arimax_vs_sarima_comparison.csv")
print("  - arimax_forecasts.png")
print("  - sarima_vs_arimax_comparison.png")
