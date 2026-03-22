"""
This script extends the ARIMAX model by adding binary dummy variables
for major policy interventions, to quantify "step-changes" in emissions
that cannot be explained by GDP alone.

Three models compared per country:
  1. SARIMA       — CO2 history only
  2. ARIMAX       — CO2 + GDP
  3. ARIMAX+Dummy — CO2 + GDP + policy dummy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings, os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
COUNTRIES   = ["USA", "China", "India", "Germany", "Brazil", "South Africa"]
TRAIN_END   = 2010
TEST_START  = 2011
TEST_END    = 2022
DATA_START  = 1900
SARIMA_ORDERS = {
    "USA":          (0, 1, 4),
    "China":        (2, 2, 1),
    "India":        (1, 2, 1),
    "Germany":      (2, 1, 2),
    "Brazil":       (2, 1, 0),
    "South Africa": (1, 1, 1),
}

# Each entry: (year, short label, rationale)
# The dummy = 0 before this year, 1 from this year onward.
POLICY_DUMMIES = {
    "USA": (1990, "Clean Air Act Amendments 1990",
            "Major revision targeting acid rain, urban air pollution, "
            "toxic emissions; led to structural shift in US emissions trajectory"),
    "Germany": (1990, "Reunification + EU climate policy",
                "Fall of Berlin Wall caused East German industrial collapse; "
                "simultaneous adoption of first CO2 reduction targets"),
    "China": (2006, "11th Five-Year Plan",
              "First binding energy intensity targets; marked shift from "
              "unconstrained growth to efficiency-aware industrialization"),
    "India": (2008, "National Action Plan on Climate Change",
              "Eight national missions including solar, energy efficiency; "
              "India's first formal climate policy framework"),
    "Brazil": (2004, "PPCDAm deforestation action plan",
               "Action Plan for Prevention and Control of Deforestation in "
               "the Amazon; land-use change is Brazil's largest CO2 source"),
    "South Africa": (2010, "Integrated Resource Plan 2010",
                     "First comprehensive energy plan committing to renewables "
                     "and diversification away from coal dependence"),
}

OUTPUT_DIR = "/Users/sihewang/PycharmProjects/ds4002-Project-2-CO2vsGDP-Study/OUTPUT"
os.makedirs(OUTPUT_DIR, exist_ok=True)
co2_raw = pd.read_csv("/Users/sihewang/PycharmProjects/ds4002-Project-2-CO2vsGDP-Study/DATA/co2_pcap_cons.csv", encoding="utf-8-sig")
gdp_raw = pd.read_csv("/Users/sihewang/PycharmProjects/ds4002-Project-2-CO2vsGDP-Study/DATA/gdp_pcap.csv", encoding="utf-8-sig")

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

def make_dummy(index, break_year):
    # Create a binary dummy: 0 before break_year, 1 from break_year onward
    return pd.Series((index.year >= break_year).astype(float),
                     index=index, name="policy_dummy")

#Load previous results
sarima_df = pd.read_csv(os.path.join(OUTPUT_DIR, "sarima_summary.csv"))
arimax_df = pd.read_csv(os.path.join(OUTPUT_DIR, "arimax_vs_sarima_comparison.csv"))

sarima_lookup = sarima_df.set_index("Country")["RMSE"].to_dict()
arimax_lookup = arimax_df.set_index("Country")["ARIMAX_RMSE"].to_dict()

# Start modeling
results = []
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for i, country in enumerate(COUNTRIES):
    break_year, policy_label, rationale = POLICY_DUMMIES[country]
    order = SARIMA_ORDERS[country]
    print(f"\n-- {country} --")
    print(f"  Policy: {policy_label} (dummy=1 from {break_year})")
    co2 = extract_series(co2_raw, country, "co2", DATA_START, TEST_END)
    gdp = extract_series(gdp_raw, country, "gdp", DATA_START, TEST_END)
    common_idx = co2.index.intersection(gdp.index)
    co2 = co2.loc[common_idx]
    gdp = gdp.loc[common_idx]

    # Build exog: GDP (standardized) + policy dummy
    gdp_mean = gdp[gdp.index.year <= TRAIN_END].mean()
    gdp_std  = gdp[gdp.index.year <= TRAIN_END].std()
    gdp_scaled = (gdp - gdp_mean) / gdp_std
    dummy = make_dummy(co2.index, break_year)

    exog = pd.DataFrame({"gdp": gdp_scaled, "policy": dummy})

    co2_train = co2[co2.index.year <= TRAIN_END]
    co2_test  = co2[co2.index.year >= TEST_START]
    exog_train = exog[exog.index.year <= TRAIN_END]
    exog_test  = exog[exog.index.year >= TEST_START]

    print(f"  Order: ARIMA{order}")
    print(f"  Train: {co2_train.index[0].year}-{co2_train.index[-1].year} ({len(co2_train)} obs)")
    print(f"  Test:  {co2_test.index[0].year}-{co2_test.index[-1].year} ({len(co2_test)} obs)")

    # Fit
    model = SARIMAX(co2_train, exog=exog_train, order=order, enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    aic = fit.aic
    # Extract coefficients — exog params are the first N params
    gdp_coef = fit.params.iloc[0]
    gdp_pval = fit.pvalues.iloc[0]
    dummy_coef = fit.params.iloc[1]
    dummy_pval = fit.pvalues.iloc[1]

    print(f"  AIC: {aic:.2f}")
    gdp_sig = "sig" if gdp_pval < 0.05 else "NOT sig"
    dummy_sig = "sig" if dummy_pval < 0.05 else "NOT sig"
    print(f"  GDP coef:    {gdp_coef:+.4f} (p={gdp_pval:.4f}, {gdp_sig})")
    print(f"  Dummy coef:  {dummy_coef:+.4f} (p={dummy_pval:.4f}, {dummy_sig})")
    print(f"  Interpretation: policy caused a step-change of {dummy_coef:+.3f} tonnes CO2/capita")

    # Forecast
    forecast_obj = fit.get_forecast(steps=len(co2_test), exog=exog_test.values)
    forecast_mean = forecast_obj.predicted_mean
    forecast_ci   = forecast_obj.conf_int(alpha=0.05)
    forecast_mean.index = co2_test.index
    forecast_ci.index   = co2_test.index

    rmse = np.sqrt(mean_squared_error(co2_test, forecast_mean))
    mae  = mean_absolute_error(co2_test, forecast_mean)

    sarima_rmse = sarima_lookup[country]
    arimax_rmse = arimax_lookup[country]

    print(f"  Test RMSE: {rmse:.4f}")
    print(f"  vs SARIMA:       {((rmse - sarima_rmse)/sarima_rmse)*100:+.1f}%")
    print(f"  vs ARIMAX (GDP): {((rmse - arimax_rmse)/arimax_rmse)*100:+.1f}%")

    results.append({
        "Country": country,
        "order": str(order),
        "policy_event": policy_label,
        "break_year": break_year,
        "SARIMA_RMSE": round(sarima_rmse, 4),
        "ARIMAX_RMSE": round(arimax_rmse, 4),
        "ARIMAX_Dummy_RMSE": round(rmse, 4),
        "ARIMAX_Dummy_AIC": round(aic, 2),
        "GDP_coef": round(float(gdp_coef), 6),
        "GDP_pval": round(float(gdp_pval), 4),
        "Dummy_coef": round(float(dummy_coef), 4),
        "Dummy_pval": round(float(dummy_pval), 4),
        "Dummy_sig": "Yes" if dummy_pval < 0.05 else "No",
    })

    ax = axes[i]
    ax.plot(co2_train.index, co2_train, color="steelblue", label="Train", linewidth=1)
    ax.plot(co2_test.index, co2_test, color="black", label="Actual", linewidth=2)
    ax.plot(co2_test.index, forecast_mean, color="red", linestyle="--",
            label="ARIMAX+Dummy", linewidth=2)
    ax.fill_between(co2_test.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1],
                    color="red", alpha=0.15, label="95% CI")
    ax.axvline(pd.Timestamp(f"{break_year}-01-01"), color="green",
               linestyle="-", alpha=0.7, linewidth=2, label=f"Policy: {break_year}")
    ax.axvline(pd.Timestamp(f"{TRAIN_END}-01-01"), color="gray",
               linestyle=":", alpha=0.5)
    ax.set_title(f"{country}\nRMSE={rmse:.3f} | Dummy={dummy_coef:+.3f}t (p={dummy_pval:.3f})",
                 fontsize=10)
    ax.set_ylabel("CO2/cap (t)")
    ax.legend(fontsize=6.5, loc="upper left")

fig.suptitle("ARIMAX + Policy Dummy — CO2 per Capita Forecast", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "arimax_dummy_forecasts.png"),
            dpi=200, bbox_inches="tight")
plt.close()

# Three-model comparison
results_df = pd.DataFrame(results)
fig2, ax2 = plt.subplots(figsize=(12, 6))
x = np.arange(len(results_df))
w = 0.25

bars1 = ax2.bar(x - w, results_df["SARIMA_RMSE"], w, label="SARIMA", color="steelblue", alpha=0.8)
bars2 = ax2.bar(x, results_df["ARIMAX_RMSE"], w, label="ARIMAX (GDP)", color="coral", alpha=0.8)
bars3 = ax2.bar(x + w, results_df["ARIMAX_Dummy_RMSE"], w, label="ARIMAX+Dummy", color="seagreen", alpha=0.8)

ax2.set_xlabel("Country")
ax2.set_ylabel("Test RMSE (tonnes CO2/capita)")
ax2.set_title("Three-Model Comparison: SARIMA vs ARIMAX vs ARIMAX+Policy Dummy")
ax2.set_xticks(x)
ax2.set_xticklabels(results_df["Country"], rotation=15)
ax2.legend()

fig2.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, "three_model_comparison.png"),
             dpi=200, bbox_inches="tight")
plt.close()

print("\n" + "=" * 70)
print("THREE-MODEL COMPARISON")
print("=" * 70)
display = results_df[["Country", "policy_event", "break_year",
                       "SARIMA_RMSE", "ARIMAX_RMSE", "ARIMAX_Dummy_RMSE",
                       "Dummy_coef", "Dummy_pval", "Dummy_sig"]]
print(display.to_string(index=False))
results_df.to_csv(os.path.join(OUTPUT_DIR, "three_model_comparison.csv"), index=False)
