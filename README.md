# ds4002-Project-2-CO2vsGDP-Study
For this analysis, we will look at select countries: U.S., Brazil, Germany, India, China, and South Africa.
## Section 1: Software and platform section
For this project, our team members used Apple Mac OS but can be easily reproduced in any operating system due to the cross-platform nature of Python. The notebooks are written in Visual Studio Code and can be ran with the following dependencies:
```
pip install plotly
pip install pandas
pip install scikit-learn
pip install statsmodels
```

## Section 2: A Map of your documentation.
```
.
├── DATA
│   ├── co2_pcap_cons.csv
│   └── gdp_pcap.csv
├── LICENSE
├── OUTPUT
│   ├── arimax_dummy_forecasts.png
│   ├── arimax_forecasts.png
│   ├── arimax_vs_sarima_comparison.csv
│   ├── co2_correlation_heatmap.png
│   ├── correlation_results.csv
│   ├── sarima_forecasts.png
│   ├── sarima_residual_diagnostics.png
│   ├── sarima_summary.csv
│   ├── sarima_vs_arimax_comparison.png
│   ├── scatter_regression.png
│   ├── three_model_comparison.csv
│   ├── three_model_comparison.png
│   └── time_series_overlay.png
├── README.md
└── SCRIPT
    ├── ARIMAX_analysis.py
    ├── ARIMAX_dummy_analysis.py
    ├── EDA.ipynb
    ├── SARIMA_analysis.py
    └── correlation_analysis.py
```
## Section 3: Instructions for reproducing your results. 

For each of the models, perform the following:

1. Open this repo in a Python environment like VSCode, Spyder, or Jupyter notebook.
2. Install the dependencies presented in section 1.
3. Run the script or notebook.