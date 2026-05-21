# MULTIVARIATE HYBRID ARIMA-LSTM FRAMEWORK: STRATEGIC HEDGING FOR COMMODITY MARKETS

## I. EXECUTIVE SUMMARY & STRATEGIC INTENT
This repository deploys a production-grade, multivariate hybrid **ARIMA-LSTM** time-series forecasting architecture. The system is engineered not merely to predict ICE London Robusta Coffee Prices, but to establish a **variance-based quantitative shield (95% CI)** for financial hedging against exogenous macroeconomic shocks. 

By systematically decoupling linear market drift from nonlinear volatility, the framework translates raw global indicators (DXY, EFFR, WTI Crude) into actionable corporate pricing thresholds, specifically targeting commodity export risk management.

## II. MATHEMATICAL FOUNDATION & ANTI-LEAKAGE PROTOCOL
The system architecture operates strictly on **Zhang’s (2003) structural decomposition theorem ($Y_t = L_t + N_t$)**, ensuring methodological rigor appropriate for Q3 academic standards.

* **Linear Baseline Filtration (ARIMA):** An ARIMA(0,1,1) model operates as the primary filter, extracting deterministic drift and yielding isolated nonlinear residuals ($e_t = Y_t - \hat{L}_t$).
* **Multivariate Neural Coupling (LSTM):** A 60-day sliding window tensor integrates the ARIMA residuals with macro-financial vectors. The deep LSTM network processes this matrix to capture complex market interactions under *Ceteris Paribus* constraints.
* **Strict Anti-Leakage Protocol:** To neutralize forward-looking bias (Data Leakage), the `MinMaxScaler` calibration (`fit_transform`) is strictly isolated within the training partition. The validation/testing array is processed exclusively via passive transformation (`transform`).

## III. EMPIRICAL VALIDATION (OUT-OF-SAMPLE)
The hybrid framework's superiority is mathematically validated against both linear baselines and standalone deep learning networks over a 300-day out-of-sample testing horizon.

| Architecture Evaluation | MAE (USD) | RMSE (USD) | MAPE (%) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| Univariate ARIMA(0,1,1) Baseline | *Failed* | *Failed* | *Flat Drift* | $< 0.0000$ |
| Standalone Multivariate LSTM | 142.50 | 184.21 | 3.82% | 0.6418 |
| **Proposed Hybrid ARIMA-LSTM** | **78.42** | **105.15** | **2.19%** | **0.8143** |

## IV. CORPORATE HEDGING IMPLICATIONS (THE "SO WHAT?")
The ultimate output is a 30-day recursive forward simulation under static macro conditions. The system establishes a structural mean-reverting stabilization trajectory, bounded by a 95% Confidence Interval. 
* **The Quantitative Floor:** The system locks a lower bound risk horizon at **4,073 USD/Ton**.
* **Strategic Execution:** Corporate Risk Managers (CFOs) are advised to utilize this specific mathematical threshold as the baseline for executing **Put Options** or defining futures margin allocations, effectively insulating export net margins from systemic downside volatility.

## V. MLOPS PIPELINE & WORKSPACE TOPOLOGY
The repository enforces a deterministic execution route, isolating raw data from processed matrices and explicitly quarantining legacy code to ensure end-to-end reproducibility.

```text
ARIMA-LSTM-Coffee-Forecasting/
├── data/
│   ├── raw/                  # Asynchronous exogenous market indicators
│   └── processed/            # Cross-market synchronized matrices (ffill imputed)
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   └── 02_hybrid_arima_lstm.ipynb   # Core execution: Calibration & Validation
├── reports/                  
│   ├── figures/              # High-resolution (300 DPI) academic validations
│   └── Future_30Days.csv     # 30-day quantitative hedge matrix
├── legacy_v1/                # Quarantined deprecated codebase
├── requirements.txt          # Explicit version pinning for environment replication
└── LICENSE                   # MIT License with Co-authorship matrix
## VI. DEPLOYMENT INSTRUCTIONS
To replicate the exact research environment and prevent ModuleNotFoundError or structural tensor mismatches (TensorFlow >= 2.9.1 required), execute the locked dependency matrix:

Bash
pip install -r requirements.txt
Execute the notebooks strictly in their designated numerical order to maintain the integrity of the data pipeline.

## VII. INTELLECTUAL PROPERTY & ATTRIBUTION
Distributed under the MIT License.

Copyright (c) 2026 Cao Ha Hai Dang & 2imMinh.

Any academic citation, commercial exploitation, or structural derivation of this pipeline must explicitly reference this repository and preserve the original co-authorship attribution.