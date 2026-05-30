# MULTIVARIATE TIME-SERIES BENCHMARKING: STRATEGIC HEDGING FOR COMMODITY MARKETS

## I. EXECUTIVE SUMMARY & STRATEGIC INTENT
This repository deploys a production-grade, multivariate time-series forecasting architecture. The system is engineered to empirically benchmark predictive models on ICE London Robusta Coffee Prices and establish a **variance-based quantitative shield (95% CI)** for financial hedging against exogenous macroeconomic shocks.

By testing linear, deep learning, and hybrid architectures against market structural breaks, the framework translates raw global indicators (DXY, EFFR, WTI Crude) into actionable corporate pricing thresholds, specifically targeting commodity export risk management.

---

## II. MATHEMATICAL FOUNDATION & ANTI-LEAKAGE PROTOCOL
The system evaluates and fundamentally rejects the traditional **Zhang (2003) structural decomposition theorem ($Y_t = L_t + N_t$)**, ensuring methodological rigor appropriate for high-volatility commodity markets.

- **Rejection of Linear Constraints:** Under structural breaks, linear base filters (ARIMA) fail, turning residuals ($e_t$) into pure noise. Feeding this noise into deep learning layers causes severe **Error Amplification**, as confirmed by the Diebold-Mariano statistical test.
- **Multivariate Standalone LSTM:** The architecture proves that a Standalone LSTM network, utilizing a 60-day sliding window 3D Tensor ($3622 \times 60 \times 5$), processes macro-financial vectors directly and optimally via gate memory mechanisms, bypassing restrictive linear assumptions.
- **Strict Anti-Leakage Protocol:** To neutralize forward-looking bias (Data Leakage), the `MinMaxScaler` calibration (`fit_transform`) is strictly isolated within the training partition. The validation/testing array is processed exclusively via passive transformation (`transform`).

---

## III. EMPIRICAL VALIDATION (OUT-OF-SAMPLE)
The framework's benchmarking is mathematically validated over a blind out-of-sample testing horizon (2024–2026).

**Diebold-Mariano Pairwise Test Results:**

| Model A | Model B | DM Statistic | p-value | Conclusion |
| :--- | :--- | :---: | :---: | :--- |
| ARIMA | LSTM (Standalone) | +27.5680 | 0.0000 | B outperforms A *** |
| ARIMA | ARIMA-LSTM (Hybrid) | -2.1540 | 0.0312 | A outperforms B ** |
| ARIMA | ARIMA-Transformer (Hybrid) | -3.4210 | 0.0006 | A outperforms B *** |
| LSTM (Standalone) | ARIMA-LSTM (Hybrid) | -28.1250 | 0.0000 | A outperforms B *** |
| LSTM (Standalone) | Transformer (Standalone) | -25.3040 | 0.0000 | A outperforms B *** |

**Out-of-Sample Performance Matrix:**

| Architecture | MAE (USD/Ton) | RMSE (USD/Ton) | MAPE (%) | $R^2$ |
| :--- | :---: | :---: | :---: | :---: |
| Univariate ARIMA(2,1,2) Baseline | 1460.32 | 1834.38 | 34.86 | -1.4774 |
| Standalone Transformer | 1090.39 | 1379.71 | 26.21 | -0.4015 |
| Hybrid ARIMA-LSTM | 1471.24 | 1851.08 | 35.08 | -1.5227 |
| Hybrid ARIMA-Transformer | 1486.20 | 1868.64 | 35.44 | -1.5708 |
| **Proposed: Standalone LSTM** | **406.62** | **536.41** | **9.49** | **0.7882** |

---

## IV. CORPORATE HEDGING IMPLICATIONS (THE "SO WHAT?")
The ultimate output is a 7-day recursive forward simulation under static macro conditions (Ceteris Paribus). The system establishes a **bearish stabilization trajectory**, bounded by a 95% Confidence Interval.

**7-Day Quantitative Hedge Matrix:**

| Horizon | Forecast (USD/Ton) | Risk Floor 95% (USD/Ton) | Risk Ceiling 95% (USD/Ton) |
| :---: | :---: | :---: | :---: |
| T+1 | 3,640.47 | 2,952.17 | 4,328.78 |
| T+2 | 3,014.34 | 2,326.03 | 3,702.65 |
| T+3 | 2,711.11 | 2,022.81 | 3,399.42 |
| T+4 | 2,599.22 | 1,910.91 | 3,287.53 |
| T+5 | 2,542.48 | 1,854.17 | 3,230.79 |
| T+6 | 2,499.47 | 1,811.16 | 3,187.78 |
| T+7 | 2,455.21 | 1,766.91 | 3,143.52 |

**Strategic Execution:**
- **Short Position (Inventory & Export Management):** The Risk Floor at T+1 (**2,952.17 USD/Ton**) serves as the hard stop-loss technical threshold. Corporate Risk Managers (CFOs) are advised to utilize this level for executing **Put Options** or defining futures margin allocations, insulating export net margins from systemic downside volatility.
- **Long Position (Raw Material Procurement):** Defer procurement commitments until price converges toward the technical support zone at T+7 (**2,455.21 USD/Ton**). Counter-trend positions established at T+1 or T+2 carry unnecessary liquidity risk.

---

## V. MLOPS PIPELINE & WORKSPACE TOPOLOGY
The repository enforces a deterministic execution route, isolating raw data from processed matrices and explicitly quarantining legacy code to ensure end-to-end reproducibility.

```text
Coffee-Hedging-Strategy/
├── data/
│   ├── raw/                  # Asynchronous exogenous market indicators
│   └── processed/            # Cross-market synchronized matrices (ffill imputed)
├── notebooks/
│   ├── 01_data_preprocessing.ipynb    # ETL Pipeline: Volume parsing & Temporal merge
│   └── 02_model.ipynb                 # Core execution: Tensor generation & Forecasting
├── reports/
│   ├── figures/
│   │   ├── Fig3_1_AntiLeakage_Protocol.png
│   │   └── Fig4_1_Standalone_Loss_Convergence.png
│   └── Future_7Days.csv      # 7-day quantitative hedge matrix
├── requirements.txt          # Explicit version pinning for environment replication
└── LICENSE                   # MIT License with co-authorship matrix
```

---

## VI. DEPLOYMENT INSTRUCTIONS
To replicate the exact research environment and prevent `ModuleNotFoundError` or structural tensor mismatches (**TensorFlow >= 2.9.1 required**), execute the locked dependency matrix:

```bash
pip install -r requirements.txt
```

Execute the notebooks strictly in their designated numerical order to maintain the integrity of the data pipeline.

---

## VII. INTELLECTUAL PROPERTY & ATTRIBUTION
Distributed under the MIT License.

Copyright (c) 2026 Cao Ha Hai Dang & 2imMinh.

Any academic citation, commercial exploitation, or structural derivation of this pipeline must explicitly reference this repository and preserve the original co-authorship attribution.