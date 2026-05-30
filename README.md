# MULTIVARIATE TIME-SERIES BENCHMARKING: STRATEGIC HEDGING FOR COMMODITY MARKETS

## I. EXECUTIVE SUMMARY & STRATEGIC INTENT
This repository deploys a production-grade, multivariate time-series forecasting architecture. The system is engineered to empirically benchmark predictive models on ICE London Robusta Coffee Prices and establish a **variance-based quantitative shield (95% CI)** for financial hedging against exogenous macroeconomic shocks. 

By testing linear, deep learning, and hybrid architectures against market structural breaks, the framework translates raw global indicators (DXY, EFFR, WTI Crude) into actionable corporate pricing thresholds, specifically targeting commodity export risk management.

## II. MATHEMATICAL FOUNDATION & ANTI-LEAKAGE PROTOCOL
The system evaluates and fundamentally rejects the traditional **Zhang (2003) structural decomposition theorem ($Y_t = L_t + N_t$)**, ensuring methodological rigor appropriate for high-volatility commodity markets.

* **Rejection of Linear Constraints:** Under structural breaks, linear base filters (ARIMA) fail, turning residuals ($e_t$) into pure noise. Feeding this noise into deep learning layers causes severe **Error Amplification**.
* **Multivariate Standalone LSTM:** The architecture proves that a Standalone LSTM network, utilizing a 60-day sliding window 3D Tensor ($3622 \times 60 \times 4$), processes macro-financial vectors directly and optimally via gate memory mechanisms, bypassing restrictive linear assumptions.
* **Strict Anti-Leakage Protocol:** To neutralize forward-looking bias (Data Leakage), the `MinMaxScaler` calibration (`fit_transform`) is strictly isolated within the training partition. The validation/testing array is processed exclusively via passive transformation (`transform`).

## III. EMPIRICAL VALIDATION (OUT-OF-SAMPLE)
The framework's benchmarking is mathematically validated over a blind out-of-sample testing horizon (2024-2026). The **Diebold-Mariano (DM) Test** (Statistic: -27.9978, p-value: 0.0000) statistically confirms the failure of hybrid architectures.

| Architecture Evaluation | MAE (USD) | RMSE (USD) | MAPE (%) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| Univariate ARIMA(2,1,2) Baseline | 1460.32 | 1834.38 | 34.86% | -1.4774 |
| Standalone Transformer | 1099.31 | 1446.14 | 25.83% | -0.5397 |
| Hybrid ARIMA-LSTM | 1409.86 | 1767.13 | 33.73% | -1.2991 |
| Hybrid ARIMA-Transformer | 1536.42 | 1927.38 | 36.67% | -1.7350 |
| **Proposed Standalone LSTM** | **452.81** | **608.40** | **10.40%** | **0.7275** |

## IV. CORPORATE HEDGING IMPLICATIONS (THE "SO WHAT?")
The ultimate output is a 7-day recursive forward simulation under static macro conditions (Ceteris Paribus). The system establishes a bearish stabilization trajectory, bounded by a 95% Confidence Interval. 
* **The Quantitative Floor:** The system locks a lower bound risk horizon at **2,775.72 USD/Ton** at T+1.
* **Strategic Execution:** Corporate Risk Managers (CFOs) are advised to utilize this specific mathematical threshold as a hard stop-loss for executing **Put Options** or defining futures margin allocations, effectively insulating export net margins from systemic downside volatility.

## V. MLOPS PIPELINE & WORKSPACE TOPOLOGY
The repository enforces a deterministic execution route, isolating raw data from processed matrices and explicitly quarantining legacy code to ensure end-to-end reproducibility.

```text
Coffee-Hedging-Strategy/
├── data/
│   ├── raw/                  # Asynchronous exogenous market indicators
│   └── processed/            # Cross-market synchronized matrices (ffill imputed)
├── notebooks/
│   ├── 01_data_preprocessing.ipynb    # ETL Pipeline: Volume parsing & Temporal merge
│   └── advanced.ipynb                 # Core execution: Tensor generation & Forecasting
├── reports/                  
│   ├── figures/              
│   │   ├── Fig3_1_AntiLeakage_Protocol.png           # Architectural validation
│   │   └── Fig4_1_Standalone_Loss_Convergence.png    # Algorithmic convergence proof
│   └── Future_7Days.csv      # 7-day quantitative hedge matrix
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