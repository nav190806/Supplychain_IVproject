# Interpretable Multi-Tier Supply Chain Decision Intelligence

This project implements the architecture described in the supplied paper:

- Lead-Time Variance (LT variance)
- Supplier Criticality Index (SCI)
- Geographic Stress Index (GSI)
- LightGBM + XGBoost disruption prediction
- TreeSHAP explanations
- MILP recovery optimization
- Baseline comparison
- Evaluation metrics and visualizations

Paper basis: *An Interpretable Multi-Tier Decision Intelligence Framework for Dynamic Supply Chain Risk Mitigation and Recovery Optimization*.

## Important dataset note

The paper states that evaluation used the Kaggle **Supply Chain Disruption and Recovery Dataset**, with 124,500 observations and 38 engineered/raw dimensions. The actual dataset file was not supplied with the paper.

Therefore this implementation has two modes:

1. **Real dataset mode**: place the Kaggle CSV in `data/supply_chain.csv` and adapt the column mapping in `src/config.py` if your column names differ.
2. **Demo mode**: the pipeline automatically generates a reproducible synthetic dataset with the same conceptual variables. Demo results must NOT be presented as the paper's reported empirical results.

## Project structure

```text
supply_chain_decision_intelligence/
├── data/
│   └── supply_chain.csv              # optional real dataset
├── outputs/                           # generated plots, models and reports
├── src/
│   ├── config.py
│   ├── data_pipeline.py
│   ├── models.py
│   ├── explainability.py
│   ├── optimizer.py
│   ├── baselines.py
│   ├── evaluation.py
│   └── main.py
├── requirements.txt
├── README.md
└── run_project.py
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
python run_project.py
```

The script will:
1. Load `data/supply_chain.csv` if present.
2. Otherwise generate demo data.
3. Engineer the paper's domain features.
4. Train LightGBM and XGBoost.
5. Build a probability ensemble.
6. Calculate classification and delay-regression metrics.
7. Generate SHAP explanations.
8. Solve a MILP recovery problem.
9. Compare reactive, fixed-stock, greedy and MILP+XAI recovery strategies.
10. Save outputs to `outputs/`.

## Mapping a real dataset

The code expects these conceptual fields:

- supplier_id
- region
- transit_mode
- lead_time_days
- historical_delay_days
- order_volume
- component_exclusivity
- financial_stability
- geopolitical_stress
- climate_stress
- port_congestion
- disruption_flag

If your dataset uses different names, edit `COLUMN_MAP` in `src/config.py`.

## Paper-reported reference results

The paper reports:
- ROC-AUC: 0.934
- F1: 0.887
- Delay MAE: 1.18 days
- Proposed recovery cost: $1,070,000
- Fulfillment: 95.8%
- Lead-time delay: 31.4 hours
- CSI: 24.6%

These are reported paper results, not automatically reproduced by the demo dataset. Reproduction requires the original dataset and matching experimental protocol.
