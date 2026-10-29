"""
Main Execution Module for Interpretable Multi-Tier Supply Chain Decision Intelligence
"""

from pathlib import Path
import pandas as pd

from src.config import OUTPUT_DIR, RISK_THRESHOLD
from src.data_pipeline import (
    load_data,
    engineer_features,
    build_feature_matrix,
    TARGET,
)
from src.models import train_pipeline
from src.explainability import compute_shap, plot_parallel_coordinates
from src.optimizer import solve_recovery_optimization
from src.baselines import run_baseline_strategies
from src.evaluation import (
    save_metrics,
    save_recovery_plot,
    plot_sankey_recovery_flow,
    plot_pareto_frontier,
)


def main():
    print("=" * 60)
    print("Interpretable Multi-Tier Supply Chain Decision Intelligence")
    print("=" * 60)

    # 1. Data Ingestion & Feature Engineering
    print("\n[Step 1/6] Loading and engineering supply chain data...")
    df_raw, is_real = load_data()
    if is_real:
        print(" -> Mode: Loaded custom dataset from data/supply_chain.csv")
    else:
        print(" -> Mode: Generated synthetic demo dataset")

    df = engineer_features(df_raw)
    X = build_feature_matrix(df)
    y = df[TARGET]

    print(f" -> Processed {len(df)} records across {X.shape[1]} features.")

    # 2. Predictive Ensemble Modeling
    print("\n[Step 2/6] Training predictive ensemble models...")
    model, metrics, baselines = train_pipeline(X, y)
    
    print(f" -> Model Evaluation Metrics:")
    print(f"    - ROC-AUC Score : {metrics['roc_auc']:.4f}")
    print(f"    - F1 Score      : {metrics['f1']:.4f}")

    # 3. Model Explainability (TreeSHAP)
    print("\n[Step 3/6] Generating SHAP explainability artifacts...")
    mean_abs_shap, local_exp = compute_shap(model, X, OUTPUT_DIR)
    print(" -> Generated SHAP Feature Importance Bar Chart & Beeswarm Plot.")

    # 4. Parallel Coordinates Analysis
    print("\n[Step 4/6] Generating Parallel Coordinates multi-attribute risk plot...")
    plot_parallel_coordinates(df, OUTPUT_DIR)
    print(" -> Parallel coordinates visualization saved.")

    # 5. Prescriptive Recovery Optimization
    print("\n[Step 5/6] Running MILP recovery optimizer and baseline strategies...")
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.predict(X)

    df["predicted_risk"] = probs
    high_risk_suppliers = df[df["predicted_risk"] >= RISK_THRESHOLD]

    milp_results = solve_recovery_optimization(df, high_risk_suppliers)
    baseline_results = run_baseline_strategies(df)

    recovery_summary = pd.concat([baseline_results, milp_results], ignore_index=True)
    recovery_summary.to_csv(OUTPUT_DIR / "recovery_strategy_comparison.csv", index=False)
    print(" -> MILP Optimization vs Baseline comparisons completed.")

    # 6. Evaluation Diagrams
    print("\n[Step 6/6] Rendering final evaluation and InfoVis diagrams...")
    save_metrics(metrics, baselines, OUTPUT_DIR)
    save_recovery_plot(recovery_summary, OUTPUT_DIR)
    plot_sankey_recovery_flow(OUTPUT_DIR)
    plot_pareto_frontier(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"Pipeline executed successfully! All visual artifacts saved to:\n{OUTPUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()