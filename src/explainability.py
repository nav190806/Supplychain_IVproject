from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import shap


def compute_shap(model, X: pd.DataFrame, output_dir: Path):
    """
    Computes TreeSHAP explanations for tree-based models or ensembles.
    Handles custom wrapper classes like EnsembleClassifier by extracting base estimators.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # If model is our custom EnsembleClassifier, use its underlying LightGBM estimator
    if hasattr(model, "lgb"):
        base_model = model.lgb
    elif hasattr(model, "xgb"):
        base_model = model.xgb
    else:
        base_model = model

    # Initialize TreeExplainer on the base tree model
    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer(X)

    # 1. Feature Importance Bar Plot
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_feature_importance.png", dpi=180)
    plt.close()

    # 2. Beeswarm Plot
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_beeswarm.png", dpi=180)
    plt.close()

    # Calculate mean absolute SHAP values for summary returning
    if len(shap_values.values.shape) == 3:
        # Multi-class output
        mean_abs_shap = pd.Series(
            abs(shap_values.values[:, :, 1]).mean(axis=0), index=X.columns
        )
    else:
        # Binary output
        mean_abs_shap = pd.Series(
            abs(shap_values.values).mean(axis=0), index=X.columns
        )

    return mean_abs_shap, shap_values
def plot_parallel_coordinates(df: pd.DataFrame, output_dir: Path):
    """Generates a parallel coordinates plot for multi-attribute risk analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select numeric columns for display
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()[:5]
    if not numeric_cols:
        return

    plt.figure(figsize=(10, 5))
    pd.plotting.parallel_coordinates(df[numeric_cols], numeric_cols[0], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.title("Multi-Attribute Risk Parallel Coordinates Analysis")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_dir / "parallel_coordinates.png", dpi=180)
    plt.close()