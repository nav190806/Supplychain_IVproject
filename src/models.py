from __future__ import annotations
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

from src.config import RANDOM_STATE, TEST_SIZE


class EnsembleClassifier:
    """Soft-voting ensemble of LightGBM and XGBoost models."""

    def __init__(self, random_state=RANDOM_STATE):
        self.lgb = LGBMClassifier(random_state=random_state, verbosity=-1)
        self.xgb = XGBClassifier(random_state=random_state, eval_metric="logloss")

    def fit(self, X, y):
        self.lgb.fit(X, y)
        self.xgb.fit(X, y)
        return self

    def predict_proba(self, X):
        prob_lgb = self.lgb.predict_proba(X)[:, 1]
        prob_xgb = self.xgb.predict_proba(X)[:, 1]
        ensemble_prob = 0.5 * prob_lgb + 0.5 * prob_xgb
        return np.column_stack([1 - ensemble_prob, ensemble_prob])

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


def evaluate_model(model, X_test, y_test):
    """Calculates standard classification metrics."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        probs = model.predict(X_test)

    preds = (probs >= 0.5).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
    }


def train_pipeline(X: pd.DataFrame, y: pd.Series):
    """
    Trains baseline models and the proposed LightGBM+XGBoost Ensemble,
    returning the trained ensemble, its evaluation metrics, and baseline scores.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 1. Train Baselines
    baselines = {}

    # Baseline 1: Majority Class Dummy Classifier
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    baselines["Majority Baseline"] = evaluate_model(dummy, X_test, y_test)

    # Baseline 2: Random Forest Classifier
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100)
    rf.fit(X_train, y_train)
    baselines["Random Forest"] = evaluate_model(rf, X_test, y_test)

    # 2. Train Proposed Ensemble Model (LightGBM + XGBoost)
    ensemble = EnsembleClassifier(random_state=RANDOM_STATE)
    ensemble.fit(X_train, y_train)

    metrics = evaluate_model(ensemble, X_test, y_test)

    return ensemble, metrics, baselines