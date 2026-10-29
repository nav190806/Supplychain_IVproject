from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

from .config import (
    DATA_PATH, RANDOM_STATE, DELAY_THRESHOLD_DAYS,
    SCI_ALPHA, SCI_BETA, SCI_GAMMA, GSI_WEIGHTS, COLUMN_MAP
)

FEATURES = [
    "lead_time_days",
    "lead_time_variance",
    "historical_delay_days",
    "order_volume",
    "component_exclusivity",
    "financial_stability",
    "geopolitical_stress",
    "climate_stress",
    "port_congestion",
    "SCI",
    "GSI",
]

TARGET = "disruption_flag"
DELAY_TARGET = "delay_target_days"


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    reverse = {v: k for k, v in COLUMN_MAP.items() if v in df.columns}
    return df.rename(columns=reverse)


def generate_demo_data(n=12000, seed=RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    suppliers = [f"S{i:03d}" for i in range(1, 81)]
    regions = ["Asia", "Europe", "North America", "South America", "Middle East"]
    modes = ["Sea", "Air", "Road", "Rail"]

    supplier_id = rng.choice(suppliers, n)
    region = rng.choice(regions, n, p=[.30, .20, .25, .10, .15])
    transit_mode = rng.choice(modes, n, p=[.50, .18, .22, .10])

    lead_time = np.clip(rng.normal(12, 4, n), 2, 35)
    historical_delay = np.clip(rng.gamma(2.0, 1.3, n), 0, 18)
    order_volume = rng.lognormal(5.0, .8, n)
    exclusivity = rng.beta(2.5, 2.0, n)
    financial_stability = rng.beta(7, 2, n)
    geopolitical = rng.beta(2.0, 4.0, n)
    climate = rng.beta(2.0, 5.0, n)
    port = rng.beta(2.2, 3.0, n)

    # Latent disruption mechanism: risk rises with unstable lead time,
    # delay history, critical supplier dependency and external stress.
    supplier_volume_share = pd.Series(order_volume).groupby(supplier_id).transform(
        lambda x: x / x.sum()
    ).to_numpy()
    sci = (
        SCI_ALPHA * supplier_volume_share
        + SCI_BETA * exclusivity
        + SCI_GAMMA * (1 - financial_stability)
    )
    gsi = (
        GSI_WEIGHTS["geopolitical_stress"] * geopolitical
        + GSI_WEIGHTS["climate_stress"] * climate
        + GSI_WEIGHTS["port_congestion"] * port
    )
    lead_var_proxy = np.maximum(
        0.15, 0.12 * lead_time + 0.35 * historical_delay + rng.normal(0, .4, n)
    )

    logit = (
        -4.0
        + 0.17 * lead_var_proxy
        + 0.20 * historical_delay
        + 3.0 * sci
        + 2.0 * gsi
        + 0.012 * np.log1p(order_volume)
        + 0.25 * exclusivity
        - 1.0 * financial_stability
    )
    probability = 1 / (1 + np.exp(-logit))
    disruption = rng.binomial(1, np.clip(probability, .01, .98))

    delay_target = np.clip(
        0.6 + 0.55 * historical_delay + 1.4 * disruption
        + 2.0 * gsi + 1.0 * lead_var_proxy
        + rng.normal(0, .9, n),
        0, None
    )

    return pd.DataFrame({
        "supplier_id": supplier_id,
        "region": region,
        "transit_mode": transit_mode,
        "lead_time_days": lead_time,
        "historical_delay_days": historical_delay,
        "order_volume": order_volume,
        "component_exclusivity": exclusivity,
        "financial_stability": financial_stability,
        "geopolitical_stress": geopolitical,
        "climate_stress": climate,
        "port_congestion": port,
        "lead_time_variance": lead_var_proxy,
        "SCI": sci,
        "GSI": gsi,
        "disruption_flag": disruption,
        "delay_target_days": delay_target,
    })


def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, bool]:
    if path.exists():
        df = pd.read_csv(path)
        df = _normalise_columns(df)
        return df, True
    return generate_demo_data(), False


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure numerical fields exist.
    numeric_defaults = {
        "lead_time_days": 12.0,
        "historical_delay_days": 1.0,
        "order_volume": 100.0,
        "component_exclusivity": 0.5,
        "financial_stability": 0.75,
        "geopolitical_stress": 0.25,
        "climate_stress": 0.20,
        "port_congestion": 0.30,
    }
    for c, default in numeric_defaults.items():
        if c not in df.columns:
            df[c] = default
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())

    if "supplier_id" not in df.columns:
        df["supplier_id"] = np.arange(len(df)).astype(str)
    if "region" not in df.columns:
        df["region"] = "Unknown"
    if "transit_mode" not in df.columns:
        df["transit_mode"] = "Unknown"

    # Rolling/variance-style proxy when temporal ordering is unavailable.
    if "lead_time_variance" not in df.columns:
        supplier_mean = df.groupby("supplier_id")["lead_time_days"].transform("mean")
        df["lead_time_variance"] = (
            (df["lead_time_days"] - supplier_mean) ** 2
        ).fillna(0.0)

    # SCI: paper's formula. V_i / sum_k V_k is approximated by supplier
    # share of observed order volume.
    supplier_volume = df.groupby("supplier_id")["order_volume"].transform("sum")
    total_volume = max(df["order_volume"].sum(), 1e-9)
    volume_fraction = supplier_volume / total_volume

    df["SCI"] = (
        SCI_ALPHA * volume_fraction
        + SCI_BETA * df["component_exclusivity"]
        + SCI_GAMMA * (1 - df["financial_stability"])
    )

    df["GSI"] = sum(
        weight * df[name] for name, weight in GSI_WEIGHTS.items()
    )

    if TARGET not in df.columns:
        if "delay_target_days" in df.columns:
            df[TARGET] = (df["delay_target_days"] > DELAY_THRESHOLD_DAYS).astype(int)
        else:
            df[TARGET] = (
                df["historical_delay_days"] + df["lead_time_variance"] > DELAY_THRESHOLD_DAYS
            ).astype(int)

    if DELAY_TARGET not in df.columns:
        df[DELAY_TARGET] = (
            df["historical_delay_days"] + df["lead_time_variance"] ** 0.5
        )

    return df


def build_feature_matrix(df: pd.DataFrame):
    X = df[FEATURES].copy()
    categorical = df[["region", "transit_mode"]].astype(str)
    X = pd.concat([X, pd.get_dummies(categorical, drop_first=False)], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X
