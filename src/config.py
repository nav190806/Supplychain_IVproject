from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
DATA_PATH = DATA_DIR / "supply_chain.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Paper's operational threshold concept.
DELAY_THRESHOLD_DAYS = 3.0
RISK_THRESHOLD = 0.50

# SCI weights: alpha + beta + gamma = 1
SCI_ALPHA = 0.40
SCI_BETA = 0.30
SCI_GAMMA = 0.30

# GSI weights. These are configurable because the paper defines weighted
# stress factors but does not publish a single production coefficient set.
GSI_WEIGHTS = {
    "geopolitical_stress": 0.40,
    "climate_stress": 0.30,
    "port_congestion": 0.30,
}

COLUMN_MAP = {
    "supplier_id": "supplier_id",
    "region": "region",
    "transit_mode": "transit_mode",
    "lead_time_days": "lead_time_days",
    "historical_delay_days": "historical_delay_days",
    "order_volume": "order_volume",
    "component_exclusivity": "component_exclusivity",
    "financial_stability": "financial_stability",
    "geopolitical_stress": "geopolitical_stress",
    "climate_stress": "climate_stress",
    "port_congestion": "port_congestion",
    "disruption_flag": "disruption_flag",
}
