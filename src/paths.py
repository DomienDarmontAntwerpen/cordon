"""
Every file location Cordon uses, in one place. Holds only Path constants;
when a file cannot be found, look here first.
"""

from pathlib import Path

# --- Section: roots ---
SRC_DIR = Path(__file__).parent
REPO_ROOT = SRC_DIR.parent

# --- Section: configuration (src/config/) ---
CONFIG_DIR = SRC_DIR / "config"
CONFIG_PATH = CONFIG_DIR / "config.yaml"
PATTERNS_PATH = CONFIG_DIR / "patterns.yaml"
ROUTING_TABLE_PATH = CONFIG_DIR / "routing_table.yaml"

# --- Section: trained models (src/models/) ---
MODELS_DIR = SRC_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"

# --- Section: dashboard (src/server/static/) ---
STATIC_DIR = SRC_DIR / "server" / "static"

# --- Section: data and generated reports (repository root) ---
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"
GOLDEN_SET_PATH = DATA_DIR / "golden_set.jsonl"
