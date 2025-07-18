from pathlib import Path
# Configuration for the NVDA stock predictor project
TICKER = "NVDA"
START_DATE = "2019-01-01"
END_DATE = "2024-12-31"

# Get the absolute path to the directory containing config.py
CONFIG_DIR = Path(__file__).resolve().parent

# Assuming config.py is in something like /repo/src/
# Define repo root as two levels up (adjust if your structure is different)
REPO_ROOT = CONFIG_DIR.parent.parent

# Define the project folder name as a constant for flexibility
PROJECT_FOLDER = "NVDA_stock_predictor"

# Now define paths relative to REPO_ROOT, so they are absolute no matter where you run
RAW_DATA_PATH = REPO_ROOT / PROJECT_FOLDER / "data" / "raw" / "nvda_raw_data.csv"
PROCESSED_DATA_PATH = REPO_ROOT / PROJECT_FOLDER / "data" / "processed" / "nvda_processed_data.csv"
SCALER_PATH = REPO_ROOT / PROJECT_FOLDER / "models" / "scaler.pkl"