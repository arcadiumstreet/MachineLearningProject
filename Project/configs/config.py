from pathlib import Path
import os

# 1. Get the path of THIS file (src/config.py)
# 2. Go up parents to find the project root
#    .parent = src/
#    .parent.parent = my_project/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 3. Define standard paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

RESULTS_DIR = PROJECT_ROOT / "results"

if not os.path.exists(RAW_DATA_DIR):
    os.makedirs(RAW_DATA_DIR)
if not os.path.exists(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Optional: Verify it works when you run this file directly
if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Dir:     {DATA_DIR}")
