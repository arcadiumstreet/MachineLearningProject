from pathlib import Path

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

# Optional: Verify it works when you run this file directly
if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Dir:     {DATA_DIR}")
