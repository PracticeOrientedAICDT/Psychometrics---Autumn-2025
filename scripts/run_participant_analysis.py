from pathlib import Path
import sys
import numpy as np
import pandas as pd

#  Compute project directories
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from utils.io_utils import validate_csv_paths
from init_core.clean import get_cleaned_responses
from init_core.viz import visualise_gender_across_data

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

EYEBALL_RAW = RAW_DATA_DIR / "EyeBall.csv"
MATCHBACK_RAW = RAW_DATA_DIR / "MatchBack.csv"
MEMORYGRID_RAW = RAW_DATA_DIR / "MemoryGrid.csv"
NUMBERRECAL_RAW = RAW_DATA_DIR / "NumberRecall.csv"
PYRAMIDS_RAW = RAW_DATA_DIR / "Pyramids.csv"
QUICKCALC_RAW = RAW_DATA_DIR / "QuickCalc.csv"
RAPIDFLAG_RAW = RAW_DATA_DIR / "RapidFlag.csv"
GYRATE_RAW = RAW_DATA_DIR / "Gyrate.csv"

paths = [
    EYEBALL_RAW,
    MATCHBACK_RAW,
    MEMORYGRID_RAW,
    NUMBERRECAL_RAW,
    PYRAMIDS_RAW,
    QUICKCALC_RAW,
    RAPIDFLAG_RAW,
    GYRATE_RAW
]

def assert_raw_paths():
    results = validate_csv_paths(paths)

    invalid = [p for p, ok in results.items() if not ok]
    if invalid:
        msg = "❌ Invalid CSV path(s):\n"
        msg += "\n".join(f" - {p}" for p in invalid)
        raise FileNotFoundError(msg)

    print("✔ All CSV paths are valid and reachable.")

def show_gender():
    data_dict = {
        "QuickCalc": get_cleaned_responses(pd.read_csv(QUICKCALC_RAW)),
        "EyeBall": get_cleaned_responses(pd.read_csv(EYEBALL_RAW)),
        "MatchBack": get_cleaned_responses(pd.read_csv(MATCHBACK_RAW)),
        "MemoryGrid": get_cleaned_responses(pd.read_csv(MEMORYGRID_RAW)),
        "NumberRecall": get_cleaned_responses(pd.read_csv(NUMBERRECAL_RAW)),
        "Pyramids": get_cleaned_responses(pd.read_csv(PYRAMIDS_RAW)),
        "RapidFlag": get_cleaned_responses(pd.read_csv(RAPIDFLAG_RAW)),
        "Gyrate": get_cleaned_responses(pd.read_csv(GYRATE_RAW)),
    }
    visualise_gender_across_data(data_dict, gender_col="Gender", overlay=False)


if __name__ == "__main__":
    show_gender()