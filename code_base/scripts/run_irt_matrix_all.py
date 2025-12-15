from pathlib import Path
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

#  Compute project directories
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from utils.io_utils import validate_csv_paths
from init_core.clean import get_cleaned_responses,build_irt_matrix_from_collapsed


RAW_DATA_DIR = PROJECT_ROOT / "data" / "interim"

EYEBALL_CLEAN = RAW_DATA_DIR / "EyeBall" / "cleaned_responses.csv"
MEMORYGRID_CLEAN = RAW_DATA_DIR / "MemoryGrid"/ "cleaned_responses.csv"
PYRAMIDS_CLEAN = RAW_DATA_DIR / "Pyramids"/ "cleaned_responses.csv"
QUICKCALC_CLEAN = RAW_DATA_DIR / "QuickCalc"/ "cleaned_responses.csv"
GYRATE_CLEAN = RAW_DATA_DIR / "Gyrate"/ "cleaned_responses.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "diagnostics" / "labelled_irt"

paths = [
    EYEBALL_CLEAN,
    MEMORYGRID_CLEAN,
    PYRAMIDS_CLEAN,
    QUICKCALC_CLEAN,
    GYRATE_CLEAN
]
data_dict = {
        "QuickCalc": QUICKCALC_CLEAN,
        "EyeBall": EYEBALL_CLEAN,
        "MemoryGrid": MEMORYGRID_CLEAN,
        "Pyramids": PYRAMIDS_CLEAN,
        "Gyrate": GYRATE_CLEAN,
}
def assert_raw_paths():
    results = validate_csv_paths(paths)

    invalid = [p for p, ok in results.items() if not ok]
    if invalid:
        msg = "❌ Invalid CSV path(s):\n"
        msg += "\n".join(f" - {p}" for p in invalid)
        raise FileNotFoundError(msg)

    print("✔ All CSV paths are valid and reachable.")



def save_irt_matrices():

    for name,data in data_dict.items():
        irt = build_irt_matrix_from_collapsed(pd.read_csv(data))
        print(irt.head())
        irt.to_csv(f"{OUTPUT_DIR}/{name}.csv", index=False)



if __name__ == "__main__":
    save_irt_matrices()