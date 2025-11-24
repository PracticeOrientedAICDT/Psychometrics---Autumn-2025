from pathlib import Path
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

#  Compute project directories
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from utils.io_utils import validate_csv_paths
from init_core.clean import get_cleaned_responses
from init_core.viz import (
    visualise_age_curves_across_data,
    visualise_gender_stack_across_data,
    visualise_country_stack_across_data,
    visualise_country_of_origin_stack_across_data,
    visualise_ethnicity_stack_across_data,
    visualise_marital_status_stack_across_data,
    visualise_education_level_stack_across_data,
    visualise_number_of_children_stack_across_data,
    visualise_postcode_stack_across_data,
    visualise_language_stack_across_data,
)

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

EYEBALL_RAW = RAW_DATA_DIR / "EyeBall.csv"
MATCHBACK_RAW = RAW_DATA_DIR / "MatchBack.csv"
MEMORYGRID_RAW = RAW_DATA_DIR / "MemoryGrid.csv"
NUMBERRECAL_RAW = RAW_DATA_DIR / "NumberRecall.csv"
PYRAMIDS_RAW = RAW_DATA_DIR / "Pyramids.csv"
QUICKCALC_RAW = RAW_DATA_DIR / "QuickCalc.csv"
RAPIDFLAG_RAW = RAW_DATA_DIR / "RapidFlag.csv"
GYRATE_RAW = RAW_DATA_DIR / "Gyrate.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "diagnostics"

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

def get_cleaned_dict():
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
    return data_dict

def save_all_demographic_plots(show=False):
    """
    Generate and save ALL demographic plots from init_core.viz
    into data/diagnostics/. Returns a dict of figure objects.
    """
    data_dict = get_cleaned_dict()
    figs = {}

    figs["age_distribution"] = visualise_age_curves_across_data(
        data_dict, show=show
    )
    figs["gender_distribution"] = visualise_gender_stack_across_data(
        data_dict, show=show
    )
    figs["country_of_residence_distribution"] = visualise_country_stack_across_data(
        data_dict, show=show
    )
    #figs["country_of_origin_distribution"] = visualise_country_of_origin_stack_across_data(
    #    data_dict, show=show
    #)
    figs["ethnic_origin_distribution"] = visualise_ethnicity_stack_across_data(
        data_dict, show=show
    )
    #figs["marital_status_distribution"] = visualise_marital_status_stack_across_data(
    #    data_dict, show=show
    #)
    figs["education_level_distribution"] = visualise_education_level_stack_across_data(
        data_dict, show=show
    )
    #figs["number_of_children_distribution"] = visualise_number_of_children_stack_across_data(
    #    data_dict, show=show
    #)
    figs["postcode_distribution"] = visualise_postcode_stack_across_data(
        data_dict, show=show
    )
    figs["language_distribution"] = visualise_language_stack_across_data(
        data_dict, show=show
    )

    # ----------------------
    # SAVE ALL FIGURES
    # ----------------------
    for name, fig in figs.items():
        save_path = OUTPUT_DIR / f"{name}.png"
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return figs


if __name__ == "__main__":
    save_all_demographic_plots()