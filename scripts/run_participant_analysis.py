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
from init_core.clean import get_cleaned_responses
from init_core.viz import (
    visualise_age_curves_across_data,
    visualise_gender_stack_across_data,
    visualise_country_stack_across_data,
    visualise_ethnicity_stack_across_data,
    visualise_education_level_stack_across_data,
    visualise_postcode_stack_across_data,
    visualise_language_stack_across_data,

    visualise_score_kde_across_attempt_strategies
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
        "QuickCalc": get_cleaned_responses(pd.read_csv(QUICKCALC_RAW),attempt_mode="first"),
        "EyeBall": get_cleaned_responses(pd.read_csv(EYEBALL_RAW),attempt_mode="first"),
        "MatchBack": get_cleaned_responses(pd.read_csv(MATCHBACK_RAW),attempt_mode="first"),
        "MemoryGrid": get_cleaned_responses(pd.read_csv(MEMORYGRID_RAW),attempt_mode="first"),
        "NumberRecall": get_cleaned_responses(pd.read_csv(NUMBERRECAL_RAW),attempt_mode="first"),
        "Pyramids": get_cleaned_responses(pd.read_csv(PYRAMIDS_RAW),attempt_mode="first"),
        "RapidFlag": get_cleaned_responses(pd.read_csv(RAPIDFLAG_RAW),attempt_mode="first"),
        "Gyrate": get_cleaned_responses(pd.read_csv(GYRATE_RAW),attempt_mode="first"),
    }
    return data_dict

def save_all_demographic_plots(show: bool = False):
    """
    Generate ONE combined demographic figure with subplots
    and external legends.
    """
    data_dict = get_cleaned_dict()

    plot_specs = [
        ("age_distribution", visualise_age_curves_across_data),
        ("gender_distribution", visualise_gender_stack_across_data),
        ("country_of_residence_distribution", visualise_country_stack_across_data),
        ("ethnic_origin_distribution", visualise_ethnicity_stack_across_data),
        ("education_level_distribution", visualise_education_level_stack_across_data),
        ("language_distribution", visualise_language_stack_across_data),
    ]

    n = len(plot_specs)
    n_cols = 2
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 4 * n_rows),
        squeeze=False,
    )
    axes = axes.ravel()
    for ax, (name, fn) in zip(axes, plot_specs):
        fn(data_dict, show=False, ax=ax)

    # Hide empty axes
    for ax in axes[n:]:
        ax.set_visible(False)

    # Title
    fig.suptitle(
        "Demographic Distributions Across Assessments",
        fontsize=18,
        x=0.39,
        y=0.96
    )

    # Allow space for legends on the right
    fig.subplots_adjust(
        right=0.85,    # <-- key: leaves space for external legends
        hspace=0.5,
        wspace=0.5
    )
    fig.tight_layout(rect=(0, 0, 0.78, 0.93))
    save_path = OUTPUT_DIR / "demographic_distributions_combined.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig

def save_assessment_analysis_plots(show: bool = False):

    ASSESSMENT_FILES = {
        "MatchBack": MATCHBACK_RAW,
        "MemoryGrid": MEMORYGRID_RAW,
        "NumberRecall": NUMBERRECAL_RAW,
        "QuickCalc": QUICKCALC_RAW,
        "RapidFlag": RAPIDFLAG_RAW,
        "Gyrate": GYRATE_RAW,
    }

    individual_figs = {}
    combined_fig = None

    # ---------- Individual figures ----------
    for name, path in ASSESSMENT_FILES.items():
        df_raw = pd.read_csv(path, low_memory=False)

        dfs_dict = {
            "first": get_cleaned_responses(df_raw, attempt_mode="first"),
            "last":  get_cleaned_responses(df_raw, attempt_mode="last"),
            "best":  get_cleaned_responses(df_raw, attempt_mode="best"),
            "all":   get_cleaned_responses(df_raw, attempt_mode="all"),
        }

        fig = visualise_score_kde_across_attempt_strategies(
            dfs_dict=dfs_dict,
            assessment_name=name,
            score_col="Score",
            show=False,
            use_counts=False
        )

        #fig.savefig(OUTPUT_DIR / f"{name}_score_attempt_strategies_norm.png", dpi=200, bbox_inches="tight")
        individual_figs[name] = fig

    # ---------- Combined multi-panel figure ----------
    n = len(ASSESSMENT_FILES)
    n_cols = 3
    n_rows = int(np.ceil(n / n_cols))

    combined_fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, (name, path) in zip(axes, ASSESSMENT_FILES.items()):
        df_raw = pd.read_csv(path, low_memory=False)

        dfs_dict = {
            "first": get_cleaned_responses(df_raw, attempt_mode="first"),
            "last":  get_cleaned_responses(df_raw, attempt_mode="last"),
            "best":  get_cleaned_responses(df_raw, attempt_mode="best"),
            "all":   get_cleaned_responses(df_raw, attempt_mode="all"),
        }

        visualise_score_kde_across_attempt_strategies(
            dfs_dict=dfs_dict,
            assessment_name=name,
            score_col="Score",
            show=False,
            use_counts = False,
            ax=ax,  # draw onto this subplot
        )

    # Hide unused axes if any
    for ax in axes[len(ASSESSMENT_FILES):]:
        ax.set_visible(False)

    combined_fig.suptitle(
        "Score Distributions by Attempt Strategy Across Assessments",
        y=0.99,
    )
    combined_fig.tight_layout()

    combined_fig.savefig(
        OUTPUT_DIR / "all_assessments_score_attempt_strategies_norm.png",
        dpi=200,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    return individual_figs, combined_fig

if __name__ == "__main__":
    save_all_demographic_plots()