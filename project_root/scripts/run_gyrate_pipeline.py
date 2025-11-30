from pathlib import Path
import sys
from tune_a_value import tune_discrimination_a
from typing import Iterable

# Compute project directories
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from games.quickcalc.preprocessing import analyse
from init_core.simulate import generate_simulated_irt_matrix,add_score_column
from init_core.cat import CATConfig, run_cat
from init_core.evaluation_cat import sweep
from init_core.viz import plot_icc,simulated_plot_comparison
from utils.io_utils import validate_csv_paths

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

model = "2pl"
ABILITIES = DATA_DIR / "processed" / "Gyrate" / f"abilities_{model}.csv"
ITEMS = DATA_DIR / "processed" / "Gyrate" / f"item_params_{model}.csv"
IRT_MATRIX = DATA_DIR / "processed" / "Gyrate" / "IRTMatrix.csv"

ABILITIES_EXT = DATA_DIR / "processed" / "Gyrate" / f"abilities_{model}_EXT.csv"
ITEMS_EXT = DATA_DIR / "processed" / "Gyrate" / f"item_params_{model}_EXT.csv"
IRT_MATRIX_EXT = DATA_DIR / "processed" / "Gyrate" / "IRTMatrix_EXT.csv"

SIM_OUTPUT_DIR =  DATA_DIR / "simulated" / "Gyrate" 

TUNING_DIR = DATA_DIR / "tuning" / "Gyrate"  
NEW_ITEM_PARAMS = DATA_DIR / "tuning" / "Gyrate" / "tuned_item_params.csv"


paths = [
    ABILITIES,
    ITEMS,
    IRT_MATRIX,
    ABILITIES_EXT,
    ITEMS_EXT,
    IRT_MATRIX_EXT,
]

def assert_gyrate_paths():
    results = validate_csv_paths(paths)

    invalid = [p for p, ok in results.items() if not ok]
    if invalid:
        msg = "❌ Invalid CSV path(s):\n"
        msg += "\n".join(f" - {p}" for p in invalid)
        raise FileNotFoundError(msg)

    print("✔ All CSV paths are valid and reachable.")

def compare_sim_with_raw():
    sim_irt = generate_simulated_irt_matrix(abilities_df=pd.read_csv(ABILITIES),
                                            item_params_df=pd.read_csv(ITEMS),
                                            account_col="participant_id")
    sim = add_score_column(sim_irt)

    raw_irt = pd.read_csv(IRT_MATRIX)
    raw = add_score_column(raw_irt)

    title = f"Gyrate: Observed vs Simulated Score Distributions ({model})"
    
    fig = simulated_plot_comparison("Gyrate",
                                    simulated_scores_df=sim,
                                    scores_df=raw,
                                    title=title,
                                    save_path=f"{SIM_OUTPUT_DIR}/simulated_plot_comparison.png")
    plt.show()

def save_tuned_items_with_cat(visualise_cat=False):
    candidate_as = [round(a, 2) for a in np.arange(1, 50.00, 1)]
    num_items = len(pd.read_csv(ITEMS))
    print(num_items)

    best_a, df = tune_discrimination_a(
        num_items=num_items ,
        candidate_as=candidate_as,
        trials_per_theta=50,
        theta_grid=np.linspace(-6, 6, 25)
    )

    print(f"Best a: {best_a}")

    b_values = np.linspace(-3, 3, num_items)

    tuned_items_df = pd.DataFrame({
        "item_id": np.arange(1, num_items + 1),
        "a": best_a,
        "b": b_values,
    })

    NEW_ITEM_PARAMS.parent.mkdir(parents=True, exist_ok=True)
    tuned_items_df.to_csv(NEW_ITEM_PARAMS, index=False)
    print(f"\nSaved tuned item params → {NEW_ITEM_PARAMS}")
    print(tuned_items_df.head())

    if visualise_cat:
        summary = (
            df.groupby("a", as_index=False)["rmse_overall"]
              .first()
              .sort_values("a")
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(summary["a"], summary["rmse_overall"], marker="o")
        ax.set_xlabel("Discrimination (a)")
        ax.set_ylabel("RMSE over θ")
        ax.set_title(f"{num_items} items CAT tuning: RMSE vs a")
        ax.grid(True, linewidth=0.5, alpha=0.5)

        save_path = Path(f"{TUNING_DIR}/RMSE_vs_a.png")  # ensure it's a Path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

        plt.tight_layout()
        plt.show()
  
def plot_tuned_icc():
    tuned_items_df = pd.read_csv(NEW_ITEM_PARAMS)
    print(tuned_items_df.head())
    save_path = TUNING_DIR / "tuned_icc_plots.png"
    fig =plot_icc(item_params_df = pd.read_csv(NEW_ITEM_PARAMS),
                  assessment_name="Gyrate",
                  label_mode="legend_in",
                  legend_max_cols=1,
                  title="Gyrate: Tuned ICC Plot",
                  save_path=save_path
                  )
    plt.show()
  

if __name__ == "__main__":
   plot_tuned_icc()