from pathlib import Path
import sys
from tune_a_value import tune_discrimination_a

# Compute project directories
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))


from init_core.simulate import generate_simulated_irt_matrix,add_score_column
from init_core.cat import CATConfig, run_cat
from init_core.evaluation_cat import sweep
from init_core.viz import plot_icc,simulated_plot_comparison
from utils.io_utils import validate_csv_paths
from init_core.mechanics import MechanicsPredictor

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = PROJECT_ROOT / "data"

QUICKCALC_CLEANED_RESPONSES = DATA_DIR / "interim" / "QuickCalc" / "user_responses.csv"

model = "3PL"
ABILITIES_27 = DATA_DIR / "processed" / "QuickCalc" / f"abilities_{model}_27items.csv"
ITEMS_27 = DATA_DIR / "processed" / "QuickCalc" / f"item_params_{model}_27items.csv"
IRT_MATRIX_27 = DATA_DIR / "processed" / "QuickCalc" / "IRTMatrix_27items.csv"

ABILITIES_135 = DATA_DIR / "processed" / "QuickCalc" / f"abilities_{model}_135items.csv"
ITEMS_135 = DATA_DIR / "processed" / "QuickCalc" / f"item_params_{model}_135items.csv"
IRT_MATRIX_135 = DATA_DIR / "processed" / "QuickCalc" / "IRTMatrix_135items.csv"

SIM_OUTPUT_DIR =  DATA_DIR / "simulated" / "QuickCalc" 

TUNING_DIR = DATA_DIR / "tuning" / "QuickCalc"  
NEW_ITEM_PARAMS = DATA_DIR / "tuning" / "QuickCalc"  / "tuned_item_params.csv"

ITEM_MECH = DATA_DIR / "interim" / "QuickCalc" /  "item_mechanics.csv"
NEW_ITEM_MECH = DATA_DIR / "generated" / "QuickCalc" /  "new_item_mechanics.csv"


paths = [
    ABILITIES_27,
    ITEMS_27,
    IRT_MATRIX_27,
    ABILITIES_135,
    ITEMS_135,
    IRT_MATRIX_135,
]

def assert_quickcalc_paths():
    results = validate_csv_paths(paths)

    invalid = [p for p, ok in results.items() if not ok]
    if invalid:
        msg = "❌ Invalid CSV path(s):\n"
        msg += "\n".join(f" - {p}" for p in invalid)
        raise FileNotFoundError(msg)

    print("✔ All CSV paths are valid and reachable.")

def compare_sim_with_raw():
    sim_irt = generate_simulated_irt_matrix(abilities_df=pd.read_csv(ABILITIES_135),
                                            item_params_df=pd.read_csv(ITEMS_135),
                                            account_col="participant_id")
    sim = add_score_column(sim_irt)

    raw_irt = pd.read_csv(IRT_MATRIX_135)
    raw = add_score_column(raw_irt)

    title = f"Observed vs Simulated Score Distributions ({model}; 135 Items)"
    fig = simulated_plot_comparison("QuickCalc",
                                    simulated_scores_df=sim,
                                    scores_df=raw,
                                    title=title,
                                    save_path=f"{SIM_OUTPUT_DIR}/simulated_plot_comparison.png")
    plt.show()

def save_tuned_items_with_cat(visualise_cat=False):
    candidate_as = [round(a, 2) for a in np.arange(1, 50.00, 1)]
    num_items = 27

    best_a, df = tune_discrimination_a(
        num_items=num_items ,
        candidate_as=candidate_as,
        trials_per_theta=50,
        theta_grid=np.linspace(-5, 5, 25)
    )

    print(f"Best a: {best_a}")

    b_values = np.linspace(-5, 5, num_items)

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
        plt.tight_layout()
        plt.show()
  
def plot_tuned_icc():
    tuned_items_df = pd.read_csv(NEW_ITEM_PARAMS)
    print(tuned_items_df.head())
    fig =plot_icc(item_params_df = pd.read_csv(NEW_ITEM_PARAMS),
                  assessment_name="QuickCalc",
                  label_mode="legend_in",
                  legend_max_cols=1
                  )
    plt.show()

def generate_item_mechanics():
    mech_cols = ["difficulty", "bonus", "speedup", "releaseInterval", "levelUpHits"]

    int_mechs = ["difficulty", "bonus", "speedup", "releaseInterval", "levelUpHits"]

    mp = MechanicsPredictor(
        k=5,
        weights="distance",
        feature_cols=["b"],          # <--- only b
        int_mech_cols=int_mechs,
        
    )

    mp.fit(pd.read_csv(ITEM_MECH), mech_cols=mech_cols)

    # For a new IRT bank with only a,b:
    pred_mechanics = mp.predict(pd.read_csv(NEW_ITEM_PARAMS))
    pred_mechanics.to_csv(NEW_ITEM_MECH, index=False)


if __name__ == "__main__":
   generate_item_mechanics()