from pathlib import Path
import sys
from tune_a_value import tune_discrimination_a
from scipy.stats import gaussian_kde  # you can use this, it's standard and fine

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
<<<<<<< HEAD
from init_core.mechanics import MechanicsRegressor
=======
#from init_core.mechanics import MechanicsRegressor
from init_core.mechanics import MechanicsToBModel
>>>>>>> e93cbcc (Initial commit from main2.0)
from games.QuickCalc.preprocessing.process import get_irt_matrix

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = PROJECT_ROOT / "data"

QUICKCALC_CLEANED_RESPONSES = DATA_DIR / "interim" / "QuickCalc" / "user_responses.csv"

model = "2PL"
ABILITIES_27 = DATA_DIR / "processed" / "QuickCalc" / f"abilities_{model}_27items.csv"
ITEMS_27 = DATA_DIR / "processed" / "QuickCalc" / f"item_params_{model}.csv"
IRT_MATRIX_27 = DATA_DIR / "processed" / "QuickCalc" / "IRTMatrix_27items.csv"

ABILITIES_135 = DATA_DIR / "processed" / "QuickCalc" / f"abilities_{model}_135items.csv"
ITEMS_135 = DATA_DIR / "processed" / "QuickCalc" / f"item_params_{model}_135items.csv"
IRT_MATRIX_135 = DATA_DIR / "processed" / "QuickCalc" / "IRTMatrix_135items.csv"

SIM_OUTPUT_DIR =  DATA_DIR / "simulated" / "QuickCalc" 

TUNING_DIR = DATA_DIR / "tuning" / "QuickCalc"  
NEW_ITEM_PARAMS = DATA_DIR / "tuning" / "QuickCalc"  / "tuned_item_params.csv"

ITEM_MECH = DATA_DIR / "interim" / "QuickCalc" /  "item_mechanics.csv"
NEW_ITEM_MECH = DATA_DIR / "generated" / "QuickCalc" /  "new_item_mechanics.csv"


EXPERIMENT_DIR = DATA_DIR / "experimental" / "QuickCalc"


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

def save_irt_matrix():
    irt = get_irt_matrix(pd.read_csv(QUICKCALC_CLEANED_RESPONSES),return_wide=True)
    print(len(irt))
    print(irt.head())
    # Drop the participant_id column
    irt = irt.drop(columns=["participant_id"])
    # Save to a CSV file too if needed
    out_path = EXPERIMENT_DIR / "interim" / "IRT_MATRIX_27_.csv"

    irt.to_csv(out_path, index=False)
    print(f"Saved wide matrix to {out_path}")

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

def update_item_mechanics_with_irt(

) -> pd.DataFrame:
    # Load files
    item_mech  = pd.read_csv(ITEM_MECH)
    new_params = pd.read_csv(ITEMS_27)

    # ✅ Shift IRT item_id from 1-based → 0-based
    new_params["item_id"] = new_params["item_id"] - 1

    # ✅ Collapse Level column to produce stable item difficulty per item_id
    # We keep the row whose difficulty matches its Level number when possible
    collapsed = (
        item_mech
        .groupby("level", as_index=False)
        .apply(lambda g:
            g[g["level"] == g["difficulty"]].head(1)
            if any(g["level"] == g["difficulty"])
            else g.head(1)
        )
        .reset_index(drop=True)
    )

    # ✅ Now merge
    merged = collapsed.merge(
        new_params[["item_id", "a", "b"]],
        on="item_id",
        how="left",
        suffixes=("", "_new")
    )
    merged["item_id"] = range(len(merged))

    # ✅ Overwrite a/b
    if "a_new" in merged.columns:
        merged["a"] = merged.pop("a_new")
    if "b_new" in merged.columns:
        merged["b"] = merged.pop("b_new")

    # ✅ Save if requested

    merged.to_csv(ITEM_MECH, index=False)

    return merged


def generate_item_mechanics(target_b=0.0, top_k=27):
    item_mech = pd.read_csv(ITEM_MECH)
    irt = pd.read_csv(ITEMS_27)

    # join mechanics with b (you already do this logic elsewhere)
    df = item_mech.merge(
        irt[["item_id", "b"]],
        left_on="level",
        right_on="item_id",
        how="inner"
    )

    feature_cols = [
        "n_functions",
        "releaseInterval",
        "speedup",
        "bonus",
        "balloon_lifetime_min",
        "balloon_lifetime_max",
        "has_addition",
        "has_subtraction",
        "has_multiplication",
        "has_division",
        "has_fraction",
        "has_percentage",
        "left_min", "left_max",
        "right_min", "right_max",
        "limit_min", "limit_max",
    ]

    model = MechanicsToBModel(alpha=1.0).fit(
        df=df,
        feature_cols=feature_cols,
        target_col="b"
    )

    # generate candidates (reuse your existing grid logic)
    candidates = generate_candidate_mechanics(feature_cols)
    candidates["pred_b"] = model.predict_b(candidates)
    candidates["abs_err"] = (candidates["pred_b"] - target_b).abs()

    out = candidates.sort_values("abs_err").head(top_k)

    NEW_ITEM_MECH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(NEW_ITEM_MECH, index=False)

    print(
        f"Saved generated mechanics → {NEW_ITEM_MECH}\n"
        f"b-model R²={model.stats_['r2']:.3f}, MAE={model.stats_['mae']:.3f}"
    )

def samp_sim_comparison():
    OUT_DIR = Path("/Users/tt25013/Documents/GitHub/Psychometrics---Autumn-2025/data/experimental/QuickCalc")


    # ✅ Load observed (raw) wide matrix
    raw_irt = pd.read_csv(OUT_DIR / "interim/IRT_MATRIX_27.csv")
    raw = add_score_column(raw_irt)
    raw_scores = raw["Score"]

    # ✅ Loop over different subsample IRT fits and generate simulated wide matrices
    sample_sizes = ["", "1000", "2000", "3000", "4000"]

    sim_scores_list = []
    for s in sample_sizes:
        label = "All" if s == "" else s  # for legend
        # Load abilities and item parameters for this sample size
        if s != "":
            s = f"_{s}"
        abilities_path = OUT_DIR / "processed" / f"abilities.csv"
        items_path     = OUT_DIR / "processed" / f"item_params{s}.csv"

        abilities_df = pd.read_csv(abilities_path)
        items_df     = pd.read_csv(items_path)

        # Generate simulated wide IRT matrix using your real function
        sim_irt_df = generate_simulated_irt_matrix(
            abilities_df   = abilities_df,
            item_params_df = items_df,
            account_col    = "participant_id"
        )

        # Compute total scores for simulated people
        sim = add_score_column(sim_irt_df)
        sim_scores = sim["Score"]
        sim_scores_list.append((label, sim_scores))

    # ---------- plotting (raw + all sims) ----------
    # bins based on raw scores
    # with:
    kde_obs = gaussian_kde(raw_scores)
    x_line = np.linspace(0, 27, 300)
    y_obs = kde_obs(x_line)

    plt.figure()
    plt.plot(x_line, y_obs, label="Observed", linewidth=2, color="black")

    for label, scores in sim_scores_list:
        kde_sim = gaussian_kde(scores)
        y_sim = kde_sim(x_line)
        plt.plot(x_line, y_sim, label=f"Simulated (n={label})", alpha=0.8)

    plt.xlabel("Total Score")
    plt.ylabel("Density")
    plt.title("QuickCalc: Sample vs Simulation Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
   generate_item_mechanics()