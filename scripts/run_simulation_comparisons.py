from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

DATA_DIR = PROJECT_ROOT / "data"

from init_core.simulate import generate_simulated_irt_matrix, add_score_column
from init_core.viz import simulated_plot_comparison, plot_icc
from init_core.diagnostics.compare_distributions import (
    get_distribution_comparison_summary,
    print_distribution_comparison_summary,
)

model = "2pl"

# --- File paths ---

G_ABILITIES = DATA_DIR / "processed" / "Gyrate" / f"abilities_{model}.csv"
G_ITEMS = DATA_DIR / "processed" / "Gyrate" / f"item_params_{model}.csv"
G_IRT_MATRIX = DATA_DIR / "processed" / "Gyrate" / "IRTMatrix.csv"

G_ABILITIES_EXT = DATA_DIR / "processed" / "Gyrate" / f"abilities_{model}_EXT.csv"
G_ITEMS_EXT = DATA_DIR / "processed" / "Gyrate" / f"item_params_{model}_EXT.csv"
G_IRT_MATRIX_EXT = DATA_DIR / "processed" / "Gyrate" / "IRTMatrix_EXT.csv"

Q_ABILITIES_27 = DATA_DIR / "processed" / "QuickCalc" / f"abilities_{model}_27items.csv"
Q_ITEMS_27 = DATA_DIR / "processed" / "QuickCalc" / f"item_params_{model}_27items.csv"
Q_IRT_MATRIX_27 = DATA_DIR / "processed" / "QuickCalc" / "IRTMatrix_27items.csv"

Q_ABILITIES = DATA_DIR / "processed" / "QuickCalc" / f"abilities_{model}.csv"
Q_ITEMS = DATA_DIR / "processed" / "QuickCalc" / f"item_params_{model}.csv"
Q_IRT_MATRIX_27 = DATA_DIR / "processed" / "QuickCalc" / "IRTMatrix_27items.csv"

Q_ABILITIES_135 = DATA_DIR / "processed" / "QuickCalc" / f"abilities_{model}_135items.csv"
Q_ITEMS_135 = DATA_DIR / "processed" / "QuickCalc" / f"item_params_{model}_135items.csv"
Q_IRT_MATRIX_135 = DATA_DIR / "processed" / "QuickCalc" / "IRTMatrix_135items.csv"

E_ABILITIES = DATA_DIR / "processed" / "EyeBall" / f"abilities_{model}.csv"
E_ITEMS = DATA_DIR / "processed" / "EyeBall" / f"item_params_{model}.csv"
E_IRT_MATRIX = DATA_DIR / "processed" / "EyeBall" / "IRTMatrix.csv"

P_ABILITIES = DATA_DIR / "processed" / "Pyramids" / f"abilities_{model}.csv"
P_ITEMS = DATA_DIR / "processed" / "Pyramids" / f"item_params_{model}.csv"
P_IRT_MATRIX = DATA_DIR / "processed" / "Pyramids" / "IRT_Matrix.csv"


M_ABILITIES = DATA_DIR / "processed" / "MemoryGrid" / f"abilities_{model}.csv"
M_ITEMS = DATA_DIR / "processed" / "MemoryGrid" / f"item_params_{model}.csv"
M_IRT_MATRIX = DATA_DIR / "processed" / "MemoryGrid" / "IRT_Matrix.csv"




def _simulate_and_compare_and_plot(
    assessment_name: str,
    title: str,
    abilities_path: Path,
    items_path: Path,
    irt_matrix_path: Path,
    ax: plt.Axes,
) -> None:

    # --- Raw scores ---
    raw_irt = pd.read_csv(irt_matrix_path)
    raw = add_score_column(raw_irt)

    # --- Simulated scores ---
    sim_irt = generate_simulated_irt_matrix(
        abilities_df=pd.read_csv(abilities_path),
        item_params_df=pd.read_csv(items_path),
        account_col="participant_id",
    )
    sim = add_score_column(sim_irt)

    # --- Stats summary ---
    summary = get_distribution_comparison_summary(
        raw_scores=raw["Score"],
        simulated_scores=sim["Score"],
    )

    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)
    print_distribution_comparison_summary(summary)

    # --- Plot onto given axis ---
    simulated_plot_comparison(
        assessment_name=assessment_name,
        scores_df=raw,
        simulated_scores_df=sim,
        title=title,
        ax=ax,
    )


def compose_all_sim_plots_one_figure():
    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    axes = axes.flatten()  # 0,1,2,3,4

    # --- Gyrate original ---
    _simulate_and_compare_and_plot(
        assessment_name="Gyrate",
        title="Gyrate (original item set, 2PL)",
        abilities_path=G_ABILITIES,
        items_path=G_ITEMS,
        irt_matrix_path=G_IRT_MATRIX,
        ax=axes[0],
    )

    # --- Gyrate EXT ---
    _simulate_and_compare_and_plot(
        assessment_name="Gyrate",
        title="Gyrate (extended item set, 2PL)",
        abilities_path=G_ABILITIES_EXT,
        items_path=G_ITEMS_EXT,
        irt_matrix_path=G_IRT_MATRIX_EXT,
        ax=axes[1],
    )

    # --- QuickCalc 27 items ---
    _simulate_and_compare_and_plot(
        assessment_name="QuickCalc",
        title="QuickCalc (original item set, 2PL)",
        abilities_path=Q_ABILITIES_27,
        items_path=Q_ITEMS_27,
        irt_matrix_path=Q_IRT_MATRIX_27,
        ax=axes[2],
    )

    # --- QuickCalc 135 items ---
    _simulate_and_compare_and_plot(
        assessment_name="QuickCalc",
        title="QuickCalc (extended item set, 2PL)",
        abilities_path=Q_ABILITIES_135,
        items_path=Q_ITEMS_135,
        irt_matrix_path=Q_IRT_MATRIX_135,
        ax=axes[3],
    )
        # --- QuickCalc 135 items ---
    _simulate_and_compare_and_plot(
        assessment_name="EyeBa::",
        title="EyeBall (2PL)",
        abilities_path=E_ABILITIES,
        items_path=E_ITEMS,
        irt_matrix_path=E_IRT_MATRIX,
        ax=axes[4],
    )


    fig.suptitle(
        "Simulated vs Observed Score Distributions Across Models",
        fontsize=18,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()
    return fig


def compose_all_sim_plots_one_figure():
    fig, axes = plt.subplots(1, 5, figsize=(16, 5))
    axes = axes.flatten()  # 0,1,2,3,4

    # --- Gyrate original ---
    _simulate_and_compare_and_plot(
        assessment_name="Gyrate",
        title=f"Gyrate ({model})",
        abilities_path=G_ABILITIES,
        items_path=G_ITEMS,
        irt_matrix_path=G_IRT_MATRIX,
        ax=axes[0],
    )


    # --- QuickCalc 27 items ---
    _simulate_and_compare_and_plot(
        assessment_name="QuickCalc",
        title=f"QuickCalc ({model})",
        abilities_path=Q_ABILITIES_27,
        items_path=Q_ITEMS_27,
        irt_matrix_path=Q_IRT_MATRIX_27,
        ax=axes[1],
    )

        # --- Eyeball 135 items ---
    _simulate_and_compare_and_plot(
        assessment_name="EyeBall",
        title=f"EyeBall ({model})",
        abilities_path=E_ABILITIES,
        items_path=E_ITEMS,
        irt_matrix_path=E_IRT_MATRIX,
        ax=axes[2],
    )
    _simulate_and_compare_and_plot(
        assessment_name="Pyramids",
        title=f"Pyramids ({model})",
        abilities_path=P_ABILITIES,
        items_path=P_ITEMS,
        irt_matrix_path=P_IRT_MATRIX,
        ax=axes[3],
    )
    _simulate_and_compare_and_plot(
        assessment_name="MemoryGrid",
        title=f"MemoryGrid ({model})",
        abilities_path=M_ABILITIES,
        items_path=M_ITEMS,
        irt_matrix_path=M_IRT_MATRIX,
        ax=axes[4],
    )



    fig.tight_layout(rect=[0, 0, 1, 0.95])

    #plt.show()
    return fig

def compose_icc_plots_one_figure():
    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    # ✅ remove the 6th axis visually
    axes[5].axis("off")

    plot_icc(pd.read_csv(G_ITEMS), assessment_name="Gyrate", ax=axes[0])
    plot_icc(pd.read_csv(Q_ITEMS), assessment_name="QuickCalc", ax=axes[1])
    plot_icc(pd.read_csv(E_ITEMS), assessment_name="Gyrate", ax=axes[2])
    plot_icc(pd.read_csv(P_ITEMS), assessment_name="Pyramids", ax=axes[3])
    plot_icc(pd.read_csv(M_ITEMS), assessment_name="MatchBack", ax=axes[4])

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    return fig
if __name__ == "__main__":
    compose_icc_plots_one_figure()
