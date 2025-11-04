import pandas as pd
import os
import matplotlib.pyplot as plt
from irt.visualise import plot_percentile_distribution, plot_icc, compose_plots

# Load the simulated (or expected) scores
scores_df = pd.read_csv("data/QuickCalc/scores_deterministic.csv")  # or scores_simulated.csv
items_df  = pd.read_csv("data/QuickCalc/item_params.csv")      # must have a,b; c optional

# Build two plotters that accept ax=...
def plot_percentiles(ax=None):
    return plot_percentile_distribution(
        df=scores_df,
        account_col="participant_id",   # falls back to this if AccountId absent
        score_col="Score",
        bins=20,
        ax=ax,
        title="QuickCalc Percentile Distribution"
    )

def plot_iccs(ax=None):
    return plot_icc(
        item_params_df=items_df,        # columns: item_id|QuestionID, a, b, (c)
        items=None,                     # or pass a list like ["1","2","3"]
        ax=ax,
        title="QuickCalc Item Characteristic Curves",
        show_legend=True
    )

# Compose into one figure (2 cols)
fig, axes = compose_plots(
    plotters=[plot_percentiles, plot_iccs],
    ncols=2,
    figsize=(12, 5),
    suptitle="QuickCalc — Simulated Scores & ICCs"
)

out_dir = "data/QuickCalc/plots"
os.makedirs(out_dir, exist_ok=True)
fig.savefig(f"{out_dir}/quickcalc_withc_filled.png", dpi=300, bbox_inches="tight")
print(f"✅ Saved {out_dir}/quickcalc_overview.png")
