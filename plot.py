import pandas as pd
import matplotlib.pyplot as plt
from irt.visualise import plot_percentile_distribution, plot_score_distribution, plot_icc, compose_plots

scores_df = pd.read_csv("data/QuickCalc/scores_simulated.csv")  
items_df  = pd.read_csv("data/QuickCalc/item_params.csv")

def plot_scores(ax=None):
    return plot_score_distribution(
        df=scores_df,
        account_col="participant_id",
        score_col="Score",
        bins='auto',
        ax=ax,
        title="Raw Score Distribution"
    )

def plot_percentiles(ax=None):
    return plot_percentile_distribution(
        df=scores_df,
        account_col="participant_id",
        score_col="Score",
        bins='auto',
        ax=ax,
        title="Percentile Distribution"
    )

def plot_iccs(ax=None):
    return plot_icc(item_params_df=items_df, ax=ax, title="Item Characteristic Curves")

fig, axes = compose_plots(
    plotters=[plot_scores, plot_percentiles, plot_iccs],  # or just the first two
    ncols=2,
    figsize=(12, 8),
    suptitle="QuickCalc — Raw Scores, Percentiles, and ICCs"
)

# Save (or plt.show())
import os
os.makedirs("data/QuickCalc/plots", exist_ok=True)
fig.savefig("data/QuickCalc/plots/quickcalc_new.png", dpi=300, bbox_inches="tight")
