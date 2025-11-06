import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Callable, Sequence, Tuple
from matplotlib.gridspec import GridSpec
import math
from matplotlib.cm import get_cmap

plt.rcParams["font.family"] = "Times New Roman"     

# ---------------------------
# 1) Percentile distribution
# ---------------------------
def plot_percentile_distribution(
    df: pd.DataFrame,
    account_col: str = "AccountId",
    score_col: str = "Score",
    bins: int = 20,
    warn_inconsistent: bool = True,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
):
    """
    Plot distribution of per-account score percentiles.
    Returns (fig, ax). Does NOT call plt.show().
    """
    if account_col not in df.columns:
        account_col = "participant_id"
        if account_col not in df.columns:
            raise ValueError(f"Column '{account_col}' not found.")
    if score_col not in df.columns:
        raise ValueError(f"Column '{score_col}' not found.")

    # One score per account (warn if inconsistent)
    per_acc = df.groupby(account_col)[score_col].agg(["nunique", "first"])
    if warn_inconsistent:
        inconsistent = per_acc.query("nunique > 1")
        if not inconsistent.empty:
            print(
                f"⚠️ {len(inconsistent)} account(s) have differing '{score_col}' values. "
                f"Using first value per account."
            )

    scores = pd.to_numeric(per_acc["first"], errors="coerce").dropna()
    if scores.empty:
        raise ValueError("No valid numeric scores to compute percentiles.")

    percentiles = scores.rank(method="average", pct=True) * 100.0
    sample_size = len(percentiles)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    ax.hist(percentiles, bins=bins, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Frequency")
    ax.set_xlim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(title or f"WordMatch Percentile Distribution (n={sample_size})")

    return fig, ax

# ---------------
# 2) Score distribution plot
# ---------------
def plot_score_distribution(df,
                            score_col="Score",
                            bins=25,
                            ax: Optional[plt.Axes] = None,
                            title=None,
                            noramlise_x = True):
    
    fig = ax.figure
    scores = df[score_col].dropna()
    x_axis = "Score"
    new_title = "Raw Score Distribution"
    if noramlise_x:
        scores = 100 * (scores - scores.min()) / (scores.max() - scores.min())
        x_axis = "Noramalised Score"
        new_title = "Normalised Score Distribution"
    
    
    ax.hist(scores, bins=bins, edgecolor="black")
    ax.set_xlabel(x_axis)
    ax.set_ylabel("Frequency")
    ax.set_title(title or new_title)
    ax.grid(True, linestyle="--", alpha=0.5)

    return fig,ax

# ---------------
# 3) ICC plotter
# ---------------
def plot_icc(
    item_params_df: pd.DataFrame,
    items: Optional[List] = None,        # optional subset of QuestionIDs
    ax: Optional[plt.Axes] = None,
    assessment_name: str = None,
    label_mode: str = "inline",          # "inline" | "legend" | "none"
    cmap: str = "cividis",               
    figsize: tuple = (12, 6),
    legend_max_cols: int = 2,
    legend_fontsize: int = 8,
    x_range: Optional[Tuple[float, float]] = [-4,4],   # <- set to (min,max) to override
    x_pad: float = 0.0,                               # extra padding if desired
    random_seed: Optional[int] = None                                 # <- extra padding if you want any
):
    """
    Plot ICCs (2PL/3PL) with gradient colors and flexible labelling.

    x_range:
      - None (default): x-axis tightly fits data, i.e. [min(b), max(b)], no gaps.
      - (lo, hi): force a specific x range (e.g., (-4, 4)).

    label_mode:
      - "inline": annotate each curve near θ=b (inflection point)
      - "legend": compact legend below axes, multi-column
      - "none":   no labels
    """

    title= f"{assessment_name}: IRT Analysis"
    # --- Identify identifier column ---
    if "QuestionID" in item_params_df.columns:
        id_col = "QuestionID"
    elif "item_id" in item_params_df.columns:
        id_col = "item_id"
    else:
        raise ValueError("Expected column 'QuestionID' or 'item_id' in item_params_df.")

    # --- Ensure parameters present ---
    if not {"a", "b"}.issubset(item_params_df.columns):
        raise ValueError("item_params_df must include 'a' and 'b'.")
    df = item_params_df.copy()
    if "c" not in df.columns:
        df["c"] = 0.0

    # Optional subset
    if items is not None:
        df = df[df[id_col].isin(items)]
        if df.empty:
            raise ValueError("No matching items to plot after filtering.")

    # Sort by b (helps readability / color order)
    df = df.sort_values(by="b").reset_index(drop=True)

    # ---- Determine x-axis range tightly ----
    if x_range is None:
        b_min = float(df["b"].min())
        b_max = float(df["b"].max())
        # apply tiny/optional padding (default 0.0 for truly no gap)
        lo = b_min - x_pad
        hi = b_max + x_pad
    else:
        lo, hi = map(float, x_range)

    # Ability axis sampled exactly over the plotting domain
    theta = np.linspace(lo, hi, 200)

    # Figure/axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # ---- Colors ----
    n = len(df)

    def _random_colors(n_, seed=None):
        rng = np.random.default_rng(seed)
        # start from tab20 qualitative colors, repeat if needed, then shuffle
        base = np.array(get_cmap("tab20").colors)
        reps = int(np.ceil(n_ / len(base)))
        colors_ = np.vstack([base for _ in range(reps)])[:n_]
        rng.shuffle(colors_, axis=0)
        return [tuple(c) for c in colors_]

    if isinstance(cmap, str) and cmap.lower() == "random":
        colors = _random_colors(n, seed=random_seed)
    else:
        cmap_obj = get_cmap(cmap)   # e.g. "rainbow_bgyr_35_85_c72"

        xs = np.linspace(0,1, n)   # sample nicely inside the map
        colors = [cmap_obj(x) for x in xs]


    # Plot curves
    handles = []
    labels = []
    for i, row in enumerate(df.itertuples(index=False)):
        a, b, c = float(getattr(row, "a")), float(getattr(row, "b")), float(getattr(row, "c"))
        qid = getattr(row, id_col)
        P = c + (1 - c) / (1 + np.exp(-a * (theta - b)))
        h, = ax.plot(theta, P, color=colors[i], linewidth=2)
        handles.append(h)
        labels.append(str(qid))

    # Axes formatting
    ax.set_xlabel("Ability (θ)")
    ax.set_ylabel("P(correct)")
    ax.set_ylim(0, 1)
    ax.set_xlim(lo, hi)     # <- no bigger than requested/domain range
    ax.margins(x=0)         # <- disable extra matplotlib padding on x
    ax.set_title(title,fontweight="bold",pad=25,fontsize=16)
    ax.text(0.5, 1.02, "Item Characteristic Curves",
        transform=ax.transAxes,
        ha="center", va="bottom", fontsize=12)


    # ----- Labeling strategies -----
    if label_mode.lower() == "inline":
        # Place each label at θ ≈ b (inflection), y = c + (1-c)/2; clip to visible range
        for i, row in enumerate(df.itertuples(index=False)):
            a, b, c = float(getattr(row, "a")), float(getattr(row, "b")), float(getattr(row, "c"))
            qid = getattr(row, id_col)

            x = float(np.clip(b, lo, hi))  # keep inside frame
            y = float(np.clip(c + 0.5 * (1 - c), 0.02, 0.98))

            # light jitter to reduce overlap when many b are similar
            jitter = (i % 5 - 2) * 0.01
            y = float(np.clip(y + jitter, 0.02, 0.98))

            ax.annotate(
                str(qid),
                xy=(x, y),
                xytext=(x + 0.02*(hi-lo), y),   # offset scaled to window width
                textcoords="data",
                fontsize=8,
                color="black",
                ha="left",
                va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6),
                arrowprops=dict(arrowstyle="-", color=colors[i], lw=0.8, alpha=0.8),
            )

    elif label_mode.lower() == "legend_out":
        items_per_col = 18
        ncol = min(legend_max_cols, max(1, math.ceil(n / items_per_col)))
        ax.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=ncol,
            frameon=False,
            fontsize=legend_fontsize,
            title=id_col
        )
        fig.tight_layout(rect=[0, 0.05, 1, 1])

    elif label_mode.lower() == "legend_in":
        items_per_col = 8
        ncol = min(legend_max_cols, max(1, math.ceil(n / items_per_col)))
        ax.legend(
            handles, labels,
            loc="upper right",            # put it inside top-right corner
            ncol=ncol,
            frameon=True,                 # small box frame looks neater inside
            fancybox=True,
            framealpha=0.8,
            facecolor="white",
            fontsize=legend_fontsize,
            title=id_col,
            title_fontsize=legend_fontsize + 1
        )
        fig.tight_layout()
    ax.axvline(0, color="lightgrey", linewidth=1, linestyle="--", alpha=1)


    return fig

# ---------------------------------------
# 4) Simulated Plot Comparison

def simulated_plot_comparison(assessment_name,
                          scores_df = None,
                          simulated_scores_df = None
                          ):
    
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(1, figure=fig,ncols=2) 
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    bins = 25

  
    plot_score_distribution(
            scores_df,
            ax=ax1,
            title="Score Distribution: Observed", 
            bins=bins,score_col="Score"
        )
    plot_score_distribution(
            simulated_scores_df,
            ax=ax2,
            title="Score Distribution: Simulation", 
            bins=bins,score_col="Score"
        )
    
    fig.suptitle(f"{assessment_name}: IRT Simulation Comparison", fontsize=16, fontweight="bold")
    return fig
    #fig.tight_layout(rect=[0, 0, 1, 0.95])  
    

# ---------------------------------------
# 5) Combine Plots
# ---------------------------------------
def compose_plots(
    plotters: Sequence[Callable[..., Tuple[plt.Figure, plt.Axes]]],
    ncols: int = 2,
    figsize: Tuple[int, int] = (12, 8),
    suptitle: Optional[str] = None,
    tight: bool = True,
):
    """
    Create a figure and call each plotter with its own Axes in a grid.
    Each plotter must accept a single kwarg 'ax' and draw on it.
    Returns (fig, axes_array).
    """
    if len(plotters) == 0:
        raise ValueError("plotters list is empty.")

    nrows = int(np.ceil(len(plotters) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    axes_list = np.atleast_1d(axes).ravel().tolist()

    for plotter, ax in zip(plotters, axes_list):
        plotter(ax=ax)

    for ax in axes_list[len(plotters):]:
        ax.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, y=0.98)

    if tight:
        fig.tight_layout()

    return fig, axes
