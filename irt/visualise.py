import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Callable, Sequence, Tuple

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

# ---------------------------
# 2) Score distribution
# ---------------------------
def plot_score_distribution(
    df: pd.DataFrame,
    account_col: str = "AccountId",
    score_col: str = "Score",
    bins: int = 20,
    warn_inconsistent: bool = True,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
):
    """
    Plot a histogram of *raw scores* (not percentiles).
    Returns (fig, ax). Does NOT call plt.show().
    """
    # Identify account column (fallback to participant_id)
    if account_col not in df.columns:
        account_col = "participant_id"
        if account_col not in df.columns:
            raise ValueError(f"Column '{account_col}' not found.")

    if score_col not in df.columns:
        raise ValueError(f"Column '{score_col}' not found.")

    # If a Simulation column exists, assume caller pre-filtered; otherwise just take
    # one score per account (consistent with percentile plotter)
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
        raise ValueError("No valid numeric scores to plot.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    ax.hist(scores, bins=bins, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Raw score")
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(title or f"Raw Score Distribution (n={len(scores)})")

    return fig, ax

# ---------------
def plot_icc(
    item_params_df: pd.DataFrame,
    items: Optional[List] = None,       # optional subset of QuestionIDs
    ax: Optional[plt.Axes] = None,
    title: str = "Item Characteristic Curves",
    show_legend: bool = True
):
    """
    Plot ICCs (Item Characteristic Curves) for item parameters.

    Accepts either:
      - 3PL form: columns ['QuestionID' or 'item_id', 'a', 'b', 'c']
      - 2PL form: columns ['QuestionID' or 'item_id', 'a', 'b']

    Returns (fig, ax). Does NOT call plt.show().
    """

    # --- Identify identifier column ---
    if "QuestionID" in item_params_df.columns:
        id_col = "QuestionID"
    elif "item_id" in item_params_df.columns:
        id_col = "item_id"
    else:
        raise ValueError("Expected column 'QuestionID' or 'item_id' in item_params_df.")

    # --- Check for parameter columns ---
    has_c = "c" in item_params_df.columns
    required = {"a", "b"}
    missing = required - set(item_params_df.columns)
    if missing:
        raise ValueError(f"Missing columns in item_params_df: {sorted(missing)}")

    df = item_params_df.copy()
    if not has_c:
        df["c"] = 0.0  # default for 2PL or Rasch models

    if items is not None:
        df = df[df[id_col].isin(items)]
        if df.empty:
            raise ValueError("No matching items to plot after filtering.")

    theta = np.linspace(-4, 4, 200)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    for _, row in df.iterrows():
        a, b, c = float(row["a"]), float(row["b"]), float(row["c"])
        P = c + (1 - c) / (1 + np.exp(-a * (theta - b)))
        label = f"{id_col[0].upper()}{row[id_col]}"
        ax.plot(theta, P, label=label)

    ax.set_xlabel("Ability (θ)")
    ax.set_ylabel("P(correct)")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    if show_legend:
        ax.legend(frameon=False)

    return fig, ax

# ---------------------------------------
# ---------------------------------------
# 4) Combine Plots
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

    # Hide any unused axes
    for ax in axes_list[len(plotters):]:
        ax.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, y=0.98)

    if tight:
        fig.tight_layout()

    return fig, axes
