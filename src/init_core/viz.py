import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Callable, Sequence, Tuple,Union
from matplotlib.gridspec import GridSpec
import math
from matplotlib.cm import get_cmap
from scipy.stats import gaussian_kde
import seaborn as sns
from pathlib import Path
from typing import Dict


#plt.rcParams["font.family"] = "Times New Roman"     
#plt.rcParams["font.family"] = "CMU Sans Serif"
plt.rcParams["font.family"] = "STIX Two Text"
# ---------------------------
# 1) Percentile distribution
# ---------------------------

def plot_percentile_distribution(df: pd.DataFrame,
    account_col: str = "AccountId",
    score_col: str = "Score",
    bins: int = 20,
    warn_inconsistent: bool = True,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None):

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
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4))
    else:
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
    label_mode: str = "legend_in",          # "inline" | "legend" | "none"
    cmap: str = "random",               
    figsize: tuple = (12, 6),
    legend_max_cols: int = 5,
    legend_fontsize: int = 8,
    x_range: Optional[Tuple[float, float]] = [-4,4],   # <- set to (min,max) to override
    x_pad: float = 0.0,                               # extra padding if desired
    random_seed: Optional[int] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = None                                 # <- extra padding if you want any
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
    if title is None:
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
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    return fig

# ---------------------------------------
# 4) Simulated Plot Comparison

def simulated_plot_comparison(
    assessment_name,
    scores_df=None,
    simulated_scores_df=None,
    normalised_scores=False,
    bins=25,
    title = None,
    save_path = None
):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Extract arrays
    real_scores = scores_df["Score"].dropna().values
    sim_scores  = simulated_scores_df["Score"].dropna().values

    # Optional normalisation
    if normalised_scores:
        real_scores = real_scores / real_scores.max()
        sim_scores  = sim_scores / sim_scores.max()

    # Plot histograms overlapped
    sns.histplot(
        real_scores,
        bins=bins,
        kde=False,
        color="royalblue",
        alpha=0.35,
        label="Observed Scores",
        ax=ax,
        stat="density"
    )
    sns.kdeplot(
        real_scores,
        color="royalblue",
        linewidth=2,
        ax=ax
    )

    # Histogram + KDE for simulated
    sns.histplot(
        sim_scores,
        bins=bins,
        kde=False,
        color="darkorange",
        alpha=0.35,
        label="Simulated Scores",
        ax=ax,
        stat="density"
    )
    sns.kdeplot(
        sim_scores,
        color="darkorange",
        linewidth=2,
        ax=ax
    )
    # Labels and aesthetics
    if title is None:
        title = f"{assessment_name}: Score Distributions"
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.legend()

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)  # ensure it's a Path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig

def raw_sim_tuned_comparison_plot(assessment_name,
                          scores_df = None,
                          simulated_scores_df = None,
                          tuned_sim_scores_df = None,
                          normalised_scores = False,
                          bins=25
                          ):
    
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(1, figure=fig,ncols=3) 
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

  
    plot_score_distribution(
            scores_df,
            ax=ax1,
            title="Score Distribution: Observed", 
            bins=bins,score_col="Score",
            noramlise_x=normalised_scores
        )
    plot_score_distribution(
            simulated_scores_df,
            ax=ax2,
            title="Score Distribution: Simulation", 
            bins=bins,score_col="Score",
            noramlise_x=normalised_scores
        )
    plot_score_distribution(
            tuned_sim_scores_df,
            ax=ax3,
            title="Score Distribution: Tuned Simulation", 
            bins=bins,score_col="Score",
            noramlise_x=normalised_scores
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


#-------------------------------------------------------------------
# 6) PARTICIAPANT ANALYSIS VISUALS 

def visualise_age_curves_across_data(
    data_dict: Dict[str, pd.DataFrame],
    dob_col: str = "DateOfBirth",
    overlay: bool = True,
    bandwidth: float = 2.0,   # smoothing factor
    show: bool = True,
):
    # ---- Extract ages ----
    ages_by_assessment = {}
    for name, df in data_dict.items():
        if dob_col not in df.columns:
            raise ValueError(f"Column '{dob_col}' not found for '{name}'")

        dob = pd.to_datetime(df[dob_col], errors="coerce").dropna()
        today = pd.Timestamp("today").normalize()

        ages = dob.apply(
            lambda d: today.year - d.year
            - int((today.month, today.day) < (d.month, d.day))
        )

        if len(ages) > 0:
            ages_by_assessment[name] = ages

    if not ages_by_assessment:
        raise ValueError("No valid ages found in any dataset.")

    names = list(ages_by_assessment.keys())

    # Shared x-range across all assessments
    all_ages = pd.concat(ages_by_assessment.values())
    x_vals = np.linspace(all_ages.min(), all_ages.max(), 300)

    # Kernel density estimator (manual simple version)
    def kde(ages, x, bw):
        ages_arr = ages.values.reshape(-1, 1)
        return np.mean(
            np.exp(-0.5 * ((x.reshape(-1, 1) - ages_arr.T) / bw) ** 2),
            axis=1,
        )

    # ---- Overlay mode ----
    if overlay:
        fig, ax = plt.subplots(figsize=(10, 6))

        for name in names:
            ages = ages_by_assessment[name]
            y = kde(ages, x_vals, bandwidth)
            y = y / y.max()  # normalise heights for visual comparison

            ax.plot(x_vals, y, label=name, linewidth=2)

        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Relative density")
        ax.set_title(
            "Kernel Density Estimates of Participant Age Distributions Across Assessments"
        )
        ax.legend()
        fig.tight_layout()

        if show:
            plt.show()

        return fig

    # ---- Separate subplots ----
    n = len(names)
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 3 * n_rows),
        sharex=True,
        sharey=True,
    )
    axes = np.array(axes).reshape(-1)

    for ax, name in zip(axes, names):
        ages = ages_by_assessment[name]
        y = kde(ages, x_vals, bandwidth)
        y = y / y.max()  # normalise heights per assessment

        ax.plot(x_vals, y, linewidth=2)
        ax.set_title(name)
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Relative density")

    # Hide unused axes
    for ax in axes[len(names):]:
        ax.set_visible(False)

    fig.suptitle("Age Distribution by Assessment (Smoothed Curves)", y=1.02)
    fig.tight_layout()

    if show:
        plt.show()

    return fig

# ============================================================
# Generic helper for 100%-stacked categorical plots
# ============================================================

def _visualise_stacked_categorical_across_data(
    data_dict: Dict[str, pd.DataFrame],
    column: str,
    title: str,
    label_name: str,
    top_n: int = 10,
    as_percentage: bool = True,
    show: bool = True,
):
    # -----------------------------
    # Collect non-null categorical data
    # -----------------------------
    cat_series_by_assessment = {}
    all_values = []

    for name, df in data_dict.items():
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataset '{name}'.")

        vals = df[column].dropna().astype(str)
        if len(vals) > 0:
            cat_series_by_assessment[name] = vals
            all_values.extend(vals)

    if not cat_series_by_assessment:
        raise ValueError(
            f"No valid non-NaN values found for '{column}' in any dataset."
        )

    # -----------------------------
    # Determine global top-N categories
    # -----------------------------
    global_counts = pd.Series(all_values).value_counts()
    top_categories = list(global_counts.head(top_n).index)

    # -----------------------------
    # Build table: assessment -> [vals for each top category + 'Other']
    # -----------------------------
    table = {}
    assessments = list(cat_series_by_assessment.keys())
    categories = top_categories + ["Other"]

    for name, vals in cat_series_by_assessment.items():
        counts = vals.value_counts()
        total = len(vals)

        row = []
        other_sum = 0.0

        for c in top_categories:
            v = counts.get(c, 0)
            if as_percentage:
                v = v / total * 100.0
            row.append(v)

        # Everything not in top_categories gets grouped into 'Other'
        for cat, count in counts.items():
            if cat not in top_categories:
                v = float(count)
                if as_percentage:
                    v = count / total * 100.0
                other_sum += v

        row.append(other_sum)
        table[name] = row

    # -----------------------------
    # Plot 100% stacked bars
    # -----------------------------
    fig, ax = plt.subplots(figsize=(12, 7))

    assessments_order = assessments
    bottom = np.zeros(len(assessments_order))
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, cat in enumerate(categories):
        heights = [table[name][i] for name in assessments_order]
        ax.bar(
            assessments_order,
            heights,
            bottom=bottom,
            label=cat,
            color=colours[i % len(colours)],
        )
        bottom += heights

    ax.set_ylabel("% of participants" if as_percentage else "Count")
    ax.set_title(title)
    ax.set_xticklabels(assessments_order, rotation=45, ha="right")
    ax.legend(title=label_name, bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()

    if show:
        plt.show()

    return fig

# ============================================================
# Wrappers for specific demographic fields
# ============================================================
def visualise_gender_stack_across_data(
    data_dict: Dict[str, pd.DataFrame],
    column: str = "Gender",
    top_n: int = 5,
    as_percentage: bool = True,
    show: bool = True,
):
    """
    100%-stacked bar chart of Gender across assessments.
    """
    return _visualise_stacked_categorical_across_data(
        data_dict=data_dict,
        column=column,
        title="Gender Distribution Across Assessments",
        label_name="Gender",
        top_n=top_n,
        as_percentage=as_percentage,
        show=show,
    )


def visualise_country_stack_across_data(
    data_dict: Dict[str, pd.DataFrame],
    column: str = "CountryOfResidence",
    top_n: int = 10,
    as_percentage: bool = True,
    show: bool = True,
):
    """
    100%-stacked bar chart of CountryOfResidence across assessments.
    """
    return _visualise_stacked_categorical_across_data(
        data_dict=data_dict,
        column=column,
        title="Country of Residence Distribution Across Assessments",
        label_name="Country",
        top_n=top_n,
        as_percentage=as_percentage,
        show=show,
    )


def visualise_ethnicity_stack_across_data(
    data_dict: Dict[str, pd.DataFrame],
    column: str = "EthnicOrigin",
    top_n: int = 10,
    as_percentage: bool = True,
    show: bool = True,
):
    """
    100%-stacked bar chart of EthnicOrigin across assessments.
    """
    return _visualise_stacked_categorical_across_data(
        data_dict=data_dict,
        column=column,
        title="Ethnic Origin Distribution Across Assessments",
        label_name="Ethnic origin",
        top_n=top_n,
        as_percentage=as_percentage,
        show=show,
    )


def visualise_country_of_origin_stack_across_data(
    data_dict: Dict[str, pd.DataFrame],
    column: str = "CountryOfOrigin",
    top_n: int = 10,
    as_percentage: bool = True,
    show: bool = True,
):
    """
    100%-stacked bar chart of CountryOfOrigin across assessments.
    """
    return _visualise_stacked_categorical_across_data(
        data_dict=data_dict,
        column=column,
        title="Country of Origin Distribution Across Assessments",
        label_name="Country of origin",
        top_n=top_n,
        as_percentage=as_percentage,
        show=show,
    )


def visualise_marital_status_stack_across_data(
    data_dict: Dict[str, pd.DataFrame],
    column: str = "MaritalStatus",
    top_n: int = 10,
    as_percentage: bool = True,
    show: bool = True,
):
    """
    100%-stacked bar chart of MaritalStatus across assessments.
    """
    return _visualise_stacked_categorical_across_data(
        data_dict=data_dict,
        column=column,
        title="Marital Status Distribution Across Assessments",
        label_name="Marital status",
        top_n=top_n,
        as_percentage=as_percentage,
        show=show,
    )


def visualise_education_level_stack_across_data(
    data_dict: Dict[str, pd.DataFrame],
    column: str = "Educationlevel",
    top_n: int = 10,
    as_percentage: bool = True,
    show: bool = True,
):
    """
    100%-stacked bar chart of EducationLevel across assessments.
    """
    return _visualise_stacked_categorical_across_data(
        data_dict=data_dict,
        column=column,
        title="Education Level Distribution Across Assessments",
        label_name="Education level",
        top_n=top_n,
        as_percentage=as_percentage,
        show=show,
    )


def visualise_number_of_children_stack_across_data(
    data_dict: Dict[str, pd.DataFrame],
    column: str = "NumberOfChildren",
    top_n: int = 10,
    as_percentage: bool = True,
    show: bool = True,
):
    """
    100%-stacked bar chart of NumberOfChildren across assessments.
    Treats number of children as a categorical variable (0,1,2,...).
    """
    return _visualise_stacked_categorical_across_data(
        data_dict=data_dict,
        column=column,
        title="Number of Children Distribution Across Assessments",
        label_name="Number of children",
        top_n=top_n,
        as_percentage=as_percentage,
        show=show,
    )


def visualise_postcode_stack_across_data(
    data_dict: Dict[str, pd.DataFrame],
    column: str = "Postcode",
    top_n: int = 10,
    as_percentage: bool = True,
    show: bool = True,
):
    """
    100%-stacked bar chart of Postcode across assessments.
    """
    return _visualise_stacked_categorical_across_data(
        data_dict=data_dict,
        column=column,
        title="Postcode Distribution Across Assessments",
        label_name="Postcode",
        top_n=top_n,
        as_percentage=as_percentage,
        show=show,
    )


def visualise_language_stack_across_data(
    data_dict: Dict[str, pd.DataFrame],
    column: str = "PreferredLanguageCode",
    top_n: int = 10,
    as_percentage: bool = True,
    show: bool = True,
):
    """
    100%-stacked bar chart of PreferredLanguageCode across assessments.
    """
    return _visualise_stacked_categorical_across_data(
        data_dict=data_dict,
        column=column,
        title="Preferred Language Distribution Across Assessments",
        label_name="Language",
        top_n=top_n,
        as_percentage=as_percentage,
        show=show,
    )
