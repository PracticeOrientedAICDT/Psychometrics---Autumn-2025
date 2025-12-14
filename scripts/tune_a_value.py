import numpy as np
import pandas as pd
from math import sqrt
from typing import Sequence, Optional, Tuple
from pathlib import Path
import sys
from tqdm import tqdm

# Make src importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from init_core.cat import CATConfig, run_cat  # adjust import path if needed

def tune_discrimination_a(
    num_items: int,
    candidate_as: Optional[Sequence[float]] = None,
    theta_grid: Optional[Sequence[float]] = None,
    b_min: float = -3.0,
    b_max: float = 3.0,
    trials_per_theta: int = 50,
    se_target: float = 0.30,
    max_items: int = 20,
    top_k_randomesque: int = 3,
    grid_lo: float = -4.0,
    grid_hi: float = 4.0,
    grid_pts: int = 61,
    prior_mu: float = 0.0,
    prior_sd: float = 1.0,
    base_seed: int = 1234,
) -> Tuple[float, pd.DataFrame]:
    """
    Search for the best single discrimination value 'a' for a synthetic item bank
    whose difficulty parameters 'b' are evenly spaced from -3 to 3.

    For each candidate a:
      - Build an item bank with num_items items:
            b_i spaced in [-3, 3], all items share the same 'a'.
      - Run a CAT simulation over a grid of true thetas.
      - Compute RMSE of estimated theta_hat vs theta_true over all sims.

    Parameters
    ----------
    num_items : int
        Number of items in the synthetic item bank.
    candidate_as : sequence of float, optional
        Values of 'a' to test. If None, defaults to np.linspace(0.5, 3.0, 11).
    theta_grid : sequence of float, optional
        True thetas to evaluate. If None, defaults to np.linspace(-3, 3, 13).
    trials_per_theta : int
        How many CAT runs to simulate per theta per a.
    se_target : float
        CAT stopping rule: stop when SE <= se_target (unless max_items is hit).
    max_items : int
        Maximum items in a CAT session.
    top_k_randomesque : int
        Number of top-information items to randomise over (randomesque).
    grid_lo, grid_hi, grid_pts : float/int
        Theta grid configuration inside CAT (for the posterior).
    prior_mu, prior_sd : float
        Prior mean and SD for theta.
    base_seed : int
        Base random seed; we offset this per (a, theta, trial) for reproducibility.

    Returns
    -------
    best_a : float
        The candidate 'a' that achieved the lowest overall RMSE.
    results_df : pd.DataFrame
        DataFrame with one row per (a, theta_true, trial) containing:
        ['a', 'theta_true', 'theta_hat', 'rmse_overall'] plus summary rows.
        You can group/aggregate further as needed.
    """
    if candidate_as is None:
        candidate_as = np.linspace(0.5, 3.0, 11)  # e.g. 0.5, 0.75, ..., 3.0

    if theta_grid is None:
        theta_grid = np.linspace(-3.0, 3.0, 13)   # -3, -2.5, ..., 2.5, 3

    candidate_as = list(candidate_as)
    theta_grid = list(theta_grid)

    all_rows = []

    # Precompute the b values for the bank shape
    b_values = np.linspace(b_min, b_max, num_items)

    for a_idx, a_val in enumerate(tqdm(candidate_as, desc="Testing a values")):
       
        # Build synthetic item bank: item_id, a, b (add 'c' if your code expects it)
        items_df = pd.DataFrame({
            "item_id": np.arange(1, num_items + 1),
            "a": np.full(num_items, a_val, dtype=float),
            "b": b_values.astype(float),
        })

        # CAT configuration
        cfg = CATConfig(
            mode="cat",
            se_target=se_target,
            max_items=max_items,
            top_k_randomesque=top_k_randomesque,
            grid_lo=grid_lo,
            grid_hi=grid_hi,
            grid_pts=grid_pts,
            prior_mu=prior_mu,
            prior_sd=prior_sd,
            item_repeats=1,
            verbose=False
        )

        for t_idx, theta_true in enumerate(tqdm(theta_grid, leave=False, desc=f"  θ grid for a={a_val}")):
            # Inner loop: trials
            for trial in tqdm(
                range(trials_per_theta),
                leave=False,
                desc="    trials"
            ):
                seed = (
                    base_seed
                    + a_idx * 10_000
                    + t_idx * 100
                    + trial
                )

                records, posters = run_cat(
                    df=items_df,
                    cfg=cfg,
                    true_theta=theta_true,
                    interactive=False,
                    seed=seed,
                    save_csv=None,
                )

                theta_hat = posters[-1].mean
                all_rows.append({
                    "a": a_val,
                    "theta_true": theta_true,
                    "theta_hat": float(theta_hat),
                })

    results_df = pd.DataFrame(all_rows)
    # Compute overall RMSE per 'a'
    rmse_per_a = []
    for a_val in candidate_as:
        sub = results_df[results_df["a"] == a_val]
        errors = sub["theta_hat"] - sub["theta_true"]
        rmse = sqrt(np.mean(errors.values.astype(float) ** 2))
        rmse_per_a.append({"a": a_val, "rmse_overall": rmse})

    rmse_df = pd.DataFrame(rmse_per_a)

    # Merge RMSE back into results_df for convenience
    results_df = results_df.merge(rmse_df, on="a", how="left")

    # Pick best a (lowest RMSE)
    best_row = rmse_df.loc[rmse_df["rmse_overall"].idxmin()]
    best_a = float(best_row["a"])

    return best_a, results_df

def main():
    best_a, df = tune_discrimination_a(
        num_items=27,
        candidate_as=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
        trials_per_theta=50
    )
    print("Best a =", best_a)

    out = ROOT / "data" / "tuning" / "best_a_results.csv"
    out.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(out, index=False)
    print("Saved detailed results to", out)


if __name__ == "__main__":
    main()
