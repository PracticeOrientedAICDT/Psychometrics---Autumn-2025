"""
compare_distributions.py

Utilities to compare two score distributions (e.g. observed vs simulated).

Public API
---------
get_distribution_comparison_summary(raw_scores, simulated_scores)
print_distribution_comparison_summary(summary)

The main entry point `get_distribution_comparison_summary` takes two 1D
arrays / lists / pandas Series of scores and returns a nested dict with:
  - basic descriptive statistics for each distribution
  - Kolmogorov–Smirnov 2-sample test
  - Wasserstein distance
  - difference in means and a simple standardised effect size
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Mapping, Union

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance, skew, kurtosis


ArrayLike = Union[Iterable[float], np.ndarray, pd.Series]


@dataclass
class DistributionStats:
    n: int
    mean: float
    std: float
    min: float
    q25: float
    median: float
    q75: float
    max: float
    skewness: float
    excess_kurtosis: float


@dataclass
class KSResult:
    statistic: float
    p_value: float


@dataclass
class ComparisonMetrics:
    ks: KSResult
    wasserstein: float
    mean_diff: float
    standardised_mean_diff: float  # Cohen-like d using pooled SD


def _to_1d_array(x: ArrayLike, name: str) -> np.ndarray:
    """Convert input to a clean 1D NumPy array of floats with NaNs removed."""
    if isinstance(x, pd.Series):
        arr = x.to_numpy(dtype=float)
    elif isinstance(x, np.ndarray):
        arr = x.astype(float).ravel()
    else:
        arr = np.asarray(list(x), dtype=float).ravel()

    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        raise ValueError(f"{name} contains no valid (non-NaN) values.")
    return arr


def _compute_distribution_stats(scores: np.ndarray) -> DistributionStats:
    """Compute basic descriptive statistics for a 1D array of scores."""
    n = scores.size
    return DistributionStats(
        n=int(n),
        mean=float(scores.mean()),
        std=float(scores.std(ddof=1)),
        min=float(scores.min()),
        q25=float(np.percentile(scores, 25)),
        median=float(np.median(scores)),
        q75=float(np.percentile(scores, 75)),
        max=float(scores.max()),
        skewness=float(skew(scores, bias=False)),
        excess_kurtosis=float(kurtosis(scores, bias=False)),  # Fisher definition
    )


def _compute_ks_test(raw: np.ndarray, sim: np.ndarray) -> KSResult:
    """Kolmogorov–Smirnov 2-sample test."""
    res = ks_2samp(raw, sim, alternative="two-sided", mode="auto")
    return KSResult(statistic=float(res.statistic), p_value=float(res.pvalue))


def _compute_wasserstein(raw: np.ndarray, sim: np.ndarray) -> float:
    """First Wasserstein distance (Earth Mover's Distance) between distributions."""
    return float(wasserstein_distance(raw, sim))


def _compute_mean_diffs(raw: np.ndarray, sim: np.ndarray) -> Dict[str, float]:
    """Compute difference in means and a simple standardised effect size."""
    mean_raw = float(raw.mean())
    mean_sim = float(sim.mean())
    diff = mean_sim - mean_raw

    # Pooled SD (unbiased)
    var_raw = raw.var(ddof=1)
    var_sim = sim.var(ddof=1)
    n_raw = raw.size
    n_sim = sim.size
    pooled_var = ((n_raw - 1) * var_raw + (n_sim - 1) * var_sim) / (n_raw + n_sim - 2)
    pooled_sd = float(np.sqrt(pooled_var)) if pooled_var > 0 else np.nan

    d = diff / pooled_sd if pooled_sd > 0 else np.nan

    return {"mean_diff": diff, "standardised_mean_diff": float(d)}


def get_distribution_comparison_summary(
    raw_scores: ArrayLike,
    simulated_scores: ArrayLike,
) -> Dict[str, Any]:
    """
    Compute a summary of how well a simulated score distribution matches
    an observed (raw) score distribution.

    Parameters
    ----------
    raw_scores : array-like
        Observed / empirical scores (e.g. from game data).
    simulated_scores : array-like
        Scores generated from the IRT simulation.

    Returns
    -------
    summary : dict
        A nested dictionary with the following structure:

        {
          "raw": {
             "n": ...,
             "mean": ...,
             "std": ...,
             ...
          },
          "simulated": {
             "n": ...,
             "mean": ...,
             "std": ...,
             ...
          },
          "comparison": {
             "ks": {
                 "statistic": ...,
                 "p_value": ...
             },
             "wasserstein": ...,
             "mean_diff": ...,
             "standardised_mean_diff": ...
          }
        }
    """
    raw_arr = _to_1d_array(raw_scores, name="raw_scores")
    sim_arr = _to_1d_array(simulated_scores, name="simulated_scores")

    raw_stats = _compute_distribution_stats(raw_arr)
    sim_stats = _compute_distribution_stats(sim_arr)

    ks_res = _compute_ks_test(raw_arr, sim_arr)
    wdist = _compute_wasserstein(raw_arr, sim_arr)
    mean_diff_info = _compute_mean_diffs(raw_arr, sim_arr)

    comparison = ComparisonMetrics(
        ks=ks_res,
        wasserstein=wdist,
        mean_diff=mean_diff_info["mean_diff"],
        standardised_mean_diff=mean_diff_info["standardised_mean_diff"],
    )

    summary: Dict[str, Any] = {
        "raw": asdict(raw_stats),
        "simulated": asdict(sim_stats),
        "comparison": {
            "ks": asdict(comparison.ks),
            "wasserstein": comparison.wasserstein,
            "mean_diff": comparison.mean_diff,
            "standardised_mean_diff": comparison.standardised_mean_diff,
        },
    }
    return summary


def print_distribution_comparison_summary(summary: Mapping[str, Any]) -> None:
    """
    Nicely formatted console printout of the distribution comparison summary.
    """
    raw = summary["raw"]
    sim = summary["simulated"]
    comp = summary["comparison"]

    print("=== Distribution Comparison Summary ===\n")

    print("Observed (raw) scores:")
    print(f"  N     = {raw['n']}")
    print(f"  Mean  = {raw['mean']:.3f}")
    print(f"  Std   = {raw['std']:.3f}")
    print(f"  Min   = {raw['min']:.3f}")
    print(f"  Q1    = {raw['q25']:.3f}")
    print(f"  Med   = {raw['median']:.3f}")
    print(f"  Q3    = {raw['q75']:.3f}")
    print(f"  Max   = {raw['max']:.3f}")
    print(f"  Skew  = {raw['skewness']:.3f}")
    print(f"  Kurt. = {raw['excess_kurtosis']:.3f}\n")

    print("Simulated scores:")
    print(f"  N     = {sim['n']}")
    print(f"  Mean  = {sim['mean']:.3f}")
    print(f"  Std   = {sim['std']:.3f}")
    print(f"  Min   = {sim['min']:.3f}")
    print(f"  Q1    = {sim['q25']:.3f}")
    print(f"  Med   = {sim['median']:.3f}")
    print(f"  Q3    = {sim['q75']:.3f}")
    print(f"  Max   = {sim['max']:.3f}")
    print(f"  Skew  = {sim['skewness']:.3f}")
    print(f"  Kurt. = {sim['excess_kurtosis']:.3f}\n")

    ks = comp["ks"]
    print("Comparison metrics:")
    print(f"  KS statistic           = {ks['statistic']:.4f}")
    print(f"  KS p-value             = {ks['p_value']:.4e}")
    print(f"  Wasserstein distance   = {comp['wasserstein']:.4f}")
    print(f"  Mean difference (sim - raw) = {comp['mean_diff']:.3f}")
    print(f"  Standardised mean diff (d)  = {comp['standardised_mean_diff']:.3f}")
    print()
    

if __name__ == "__main__":
    # Example usage with dummy data
    rng = np.random.default_rng(42)
    raw_example = rng.normal(loc=50, scale=10, size=1000)
    sim_example = rng.normal(loc=52, scale=11, size=1000)

    summary = get_distribution_comparison_summary(raw_example, sim_example)
    print_distribution_comparison_summary(summary)
