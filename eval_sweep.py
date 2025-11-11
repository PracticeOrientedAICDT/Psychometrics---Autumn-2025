#!/usr/bin/env python3
# eval_sweep.py
"""
Sweep true-theta values and estimate bias/accuracy using CAT.py runners.

Examples:
  python eval_sweep.py --items data/QuickCalc/item_params.csv --mode fixed \
      --thetas '-3:3:0.5' --trials 50 --item-repeats 2 --outdir data/QuickCalc/eval

  python eval_sweep.py --items data/QuickCalc/item_params.csv --mode cat \
      --thetas '-2:2:0.25' --trials 100 --item-repeats 1 --max-items 25 --se-target 0.3
"""

from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import your existing runners (adjust module name/path if needed)
from CAT import run_cat, run_fixed, CATConfig


def parse_theta_spec(spec: str) -> np.ndarray:
    """Parse 'lo:hi:step' or 't1,t2,...' into a numpy array."""
    if ":" in spec:
        lo, hi, step = spec.split(":")
        lo, hi, step = float(lo), float(hi), float(step)
        n = int(np.floor((hi - lo) / step + 0.5)) + 1
        arr = lo + step * np.arange(n)
        if abs(arr[-1] - hi) > 1e-9:
            arr = np.append(arr, hi)
        return arr
    else:
        return np.array([float(x) for x in spec.split(",")])


def sweep(
    items_csv: str,
    mode: str,
    thetas: np.ndarray,
    trials: int,
    item_repeats: int = 1,
    top_k: int = 1,
    max_items: int = 20,
    se_target: float = 0.2,
    grid_lo: float = -4.0,
    grid_hi: float = 4.0,
    grid_pts: int = 61,
    prior_mu: float = 0.0,
    prior_sd: float = 1.0,
    base_seed: int | None = 1234,
) -> pd.DataFrame:
    """Run multiple trials per theta; return long DataFrame with per-trial results."""
    df_items = pd.read_csv(items_csv)

    rows = []
    rng = np.random.default_rng(base_seed)
    seed_pool = rng.integers(0, 2**31 - 1, size=len(thetas) * trials)

    cfg = CATConfig(
        mode=mode,
        se_target=se_target,
        max_items=max_items,
        top_k_randomesque=top_k,
        grid_lo=grid_lo,
        grid_hi=grid_hi,
        grid_pts=grid_pts,
        prior_mu=prior_mu,
        prior_sd=prior_sd,
        item_repeats=item_repeats,
    )

    idx = 0
    for t in thetas:
        for tr in range(trials):
            seed = int(seed_pool[idx]); idx += 1

            if mode == "cat":
                # NOTE: pass DataFrame positionally; CAT.py signature is (df, cfg, true_theta, ...)
                records, posters = run_cat(
                    df_items, cfg, float(t), False, seed, None
                )
            else:
                records, posters = run_fixed(
                    df_items, cfg, float(t), False, seed, None
                )

            est = float(posters[-1].mean)
            se = float(posters[-1].se)
            err = est - float(t)

            rows.append(
                dict(
                    theta_true=float(t),
                    trial=int(tr),
                    seed=seed,
                    est=est,
                    error=err,
                    abs_error=abs(err),
                    rmse_contrib=err**2,
                    se=se,
                )
            )

    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Group by theta_true and compute bias, MAE, RMSE, and SE summaries."""
    g = df.groupby("theta_true")
    out = pd.DataFrame({
        "bias": g["error"].mean(),
        "bias_sd": g["error"].std(ddof=1),
        "mae": g["abs_error"].mean(),
        "rmse": np.sqrt(g["rmse_contrib"].mean()),
        "se_mean": g["se"].mean(),
        "se_median": g["se"].median(),
        "n": g.size(),
    }).reset_index()
    return out


def plot_curves(agg: pd.DataFrame, outdir: str, prefix: str):
    os.makedirs(outdir, exist_ok=True)

    # Bias
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(agg["theta_true"], agg["bias"], marker="o")
    ax.axhline(0, linestyle="--", alpha=0.5)
    ax.set_xlabel("True θ")
    ax.set_ylabel("Mean signed error (bias) = E[θ̂ − θ]")
    ax.set_title("Bias vs True θ")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(outdir, f"{prefix}_bias.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"Saved {p}")

    # MAE & RMSE
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(agg["theta_true"], agg["mae"], marker="o", label="MAE")
    ax.plot(agg["theta_true"], agg["rmse"], marker="s", label="RMSE")
    ax.set_xlabel("True θ")
    ax.set_ylabel("Error")
    ax.set_title("MAE and RMSE vs True θ")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = os.path.join(outdir, f"{prefix}_mae_rmse.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"Saved {p}")

    # Reported SE
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(agg["theta_true"], agg["se_mean"], marker="o", label="Mean SE")
    ax.plot(agg["theta_true"], agg["se_median"], marker="s", label="Median SE")
    ax.set_xlabel("True θ")
    ax.set_ylabel("Posterior SE")
    ax.set_title("Reported SE vs True θ")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = os.path.join(outdir, f"{prefix}_se.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"Saved {p}")


def scatter_est_vs_true(df: pd.DataFrame, outdir: str, prefix: str):
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["theta_true"], df["est"], alpha=0.3, s=12)
    lo = float(min(df["theta_true"].min(), df["est"].min()))
    hi = float(max(df["theta_true"].max(), df["est"].max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("True θ")
    ax.set_ylabel("Estimated θ̂")
    ax.set_title("θ̂ vs True θ (all trials)")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(outdir, f"{prefix}_scatter_est_vs_true.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"Saved {p}")


def main():
    ap = argparse.ArgumentParser(description="Sweep true-theta and plot error metrics")
    ap.add_argument("--items", required=True)
    ap.add_argument("--mode", choices=["cat", "fixed"], default="fixed")
    ap.add_argument("--thetas", default="-3:3:0.5",
                    help="Theta grid either 'lo:hi:step' or 't1,t2,...'")
    ap.add_argument("--trials", type=int, default=50,
                    help="Trials (different seeds) per theta")
    ap.add_argument("--item-repeats", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--max-items", type=int, default=20,
                    help="CAT-only; ignored in fixed")
    ap.add_argument("--se-target", type=float, default=0.2,
                    help="CAT-only; ignored in fixed")
    ap.add_argument("--grid-pts", type=int, default=61)
    ap.add_argument("--grid-lo", type=float, default=-4.0)
    ap.add_argument("--grid-hi", type=float, default=4.0)
    ap.add_argument("--prior-mu", type=float, default=0.0)
    ap.add_argument("--prior-sd", type=float, default=1.0)
    ap.add_argument("--base-seed", type=int, default=1234)
    ap.add_argument("--outdir", default="data/QuickCalc/eval")
    args = ap.parse_args()

    thetas = parse_theta_spec(args.thetas)

    df = sweep(
        items_csv=args.items,
        mode=args.mode,
        thetas=thetas,
        trials=args.trials,
        item_repeats=args.item_repeats,
        top_k=args.top_k,
        max_items=args.max_items,
        se_target=args.se_target,
        grid_lo=args.grid_lo,
        grid_hi=args.grid_hi,
        grid_pts=args.grid_pts,
        prior_mu=args.prior_mu,
        prior_sd=args.prior_sd,
        base_seed=args.base_seed,
    )

    os.makedirs(args.outdir, exist_ok=True)
    raw_path = os.path.join(args.outdir, f"sweep_{args.mode}_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"Saved raw results → {raw_path}")

    agg = aggregate(df)
    agg_path = os.path.join(args.outdir, f"sweep_{args.mode}_agg.csv")
    agg.to_csv(agg_path, index=False)
    print(f"Saved aggregated metrics → {agg_path}")

    prefix = f"sweep_{args.mode}"
    plot_curves(agg, args.outdir, prefix)
    scatter_est_vs_true(df, args.outdir, prefix)


if __name__ == "__main__":
    main()
