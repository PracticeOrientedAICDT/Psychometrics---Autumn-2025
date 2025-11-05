#!/usr/bin/env python3
# scripts/cat_run.py
"""
2PL CAT with EAP updates + posterior overlay plot.

Usage (interactive):
  python scripts/cat_run.py --items data/QuickCalc/item_params.csv --interactive

Usage (simulate):
  python scripts/cat_run.py --items data/QuickCalc/item_params.csv --true-theta 0.5

Options:
  --se-target 0.18       Stop when posterior SE <= target
  --max-items 20         Max number of items
  --top-k 1              Randomesque selection among top-k informative items
  --grid-pts 61          Quadrature points for EAP
  --grid-lo -4 --grid-hi 4
  --prior-mu 0 --prior-sd 1
  --save data/QuickCalc/cat_trace.csv
  --outdir data/QuickCalc/plots
"""

from __future__ import annotations
import argparse
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- Small numeric utilities (no SciPy needed) ----------
def normal_pdf(x: np.ndarray, mu: float = 0.0, sd: float = 1.0) -> np.ndarray:
    z = (x - mu) / sd
    return (1.0 / (sd * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * z * z)

def logistic(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic/sigmoid for array x."""
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out

def p_2pl(theta: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """P(correct | theta) for 2PL items; broadcasting over theta (G,) and items (I,)."""
    return logistic((theta[:, None] - b[None, :]) * a[None, :])

def info_2pl(theta: float, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Item information I_i(theta) = a^2 * P * (1-P) for 2PL."""
    P = logistic((theta - b) * a)
    return (a ** 2) * P * (1.0 - P)


# ---------- EAP posterior container ----------
@dataclass
class Posterior:
    theta_grid: np.ndarray
    density: np.ndarray  # normalized
    mean: float
    se: float


def eap_update_2pl(
    prior_density: np.ndarray,
    theta_grid: np.ndarray,
    used_a: List[float],
    used_b: List[float],
    used_u: List[int],
) -> Posterior:
    """
    Multiply prior by likelihoods of all observed items, renormalize,
    then compute EAP mean and posterior SE.
    """
    post = prior_density.copy()
    if used_a:
        a_arr = np.asarray(used_a, dtype=float)   # (I,)
        b_arr = np.asarray(used_b, dtype=float)   # (I,)
        u_arr = np.asarray(used_u, dtype=int)     # (I,)

        # Likelihood at each grid point (G,I)
        P = p_2pl(theta_grid, a_arr, b_arr)
        # Accumulate in log-space for stability
        log_like = (u_arr * np.log(P + 1e-15) + (1 - u_arr) * np.log(1 - P + 1e-15)).sum(axis=1)  # (G,)
        post = post * np.exp(log_like - np.max(log_like))

    Z = post.sum()
    if not np.isfinite(Z) or Z <= 0:
        # Fallback: return normalized prior
        post = prior_density / prior_density.sum()
    else:
        post = post / Z

    mean = float(np.sum(theta_grid * post))
    var = float(np.sum((theta_grid - mean) ** 2 * post))
    se = math.sqrt(max(var, 0.0))
    return Posterior(theta_grid=theta_grid, density=post, mean=mean, se=se)


# ---------- CAT loop ----------
@dataclass
class CATConfig:
    se_target: float = 0.30
    max_items: int = 20
    top_k_randomesque: int = 1     # 1 = pure max information; >1 = random among top-k
    grid_lo: float = -4.0
    grid_hi: float = 4.0
    grid_pts: int = 61
    prior_mu: float = 0.0
    prior_sd: float = 1.0


@dataclass
class StepRecord:
    step: int
    item_id: str
    a: float
    b: float
    theta_hat: float
    se: float
    response: Optional[int]
    p_correct_given_prev_theta: float
    item_info_at_prev_theta: float


def select_item_max_info(theta_hat: float, a: np.ndarray, b: np.ndarray, used_mask: np.ndarray, top_k: int = 1) -> int:
    info = info_2pl(theta_hat, a, b)
    info[used_mask] = -np.inf
    if top_k <= 1:
        return int(np.nanargmax(info))
    # randomesque among top-k (filter out -inf)
    idxs = np.argsort(info)
    idxs = idxs[info[idxs] > -np.inf]
    if idxs.size == 0:
        return int(np.nanargmax(info))
    idxs = idxs[-min(top_k, idxs.size):]
    return int(np.random.choice(idxs))


def simulate_response(theta_true: float, a: float, b: float, rng: np.random.Generator) -> Tuple[int, float]:
    P = float(logistic((theta_true - b) * a))
    u = int(rng.random() < P)
    return u, P


def run_cat(
    items_df: pd.DataFrame,
    config: CATConfig,
    true_theta: Optional[float] = None,
    interactive: bool = False,
    seed: Optional[int] = 1234,
    save_csv: Optional[str] = None,
) -> Tuple[List[StepRecord], List[Posterior]]:
    # normalize columns
    df = items_df.copy()
    if "item_id" not in df.columns:
        if "QuestionID" in df.columns:
            df = df.rename(columns={"QuestionID": "item_id"})
        else:
            df.insert(0, "item_id", [str(i) for i in range(len(df))])

    for col in ["a", "b"]:
        if col not in df.columns:
            raise ValueError(f"items_df must have column '{col}' for 2PL")
    if "c" in df.columns:
        print("Note: 'c' column present but ignored for 2PL CAT.")

    item_ids = df["item_id"].astype(str).to_numpy()
    a = pd.to_numeric(df["a"]).to_numpy(dtype=float)
    b = pd.to_numeric(df["b"]).to_numpy(dtype=float)

    # grid + prior
    theta_grid = np.linspace(config.grid_lo, config.grid_hi, config.grid_pts)
    prior_density = normal_pdf(theta_grid, mu=config.prior_mu, sd=config.prior_sd)
    prior_density /= prior_density.sum()

    # init
    used_mask = np.zeros(len(df), dtype=bool)
    used_a, used_b, used_u = [], [], []
    records: List[StepRecord] = []
    posters: List[Posterior] = []

    rng = np.random.default_rng(seed)
    # initial posterior = prior
    post = eap_update_2pl(prior_density, theta_grid, used_a, used_b, used_u)
    posters.append(post)

    step = 0
    while True:
        # stopping rule (after at least one item administered)
        if (step >= 1 and post.se <= config.se_target) or (step >= config.max_items):
            print(f"STOP: step={step}, θ̂(EAP)={post.mean:.3f}, SE={post.se:.3f}")
            break

        # select item (max information at current θ̂)
        j = select_item_max_info(post.mean, a, b, used_mask, top_k=config.top_k_randomesque)
        used_mask[j] = True
        item_id = item_ids[j]
        p_corr_prev = float(logistic((post.mean - b[j]) * a[j]))
        info_prev = float(info_2pl(post.mean, a[j:j+1], b[j:j+1])[0])

        # administer
        if true_theta is not None:
            u, Ptrue = simulate_response(true_theta, a[j], b[j], rng)
            print(f"Q{step+1}: item={item_id} a={a[j]:.2f} b={b[j]:.2f} | P(true)={Ptrue:.3f} → resp={u}")
        elif interactive:
            while True:
                ans = input(f"Q{step+1}: item {item_id} (a={a[j]:.2f}, b={b[j]:.2f}) → enter 1=correct / 0=wrong: ").strip()
                if ans in {"0", "1"}:
                    u = int(ans)
                    break
                print("Please enter 0 or 1.")
        else:
            # default simulate around prior mean if neither provided
            u, _ = simulate_response(config.prior_mu, a[j], b[j], rng)

        # update posterior with this item
        used_a.append(a[j]); used_b.append(b[j]); used_u.append(u)
        post = eap_update_2pl(prior_density, theta_grid, used_a, used_b, used_u)
        posters.append(post)

        step += 1
        rec = StepRecord(
            step=step, item_id=item_id, a=float(a[j]), b=float(b[j]),
            theta_hat=post.mean, se=post.se, response=int(u),
            p_correct_given_prev_theta=p_corr_prev, item_info_at_prev_theta=info_prev
        )
        records.append(rec)
        print(f"   → θ̂(EAP)={post.mean:.3f}, SE={post.se:.3f}")

    # save trace if requested
    if save_csv:
        out = pd.DataFrame([r.__dict__ for r in records])
        out.to_csv(save_csv, index=False)
        print(f"Saved trace to {save_csv}")

    return records, posters


# ---------- Plotting ----------
def plot_posteriors_overlay(
    posters: List[Posterior],
    outpath: Optional[str] = None,
    title: str = "CAT Posterior Evolution (EAP)",
):
    """Overlay all posterior densities with a colour gradient (early=light, late=dark)."""
    if not posters:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.viridis
    n = len(posters)

    # normalise densities to integrate to 1 on the grid (should already be, but guard anyway)
    tg = posters[0].theta_grid
    for i, p in enumerate(posters):
        dens = p.density / (np.trapz(p.density, tg) + 1e-15)
        color = cmap(i / max(1, n - 1))  # 0..1
        ax.plot(tg, dens, color=color, alpha=0.9 if i == n-1 else 0.6, linewidth=1.6)
        # Optional: tiny marker for the mean
        ax.plot([p.mean], [np.interp(p.mean, tg, dens)], marker='o', ms=3, color=color)

    ax.set_xlabel("Ability θ")
    ax.set_ylabel("Density (normalized)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Colourbar legend indicating step progression
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=n))
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Step")
    cbar.set_ticks([1, n])
    cbar.set_ticklabels(["Start", "End"])

    fig.tight_layout()
    if outpath:
        import os
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        print(f"Saved posterior overlay plot → {outpath}")
    return fig, ax


def plot_theta_trace(
    records: List[StepRecord],
    outpath: Optional[str] = None,
    title: str = "θ̂(EAP) Trajectory with ±1 SE",
):
    if not records:
        return None
    steps = np.array([r.step for r in records])
    theta = np.array([r.theta_hat for r in records])
    se = np.array([r.se for r in records])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, theta, marker="o")
    ax.fill_between(steps, theta - se, theta + se, alpha=0.2)
    ax.set_xlabel("Step")
    ax.set_ylabel("θ̂ (EAP)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    if outpath:
        import os
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        print(f"Saved θ̂ trace plot → {outpath}")
    return fig, ax


# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="2PL CAT with EAP updates + posterior overlay plot")
    ap.add_argument("--items", default="data/QuickCalc/item_params.csv", help="Path to item bank CSV")
    ap.add_argument("--true-theta", type=float, default=None, help="Simulate responses with this true θ")
    ap.add_argument("--interactive", action="store_true", help="Interactive mode: type 0/1 responses")
    ap.add_argument("--se-target", type=float, default=0.18, help="Stopping SE threshold")
    ap.add_argument("--max-items", type=int, default=20, help="Maximum items administered")
    ap.add_argument("--top-k", type=int, default=1, help="Randomesque: choose randomly among top-k informative items")
    ap.add_argument("--grid-pts", type=int, default=61, help="Quadrature points for EAP")
    ap.add_argument("--grid-lo", type=float, default=-4.0)
    ap.add_argument("--grid-hi", type=float, default=4.0)
    ap.add_argument("--prior-mu", type=float, default=0.0)
    ap.add_argument("--prior-sd", type=float, default=1.0)
    ap.add_argument("--save", default=None, help="Optional CSV to save the CAT trace")
    ap.add_argument("--outdir", default="data/QuickCalc/plots", help="Directory to save plots")
    ap.add_argument("--seed", type=int, default=1234)
    return ap.parse_args()


def main():
    args = parse_args()
    items_df = pd.read_csv(args.items)

    cfg = CATConfig(
        se_target=args.se_target,
        max_items=args.max_items,
        top_k_randomesque=args.top_k,
        grid_lo=args.grid_lo,
        grid_hi=args.grid_hi,
        grid_pts=args.grid_pts,
        prior_mu=args.prior_mu,
        prior_sd=args.prior_sd,
    )

    records, posters = run_cat(
        items_df=items_df,
        config=cfg,
        true_theta=args.true_theta,
        interactive=args.interactive,
        seed=args.seed,
        save_csv=args.save,
    )

    # Make plots
    outdir = args.outdir
    post_path = f"{outdir}/cat_posterior_overlay.png"
    trace_path = f"{outdir}/cat_theta_trace.png"
    plot_posteriors_overlay(posters, outpath=post_path)
    plot_theta_trace(records, outpath=trace_path)

    # Show interactively if desired:
    # plt.show()


if __name__ == "__main__":
    main()
