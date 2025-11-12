#!/usr/bin/env python3
# scripts/CAT.py
"""
2PL CAT with EAP updates + posterior overlay plot.
Supports:
  --mode cat   → adaptive testing (select next item by information)
  --mode fixed → present all items in order, each repeated N times.

Usage examples:
  python CAT.py --mode cat --items data/QuickCalc/item_params.csv --true-theta 0.5
  python CAT.py --mode fixed --items data/QuickCalc/item_params.csv --true-theta 0.5 --item-repeats 2
"""

from __future__ import annotations
import argparse
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- Loading Data ----------
CANON_HEADERS = {"item_id", "a", "b", "c"}  # c ignored in 2PL but allowed

def load_item_bank(path: str) -> pd.DataFrame:
    """
    Robust item bank reader:
    - tolerates single/double quotes around headers/values
    - tolerates accidental index column
    - tolerates 'QuestionID' instead of 'item_id'
    - enforces numeric a/b/c and creates item_id if missing
    """
    # Try normal read first
    try:
        df = pd.read_csv(path)
    except Exception:
        # If a weird quoting scheme, retry with single-quote as quotechar
        df = pd.read_csv(path, quotechar="'", skipinitialspace=True)

    # Strip whitespace and quotes from column names
    df.columns = (df.columns
                  .str.strip()
                  .str.strip("'").str.strip('"')
                  .str.lower())

    # Drop a stray unnamed index column if present
    bad_idx_cols = [c for c in df.columns if c.startswith("unnamed")]
    if bad_idx_cols:
        df = df.drop(columns=bad_idx_cols)

    # Normalise item_id column name
    if "item_id" not in df.columns and "questionid" in df.columns:
        df = df.rename(columns={"questionid": "item_id"})

    # If still no item_id, create one
    if "item_id" not in df.columns:
        df.insert(0, "item_id", [str(i) for i in range(len(df))])

    # Strip quotes/spaces from item_id values
    df["item_id"] = (df["item_id"].astype(str)
                     .str.strip().str.strip("'").str.strip('"'))

    # Ensure required numeric columns exist
    for col in ["a", "b"]:
        if col not in df.columns:
            raise ValueError(f"items file must contain columns 'a' and 'b' (found: {list(df.columns)})")

    # Coerce numeric, fail early if bad data
    for col in ["a", "b", "c"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="raise")

    # Optional: sanity checks
    if (df["a"] <= 0).any():
        raise ValueError("All 'a' parameters must be > 0 for 2PL.")

    # Keep only known columns (order: item_id, a, b, c if present)
    keep = [c for c in ["item_id", "a", "b", "c"] if c in df.columns]
    df = df[keep].copy()

    return df


# ---------- Small numeric utilities ----------
def normal_pdf(x: np.ndarray, mu: float = 0.0, sd: float = 1.0) -> np.ndarray:
    z = (x - mu) / sd
    return (1.0 / (sd * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * z * z)

def logistic(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out

def p_2pl(theta: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return logistic((theta[:, None] - b[None, :]) * a[None, :])

def info_2pl(theta: float, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    P = logistic((theta - b) * a)
    return (a ** 2) * P * (1.0 - P)


# ---------- Posterior structure ----------
@dataclass
class Posterior:
    theta_grid: np.ndarray
    density: np.ndarray
    mean: float
    se: float


def eap_update_2pl(
    prior_density: np.ndarray,
    theta_grid: np.ndarray,
    used_a: List[float],
    used_b: List[float],
    used_u: List[int],
) -> Posterior:
    post = prior_density.copy()
    if used_a:
        a_arr = np.asarray(used_a, dtype=float)
        b_arr = np.asarray(used_b, dtype=float)
        u_arr = np.asarray(used_u, dtype=int)
        P = p_2pl(theta_grid, a_arr, b_arr)
        log_like = (u_arr * np.log(P + 1e-15) +
                    (1 - u_arr) * np.log(1 - P + 1e-15)).sum(axis=1)
        post *= np.exp(log_like - np.max(log_like))

    Z = post.sum()
    if not np.isfinite(Z) or Z <= 0:
        post = prior_density / prior_density.sum()
    else:
        post /= Z

    mean = float(np.sum(theta_grid * post))
    var = float(np.sum((theta_grid - mean) ** 2 * post))
    se = math.sqrt(max(var, 0.0))
    return Posterior(theta_grid, post, mean, se)


# ---------- Config & Records ----------
@dataclass
class CATConfig:
    mode: str = "cat"
    se_target: float = 0.30
    max_items: int = 20
    top_k_randomesque: int = 1
    grid_lo: float = -4.0
    grid_hi: float = 4.0
    grid_pts: int = 61
    prior_mu: float = 0.0
    prior_sd: float = 1.0
    item_repeats: int = 1  # shared across CAT and fixed modes


@dataclass
class StepRecord:
    step: int
    item_id: str
    a: float
    b: float
    theta_hat: float
    se: float
    response: int
    p_correct_given_prev_theta: float
    item_info_at_prev_theta: float


# ---------- Helper functions ----------
def select_item_max_info(theta_hat: float,
                         a: np.ndarray,
                         b: np.ndarray,
                         exhausted_mask: np.ndarray,
                         top_k: int = 1) -> int:
    info = info_2pl(theta_hat, a, b)
    info[exhausted_mask] = -np.inf
    if top_k <= 1:
        return int(np.nanargmax(info))
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


# ---------- CAT loop ----------
def run_cat(
    df: pd.DataFrame,
    cfg: CATConfig,
    true_theta: Optional[float],
    interactive: bool,
    seed: Optional[int],
    save_csv: Optional[str],
) -> Tuple[List[StepRecord], List[Posterior]]:
    item_ids = df["item_id"].astype(str).to_numpy()
    a = pd.to_numeric(df["a"]).to_numpy()
    b = pd.to_numeric(df["b"]).to_numpy()

    theta_grid = np.linspace(cfg.grid_lo, cfg.grid_hi, cfg.grid_pts)
    prior = normal_pdf(theta_grid, mu=cfg.prior_mu, sd=cfg.prior_sd)
    prior /= prior.sum()

    counts = np.zeros(len(df), dtype=int)
    used_a, used_b, used_u = [], [], []
    records, posters = [], []
    rng = np.random.default_rng(seed)

    post = eap_update_2pl(prior, theta_grid, used_a, used_b, used_u)
    posters.append(post)
    step = 0

    while True:
        if (step >= 1 and post.se <= cfg.se_target) or (step >= cfg.max_items):
            print(f"STOP: step={step}, θ̂={post.mean:.3f}, SE={post.se:.3f}")
            break
        exhausted = counts >= cfg.item_repeats
        if np.all(exhausted):
            print("STOP: all items exhausted.")
            break

        j = select_item_max_info(post.mean, a, b, exhausted, top_k=cfg.top_k_randomesque)
        item_id = item_ids[j]
        p_corr_prev = float(logistic((post.mean - b[j]) * a[j]))
        info_prev = float(info_2pl(post.mean, a[j:j+1], b[j:j+1])[0])

        if true_theta is not None:
            u, Ptrue = simulate_response(true_theta, a[j], b[j], rng)
            print(f"Q{step+1}: {item_id} a={a[j]:.2f} b={b[j]:.2f} "
                  f"P(true)={Ptrue:.2f} → resp={u} (rep {counts[j]+1}/{cfg.item_repeats})")
        elif interactive:
            while True:
                ans = input(f"Q{step+1}: {item_id} a={a[j]:.2f} b={b[j]:.2f} "
                            f"[rep {counts[j]+1}/{cfg.item_repeats}] → 1=correct / 0=wrong: ").strip()
                if ans in {"0", "1"}:
                    u = int(ans)
                    break
                print("Please enter 0 or 1.")
        else:
            u, _ = simulate_response(cfg.prior_mu, a[j], b[j], rng)

        used_a.append(a[j]); used_b.append(b[j]); used_u.append(u)
        post = eap_update_2pl(prior, theta_grid, used_a, used_b, used_u)
        posters.append(post)

        step += 1
        records.append(StepRecord(step, item_id, a[j], b[j],
                                  post.mean, post.se, u,
                                  p_corr_prev, info_prev))
        print(f"   → θ̂={post.mean:.3f}, SE={post.se:.3f}")
        counts[j] += 1

    if save_csv:
        pd.DataFrame([r.__dict__ for r in records]).to_csv(save_csv, index=False)
        print(f"Saved trace → {save_csv}")
    return records, posters


# ---------- Fixed loop ----------
def run_fixed(
    df: pd.DataFrame,
    cfg: CATConfig,
    true_theta: Optional[float],
    interactive: bool,
    seed: Optional[int],
    save_csv: Optional[str],
) -> Tuple[List[StepRecord], List[Posterior]]:
    item_ids = df["item_id"].astype(str).to_numpy()
    a = pd.to_numeric(df["a"]).to_numpy()
    b = pd.to_numeric(df["b"]).to_numpy()

    theta_grid = np.linspace(cfg.grid_lo, cfg.grid_hi, cfg.grid_pts)
    prior = normal_pdf(theta_grid, mu=cfg.prior_mu, sd=cfg.prior_sd)
    prior /= prior.sum()

    used_a, used_b, used_u = [], [], []
    records, posters = [], []
    rng = np.random.default_rng(seed)
    post = eap_update_2pl(prior, theta_grid, used_a, used_b, used_u)
    posters.append(post)
    step = 0

    for j in range(len(df)):
        for rep in range(cfg.item_repeats):
            item_id = item_ids[j]
            p_corr_prev = float(logistic((post.mean - b[j]) * a[j]))
            info_prev = float(info_2pl(post.mean, a[j:j+1], b[j:j+1])[0])

            if true_theta is not None:
                u, Ptrue = simulate_response(true_theta, a[j], b[j], rng)
                print(f"Q{step+1}: {item_id} a={a[j]:.2f} b={b[j]:.2f} "
                      f"P(true)={Ptrue:.2f} → resp={u} (rep {rep+1}/{cfg.item_repeats})")
            elif interactive:
                while True:
                    ans = input(f"Q{step+1}: {item_id} a={a[j]:.2f}, b={b[j]:.2f} "
                                f"[rep {rep+1}/{cfg.item_repeats}] → 1=correct / 0=wrong: ").strip()
                    if ans in {"0", "1"}:
                        u = int(ans)
                        break
                    print("Please enter 0 or 1.")
            else:
                u, _ = simulate_response(cfg.prior_mu, a[j], b[j], rng)

            used_a.append(a[j]); used_b.append(b[j]); used_u.append(u)
            post = eap_update_2pl(prior, theta_grid, used_a, used_b, used_u)
            posters.append(post)
            step += 1
            records.append(StepRecord(step, item_id, a[j], b[j],
                                      post.mean, post.se, u,
                                      p_corr_prev, info_prev))
            print(f"   → θ̂={post.mean:.3f}, SE={post.se:.3f}")

    if save_csv:
        pd.DataFrame([r.__dict__ for r in records]).to_csv(save_csv, index=False)
        print(f"Saved trace → {save_csv}")
    return records, posters


# ---------- Plotting ----------
def plot_posteriors_overlay(posters: List[Posterior], outpath: Optional[str], title: str):
    if not posters:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.viridis
    tg = posters[0].theta_grid
    n = len(posters)
    for i, p in enumerate(posters):
        dens = p.density / (np.trapezoid(p.density, tg) + 1e-15)
        color = cmap(i / max(1, n - 1))
        ax.plot(tg, dens, color=color, alpha=0.9 if i == n - 1 else 0.6)
        ax.plot([p.mean], [np.interp(p.mean, tg, dens)], "o", ms=3, color=color)
    ax.set_xlabel("Ability θ")
    ax.set_ylabel("Density")
    ax.set_title(title)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=n))
    plt.colorbar(sm, ax=ax, pad=0.02, label="Step")
    plt.tight_layout()
    if outpath:
        import os
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        print(f"Saved posterior overlay → {outpath}")


def plot_theta_trace(
    posters: List[Posterior],
    outpath: Optional[str] = None,
    title: str = "θ̂(EAP) Trajectory with ±1 SE (incl. prior)",
):
    """
    Plot θ̂ and ±1 SE over steps including step 0 (the prior).
    We read directly from `posters`, which always has posters[0] = prior.
    """
    if not posters:
        return None

    steps = np.arange(len(posters))  # 0 = prior, then 1..N = after each item
    theta = np.array([p.mean for p in posters])
    se = np.array([p.se for p in posters])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, theta, marker="o")
    ax.fill_between(steps, theta - se, theta + se, alpha=0.2)
    ax.set_xlabel("Step (0 = prior)")
    ax.set_ylabel("θ̂ (EAP)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(0, steps.max())  # make it explicit we start at 0

    fig.tight_layout()
    if outpath:
        import os
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        print(f"Saved θ̂ trace plot → {outpath}")
    return fig, ax


# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="2PL CAT or fixed mode simulator")
    ap.add_argument("--mode", choices=["cat", "fixed"], default="cat",
                    help="Adaptive CAT or fixed non-adaptive presentation")
    ap.add_argument("--items", required=True, help="Path to item bank CSV")
    ap.add_argument("--true-theta", type=float, default=None)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--se-target", type=float, default=0.2)
    ap.add_argument("--max-items", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--item-repeats", type=int, default=1,
                    help="Max repeats per item (CAT) or repeats per item (fixed)")
    ap.add_argument("--grid-pts", type=int, default=61)
    ap.add_argument("--grid-lo", type=float, default=-4.0)
    ap.add_argument("--grid-hi", type=float, default=4.0)
    ap.add_argument("--prior-mu", type=float, default=0.0)
    ap.add_argument("--prior-sd", type=float, default=1.0)
    ap.add_argument("--save", default=None)
    ap.add_argument("--outdir", default="data/QuickCalc/plots")
    ap.add_argument("--seed", type=int, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    df = load_item_bank(args.items)
    if "item_id" not in df.columns:
        df.insert(0, "item_id", [str(i) for i in range(len(df))])

    cfg = CATConfig(
        mode=args.mode,
        se_target=args.se_target,
        max_items=args.max_items,
        top_k_randomesque=args.top_k,
        grid_lo=args.grid_lo,
        grid_hi=args.grid_hi,
        grid_pts=args.grid_pts,
        prior_mu=args.prior_mu,
        prior_sd=args.prior_sd,
        item_repeats=args.item_repeats,
    )

    if args.mode == "cat":
        records, posters = run_cat(df, cfg, args.true_theta, args.interactive, args.seed, args.save)
        title_prefix = "CAT"
    else:
        records, posters = run_fixed(df, cfg, args.true_theta, args.interactive, args.seed, args.save)
        title_prefix = "Fixed"

    post_path = f"{args.outdir}/{title_prefix.lower()}_posterior.png"
    trace_path = f"{args.outdir}/{title_prefix.lower()}_theta_trace.png"
    plot_posteriors_overlay(posters, post_path, f"{title_prefix} Posterior Evolution")
    plot_theta_trace(posters, trace_path, f"{title_prefix} θ̂ Trajectory")


if __name__ == "__main__":
    main()
