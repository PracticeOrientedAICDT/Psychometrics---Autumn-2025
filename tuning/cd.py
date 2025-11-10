import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from scipy.stats import kstest, skew

from simulation.simulate import get_simulated_scores
from simulation.eval_utils import full_normality_score

def normality_score(values):
    values = np.asarray(values).astype(float)
    mu, sigma = values.mean(), values.std()
    if sigma == 0:
        return 0
    stat, _ = kstest(values, 'norm', args=(mu, sigma))
    return 1 - stat

def tune_item_params(
    abilities_df: pd.DataFrame,
    item_params_df: pd.DataFrame,
    num_lives: int = 5,
    num_sub_levels: int = 3,
    item_scoring_df: Optional[pd.DataFrame] = None,
    seed: int = 42,

    # objective shaping
    target_mean: Optional[float] = None,
    lam_mean: float = 0.0,        # kept for compatibility (not used if full_normality_score includes mean)
    lam_reg_b: float = 0.1,
    lam_reg_a: float = 0.0,

    # global coarse shift
    coarse_b_grid = np.arange(-2.0, 2.01, 0.25),
    a_global_scale: float = 1.0,

    # coordinate descent (local)
    tune_a: bool = False,
    eps_b: float = 0.05,
    eps_a: float = 0.05,
    b_cap: float = 0.30,
    a_cap: float = 0.20,
    passes: int = 2,
    enforce_monotonic_b: bool = True,

    verbose: bool = True,

    max_score = 100,
    target_std= 10,
    normal = "FULL"
):
    base = item_params_df.copy().reset_index(drop=True)
    n = len(base)
    base_b = base["b"].to_numpy(dtype=float)
    base_a = base["a"].to_numpy(dtype=float)
    has_c = "c" in base.columns
    base_c = base["c"].copy() if has_c else None

    # Targets (update as needed for your assessment)
    resolved_target_mean = (max_score / 2.0) if target_mean is None else float(target_mean)

    # Tiny tolerance to avoid accepting noise-level changes
    tol = 1e-6

    # ---------------------------------------------------
    # Helper: construct item parameters
    # ---------------------------------------------------
    def build_items(b_shift=0.0, a_scale=1.0, db=None, da=None):
        out = base.copy()
        b = base_b + b_shift + (db if db is not None else 0.0)
        a = base_a * a_scale * (1.0 + (da if da is not None else 0.0))
        if enforce_monotonic_b:
            b = np.maximum.accumulate(b)
        out["b"] = b
        out["a"] = a
        if has_c:                # <- explicitly preserve c
            out["c"] = base_c
        return out

    # ---------------------------------------------------
    # Helper: evaluate objective
    # ---------------------------------------------------
    def evaluate(items_df, db=None, da=None):
        sim = get_simulated_scores(
            abilities_df=abilities_df,
            item_params_df=items_df,
            num_lives=num_lives,
            num_sub_levels=num_sub_levels,
            item_scoring_df=item_scoring_df,
            seed=seed,
        )
        scores = sim["Score"].dropna().to_numpy(float)

        # Targets for this evaluation
        tm = resolved_target_mean
        ts = target_std if target_std is not None else scores.std(ddof=0)
        
        if normal == "FULL":
            obj = full_normality_score(
            scores,
            target_mean=tm,
            target_std=ts,
            w_ks=1.0, w_mean=0.5, w_std=0.5,
            w_skew=0.3, w_kurt=0.2
        )
        else:
            obj = normality_score(scores)
       
        # regularisation
        if db is not None:
            obj -= lam_reg_b * float(np.sum(db ** 2))
        if da is not None and lam_reg_a > 0:
            obj -= lam_reg_a * float(np.sum(da ** 2))

        # diagnostics
        mu, sigma = scores.mean(), scores.std(ddof=0)
        ks_stat, _ = kstest(scores, 'norm', args=(mu, sigma))
        norm_score = 1.0 - ks_stat
        return float(obj), float(norm_score), float(mu), sim

    
    # ---------- PHASE 1: global b-shift ----------
    if verbose:
        print("\n=== GLOBAL SHIFT SEARCH ===")
    best = {
        "obj": -1e9,
        "b_shift": 0.0,
        "a_scale": a_global_scale,
        "db": np.zeros(n),
        "da": np.zeros(n),
        "items": base,
        "sim": None,
        "norm": None,
        "mean": None,
    }

    for bs in coarse_b_grid:
        cand_items = build_items(b_shift=float(bs), a_scale=a_global_scale)
        obj, norm, mean, sim = evaluate(cand_items, db=np.zeros(n), da=np.zeros(n))
        if verbose:
            print(f"[GLOBAL] b_shift={bs:+.2f} → norm={norm:.4f}, mean={mean:.2f}")
        if obj > best["obj"] + tol:
            best.update({"obj": obj, "b_shift": float(bs), "items": cand_items,
                         "sim": sim, "norm": norm, "mean": mean})

    if verbose:
        print(f"[GLOBAL] ✅ best b_shift = {best['b_shift']:+.3f}, norm={best['norm']:.4f}\n")


    # ---------- PHASE 2: coordinate descent ----------
    db = np.zeros(n)
    da = np.zeros(n)
    if verbose:
        print("=== LOCAL COORDINATE DESCENT ===")

    improved = True
    for pass_idx in range(passes):
        if verbose:
            print(f"\n--- PASS {pass_idx+1}/{passes} ---")

        if not improved:
            if verbose:
                print("No improvements in previous pass → stopping early.")
            break
        improved = False

        # per-item b steps
        for i in range(n):
            for step in (+eps_b, -eps_b):
                trial_db = db.copy()
                trial_db[i] = np.clip(trial_db[i] + step, -b_cap, b_cap)

                cand = build_items(
                    b_shift=best["b_shift"],
                    a_scale=a_global_scale,
                    db=trial_db,
                    da=da if tune_a else None
                )
                obj, norm, mean, sim = evaluate(cand, db=trial_db, da=da if tune_a else None)

                if obj > best["obj"] + tol:
                    best.update({"obj": obj, "items": cand, "sim": sim, "norm": norm, "mean": mean})
                    db = trial_db
                    improved = True
                    if verbose:
                        cur_scores = sim["Score"].dropna().to_numpy(float)
                        cur_std = cur_scores.std(ddof=0) if cur_scores.size else float("nan")
                        tgt_std = float("nan") if target_std is None else float(target_std)
                        print(f"[LOCAL-b] item={i}, step={step:+.3f}, "
                              f"norm={norm:.4f}, mean={mean:.2f}, "
                              f"t_m={resolved_target_mean:.2f}, std={cur_std:.2f}, t_std={tgt_std:.2f}")

        # optional per-item a steps
        if tune_a:
            for i in range(n):
                for step in (+eps_a, -eps_a):
                    trial_da = da.copy()
                    trial_da[i] = np.clip(trial_da[i] + step, -a_cap, a_cap)

                    cand = build_items(
                        b_shift=best["b_shift"],
                        a_scale=a_global_scale,
                        db=db,
                        da=trial_da
                    )
                    obj, norm, mean, sim = evaluate(cand, db=db, da=trial_da)

                    if obj > best["obj"] + tol:
                        best.update({"obj": obj, "items": cand, "sim": sim, "norm": norm, "mean": mean})
                        da = trial_da
                        improved = True
                        if verbose:
                            cur_scores = sim["Score"].dropna().to_numpy(float)
                            cur_std = cur_scores.std(ddof=0) if cur_scores.size else float("nan")
                            tgt_std = float("nan") if target_std is None else float(target_std)
                            print(f"[LOCAL-a] item={i}, step={step:+.3f}, "
                                  f"norm={norm:.4f}, mean={mean:.2f}, "
                                  f"t_m={resolved_target_mean:.2f}, std={cur_std:.2f}, t_std={tgt_std:.2f}")

    # ---------- summary ----------
    best["db"] = db
    best["da"] = da

    if verbose:
        print("\n=== FINAL SUMMARY ===")
        print(f"Final normality: {best['norm']:.4f}")
        print(f"Global b_shift:  {best['b_shift']:+.3f}")
        print(f"Mean score:      {best['mean']:.2f} (target={resolved_target_mean:.2f})")
        final_scores = best["sim"]["Score"].dropna().to_numpy(float) if best["sim"] is not None else np.array([])
        final_std = final_scores.std(ddof=0) if final_scores.size else float('nan')
        tgt_std = float("nan") if target_std is None else float(target_std)
        print(f"Std  score:      {final_std:.2f} (target={tgt_std:.2f})")
        print(f"Max |δb|:        {np.max(np.abs(best['db'])):.3f}")
        if tune_a:
            print(f"Max |δa|:        {np.max(np.abs(best['da'])):.3f}")

    # ensure expected columns order
    cols = ["item_id", "a", "b"] + (["c"] if has_c else [])
    cols = [c for c in cols if c in best["items"].columns]
    best["items"] = best["items"][cols]
    return best

