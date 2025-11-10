import numpy as np
import pandas as pd
from typing import Optional, Tuple
from scipy.stats import kstest, skew, kurtosis

from simulation.simulate import get_simulated_scores

def _with_item_id(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure an 'item_id' column exists; derive from index if needed."""
    if "item_id" in df.columns:
        return df
    out = df.copy()
    out["item_id"] = out.index
    try:
        out["item_id"] = out["item_id"].astype(int)
    except Exception:
        pass
    order = ["item_id"] + [c for c in out.columns if c != "item_id"]
    return out[order]


def tune_item_params(
    abilities_df: pd.DataFrame,
    item_params_df: pd.DataFrame,
    # simulator knobs
    num_lives: int = 5,
    num_sub_levels: int = 3,
    rand_sl: bool = False,
    rand_sl_range: Tuple[int, int] = (0, 4),
    item_scoring_df: Optional[pd.DataFrame] = None,
    seed: int = 42,

    # EA hyperparams
    pop_size: int = 24,
    gens: int = 30,
    min_items: int = 8,
    max_items: Optional[int] = None,
    mut_rate: float = 0.3,
    add_drop_rate: float = 0.25,
    sig_a: float = 0.15,
    sig_b: float = 0.25,
    anneal_min: float = 0.20,

    bounds_a: Tuple[float, float] = (0.10, 3.00),
    bounds_b: Tuple[float, float] = (-3.00, 3.00),

    # target Normal for SIMULATED scores (no human data involved)
    max_score: Optional[float] = None,            # if given, defaults target to mean=max/2, std=max/4
    target_mean: Optional[float] = None,
    target_std: Optional[float] = None,
) -> pd.DataFrame:
    """
    Evolutionary search over subsets and (a,b) jitter to make the *simulated* score
    distribution match a target Normal (independent of real/human scores).
    Returns the best parameter DataFrame with an 'item_id' column (and 'c' preserved if present).
    """
    print("=== Initialising EA (simulation-only target) ===")
    rng = np.random.default_rng(seed)

    base = item_params_df.sort_index().copy()
    if not {"a", "b"}.issubset(base.columns):
        raise ValueError("item_params_df must contain at least columns 'a' and 'b'.")

    has_c = "c" in base.columns
    cols_mut = ["a", "b"]                    # only these get mutated
    cols_all = cols_mut + (["c"] if has_c else [])

    max_items = len(base) if max_items is None else int(max_items)
    # parameter bounds (NEW)
    

    # Resolve target mean/std (simulation-only)
    if target_mean is None or target_std is None:
        if max_score is None:
            raise ValueError("Provide either (target_mean & target_std) or max_score to derive them.")
        target_mean = float(max_score) / 2.0 if target_mean is None else float(target_mean)
        target_std  = float(max_score) / 4.0 if target_std  is None else float(target_std)

    print(f"Base items: {len(base)} available")
    print(f"Population size = {pop_size}, generations = {gens}")
    print(f"Item count range allowed: {min_items}–{max_items}")
    print(f"Target (simulation): mean={target_mean:.3f}, std={target_std:.3f}")
    print(f"mut_rate: {mut_rate}, add_drop_rate={add_drop_rate}")
    print(f"sig_a={sig_a}, sig_b={sig_b}, anneal_min={anneal_min}")
    print(f"bounds_a={bounds_a}, bounds_b={bounds_b}")
    print("=======================================\n")

    # ---------- helpers ----------
    def clamp_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["a"] = np.clip(df["a"].to_numpy(float), bounds_a[0], bounds_a[1])
        df["b"] = np.clip(df["b"].to_numpy(float), bounds_b[0], bounds_b[1])
        return df

    def jitter_df(df: pd.DataFrame, anneal: float = 1.0) -> pd.DataFrame:
        """Jitter only a/b. c is never touched."""
        out = df.copy()
        if "a" in out.columns:
            out["a"] = out["a"] + rng.normal(0.0, sig_a * anneal, size=len(out))
        if "b" in out.columns:
            out["b"] = out["b"] + rng.normal(0.0, sig_b * anneal, size=len(out))
        return clamp_df(out)

    def random_subset_df() -> pd.DataFrame:
        k = int(rng.integers(min_items, max_items + 1))
        chosen = rng.choice(base.index, size=k, replace=False)
        df = base.loc[chosen, cols_all].copy() if has_c else base.loc[chosen, cols_mut].copy()
        return jitter_df(df, anneal=1.0)
    
    def add_or_drop(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # ADD
        if (len(out) < max_items) and (rng.random() < 0.5):
            pool = np.setdiff1d(base.index.to_numpy(), out.index.to_numpy())
            if len(pool):
                new_id = rng.choice(pool)
                row = base.loc[[new_id], cols_all] if has_c else base.loc[[new_id], cols_mut]
                out = pd.concat([out, row])
                out = jitter_df(out, anneal=1.0)
                return out
        # DROP
        if len(out) > min_items:
            drop_id = rng.choice(out.index.to_numpy())
            out = out.drop(index=drop_id)
        return out
    
    def mutate(df: pd.DataFrame, gen: int) -> pd.DataFrame:
        out = df.copy()
        anneal = max(anneal_min, 1.0 - gen / gens)  # cools from 1.0 → anneal_min
        if rng.random() < add_drop_rate:
            out = add_or_drop(out)
        if rng.random() < mut_rate and len(out):
            m = max(1, len(out) // 4)
            idx = rng.choice(out.index.to_numpy(), size=m, replace=False)
            # jitter selected rows
            out.loc[idx, ["a", "b"]] = jitter_df(out.loc[idx, ["a", "b"]], anneal=anneal)[["a", "b"]]
        return clamp_df(out)
    
    def simulate_and_scores(par_df: pd.DataFrame) -> np.ndarray:
        df2 = _with_item_id(par_df)
        sim = get_simulated_scores(
            abilities_df=abilities_df,
            item_params_df=df2,
            num_lives=num_lives,
            num_sub_levels=num_sub_levels,
            item_scoring_df=item_scoring_df,
            seed=seed,
            rand_sl=rand_sl,
            rand_sl_range=rand_sl_range
        )
        return sim["Score"].dropna().to_numpy(float)

    def fitness_of(par_df: pd.DataFrame) -> Tuple[float, float, float]:
        """
        Returns (fitness, mean, std) for the simulated scores.
        We compute mean/std once and reuse for logging to avoid extra sim work.
        """
        scores = simulate_and_scores(par_df)
        if scores.size == 0 or not np.isfinite(scores).all():
            return float("-inf"), float("nan"), float("nan")

        mu = float(np.mean(scores))
        sigma = float(np.std(scores, ddof=0))

        # KS against same-mean/sd normal (shape-only)
        ks_stat, _ = kstest(scores, 'norm', args=(mu, sigma))
        score_ks = 1.0 - ks_stat

        # scaled penalties to pull toward (target_mean, target_std)
        mean_pen = abs(mu - target_mean) / (target_std + 1e-9)
        std_pen  = abs(sigma - target_std) / (target_std + 1e-9)
        skew_pen = min(abs(skew(scores)), 2.0)
        kurt_pen = min(abs(kurtosis(scores)), 5.0)

        fit = (1.0 * score_ks
               - 0.5 * mean_pen
               - 0.5 * std_pen
               - 0.3 * skew_pen
               - 0.2 * kurt_pen)

        return float(fit), mu, sigma

    # ----- init -----
    print("Initialising population...")
    pop = [random_subset_df() for _ in range(pop_size)]
    fitness_triplets = [fitness_of(df) for df in pop]
    fitness = np.array([t[0] for t in fitness_triplets], dtype=float)

    print("\n=== Starting Evolution ===\n")
    for gen in range(gens):
        elite_idx = int(np.argmax(fitness))
        elite_df  = pop[elite_idx]
        elite_fit, elite_mu, elite_sigma = fitness_triplets[elite_idx]

        print(f"Generation {gen+1}/{gens}")
        print(f"  Best fitness: {elite_fit:.5f}")
        print(f"  Mean: {elite_mu:.2f} (target {target_mean:.2f})  |  Std: {elite_sigma:.2f} (target {target_std:.2f})")
        print(f"  Best item count: {len(elite_df)}")

        # tournament selection
        def tour():
            i, j = rng.integers(0, pop_size, 2)
            return pop[i] if fitness[i] > fitness[j] else pop[j]

        # random immigrants every 5 gens
        if (gen + 1) % 5 == 0:
            k = max(1, pop_size // 8)
            worst = np.argsort(fitness)[:k]
            for w in worst:
                pop[w] = random_subset_df()

        # next generation
        new_pop = [elite_df]  # elitism
        while len(new_pop) < pop_size:
            p1, p2 = tour(), tour()
            # crossover (blend shared, carry c if present)
            i1, i2 = set(p1.index), set(p2.index)
            shared = sorted(i1 & i2)
            only1  = sorted(i1 - i2)
            only2  = sorted(i2 - i1)

            child = pd.DataFrame(columns=cols_all if has_c else cols_mut, dtype=float)

            if shared:
                a1 = p1.loc[shared, cols_mut]
                a2 = p2.loc[shared, cols_mut]
                w  = rng.random(size=(len(shared), 1))
                blended = a1.to_numpy() * w + a2.to_numpy() * (1 - w)
                blended_df = pd.DataFrame(blended, index=shared, columns=cols_mut)
                if has_c:
                    blended_df["c"] = base.loc[shared, "c"].to_numpy()
                child = pd.concat([child, blended_df])

            if only1:
                take1 = rng.choice(only1, size=max(0, len(only1)//2), replace=False)
                part1 = p1.loc[take1, cols_all] if has_c else p1.loc[take1, cols_mut]
                child = pd.concat([child, part1])
            if only2:
                take2 = rng.choice(only2, size=max(0, len(only2)//2), replace=False)
                part2 = p2.loc[take2, cols_all] if has_c else p2.loc[take2, cols_mut]
                child = pd.concat([child, part2])

            # repair size
            if len(child) < min_items:
                need = min_items - len(child)
                pool = np.setdiff1d(base.index.to_numpy(), child.index.to_numpy())
                if len(pool) >= need:
                    fill = rng.choice(pool, size=need, replace=False)
                    filler = base.loc[fill, cols_all] if has_c else base.loc[fill, cols_mut]
                    child = pd.concat([child, filler])
            if len(child) > max_items:
                drop = rng.choice(child.index.to_numpy(), size=len(child) - max_items, replace=False)
                child = child.drop(index=drop)

            child = mutate(child, gen)
            new_pop.append(child)

        pop = new_pop
        fitness_triplets = [fitness_of(df) for df in pop]  # (fit, mu, sigma) once per member
        fitness = np.array([t[0] for t in fitness_triplets], dtype=float)
        fitness[~np.isfinite(fitness)] = float("-inf")
        print("")

    # ----- result -----
    best_idx = int(np.argmax(fitness))
    best_df  = pop[best_idx]
    best_df2 = _with_item_id(best_df)

    print("=== Evolution Complete ===")
    print(f"Best fitness: {float(fitness[best_idx]):.6f}")
    print(f"Best number of items: {len(best_df)}")
    print("=======================================\n")

    # Ensure columns order; keep 'c' if it exists
    keep_cols = ["item_id", "a", "b"] + (["c"] if has_c else [])
    best_df2 = best_df2[keep_cols]

    return best_df2


