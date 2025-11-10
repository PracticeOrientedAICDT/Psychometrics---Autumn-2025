import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from scipy.stats import kstest, skew, kurtosis
from simulation.simulate import get_simulated_scores

def _with_item_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure an integer 'item_id' column exists.
    - If present and castable -> cast to int.
    - Else create 1..n (stable order).
    """
    out = df.copy()
    if "item_id" in out.columns:
        try:
            out["item_id"] = out["item_id"].astype(int)
        except Exception:
            out["item_id"] = np.arange(1, len(out) + 1, dtype=int)
    else:
        out["item_id"] = np.arange(1, len(out) + 1, dtype=int)
    cols = ["item_id"] + [c for c in out.columns if c != "item_id"]
    return out[cols]

def tune_item_params(
    abilities_df: pd.DataFrame,
    item_params_df: pd.DataFrame,
    responses_df: pd.DataFrame,
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

    # target shaping
    target_mean: Optional[float] = None,
    target_std: Optional[float] = None,
) -> pd.DataFrame:
    print("=== Initialising EA ===")
    rng = np.random.default_rng(seed)

    base = item_params_df.sort_index().copy()
    if not {"a", "b"}.issubset(base.columns):
        raise ValueError("item_params_df must contain at least columns 'a' and 'b'.")

    max_items = len(base) if max_items is None else int(max_items)
    has_c = "c" in base.columns
    cols_mut = ["a", "b"]                          # only these mutate
    cols_all = cols_mut + (["c"] if has_c else [])

    bounds = {"a": (0.1, 3.0), "b": (-3.0, 3.0)}

    # target distribution
    target = responses_df["Score"].dropna().to_numpy(float)
    if target.size == 0:
        raise ValueError("responses_df['Score'] is empty after dropna().")
    if target_mean is None:
        target_mean = float(target.mean())
    if target_std is None:
        target_std = float(target.std(ddof=0)) or 1.0

    print(f"Base items: {len(base)} available")
    print(f"Population size = {pop_size}, generations = {gens}")
    print(f"Item count range allowed: {min_items}–{max_items}")
    print(f"Target mean={target_mean:.3f}, target std={target_std:.3f}")
    print("=======================================\n")

    # helpers
    def clamp_df(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in cols_mut:
            lo, hi = bounds[c]
            out[c] = np.clip(out[c].to_numpy(float), lo, hi)
        return out

    def jitter_df(df: pd.DataFrame, sig_a=0.15, sig_b=0.25, anneal: float = 1.0) -> pd.DataFrame:
        out = df.copy()
        out["a"] = out["a"] + rng.normal(0, sig_a * anneal, size=len(out))
        out["b"] = out["b"] + rng.normal(0, sig_b * anneal, size=len(out))
        # NOTE: no touch to 'c'
        return clamp_df(out)

    def random_subset_df() -> pd.DataFrame:
        k = int(rng.integers(min_items, max_items + 1))
        chosen = rng.choice(base.index, size=k, replace=False)
        df = base.loc[chosen, cols_all].copy() if has_c else base.loc[chosen, cols_mut].copy()
        return jitter_df(df)

    def add_or_drop(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # ADD
        if (len(out) < max_items) and (rng.random() < 0.5):
            pool = np.setdiff1d(base.index.to_numpy(), out.index.to_numpy())
            if len(pool):
                new_id = rng.choice(pool)
                row = base.loc[[new_id], cols_all] if has_c else base.loc[[new_id], cols_mut]
                out = pd.concat([out, row])
                out = jitter_df(out)
                return out
        # DROP
        if len(out) > min_items:
            drop_id = rng.choice(out.index.to_numpy())
            out = out.drop(index=drop_id)
        return out

    def mutate(df: pd.DataFrame, gen: int) -> pd.DataFrame:
        out = df.copy()
        anneal = max(0.2, 1.0 - gen / gens)  # cool from 1.0 → 0.2
        if rng.random() < add_drop_rate:
            out = add_or_drop(out)
        if rng.random() < mut_rate and len(out):
            m = max(1, len(out) // 4)
            idx = rng.choice(out.index.to_numpy(), size=m, replace=False)
            # BUGFIX: mutate only a/b
            out.loc[idx, cols_mut] = jitter_df(out.loc[idx, cols_mut], anneal=anneal)
        return clamp_df(out)

    def crossover(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        i1, i2 = set(df1.index), set(df2.index)
        shared = sorted(i1 & i2)
        only1  = sorted(i1 - i2)
        only2  = sorted(i2 - i1)

        child = pd.DataFrame(columns=cols_all if has_c else cols_mut, dtype=float)

        # blend shared (a/b only); carry c from base if present
        if shared:
            a1 = df1.loc[shared, cols_mut]
            a2 = df2.loc[shared, cols_mut]
            w  = rng.random(size=(len(shared), 1))
            blended = a1.to_numpy() * w + a2.to_numpy() * (1 - w)
            blended_df = pd.DataFrame(blended, index=shared, columns=cols_mut)
            if has_c:
                blended_df["c"] = base.loc[shared, "c"].to_numpy()
            child = pd.concat([child, blended_df])

        # uniques
        if only1:
            take1 = rng.choice(only1, size=max(0, len(only1)//2), replace=False)
            part1 = df1.loc[take1, cols_all] if has_c else df1.loc[take1, cols_mut]
            child = pd.concat([child, part1])
        if only2:
            take2 = rng.choice(only2, size=max(0, len(only2)//2), replace=False)
            part2 = df2.loc[take2, cols_all] if has_c else df2.loc[take2, cols_mut]
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

        return clamp_df(child)

    def fitness_of(par_df: pd.DataFrame) -> Tuple[float, float, float]:
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
        scores = sim["Score"].dropna().to_numpy(float)
        if scores.size == 0 or not np.isfinite(scores).all():
            return float("-inf"), float("nan"), float("nan")

        mu = float(np.mean(scores))
        sigma = float(np.std(scores, ddof=0))

        ks_stat, _ = kstest(scores, 'norm', args=(mu, sigma))
        score_ks = 1.0 - ks_stat

        mean_pen = abs(mu - target_mean) / (target_std + 1e-9)
        std_pen  = abs(sigma - target_std) / (target_std + 1e-9)
        skew_pen = min(abs(skew(scores)), 2.0)
        kurt_pen = min(abs(kurtosis(scores)), 5.0)

        fitness = (
            1.0 * score_ks
            - 0.5 * mean_pen
            - 0.5 * std_pen
            - 0.3 * skew_pen
            - 0.2 * kurt_pen
        )

        return fitness, mu, sigma

    # init
    print("Initialising population...")
    pop = [random_subset_df() for _ in range(pop_size)]
    results = [fitness_of(df) for df in pop]
    fitness = np.array([r[0] for r in results])
    means   = np.array([r[1] for r in results])
    stds    = np.array([r[2] for r in results])

    fitness[~np.isfinite(fitness)] = float("-inf")

    print("\n=== Starting Evolution ===\n")
    for gen in range(gens):
        elite_idx = int(np.argmax(fitness))

        elite_df  = pop[elite_idx]
        elite_fit = float(fitness[elite_idx])
        elite_mu = means[elite_idx]
        elite_std = stds[elite_idx]

        print(f"Generation {gen+1}/{gens}")
        print(f"  Best fitness: {elite_fit:.5f}")
        print(f"  Best mean:    {elite_mu:.3f} (target {target_mean:.3f})")
        print(f"  Best std:     {elite_std:.3f} (target {target_std:.3f})")
        print(f"  Best items:   {len(elite_df)}")

        # tournament selection
        def tour():
            i, j = rng.integers(0, pop_size, 2)
            return pop[i] if fitness[i] > fitness[j] else pop[j]

        # random immigrants
        if (gen + 1) % 5 == 0:
            k = max(1, pop_size // 8)
            worst = np.argsort(fitness)[:k]
            for w in worst:
                pop[w] = random_subset_df()

        # next generation
        new_pop = [elite_df]  # elitism
        while len(new_pop) < pop_size:
            p1, p2 = tour(), tour()
            child = crossover(p1, p2)
            child = mutate(child, gen)
            new_pop.append(child)

        pop = new_pop
        fitness = np.array([fitness_of(df) for df in pop], dtype=float)
        fitness = np.array([r[0] for r in results])
        means   = np.array([r[1] for r in results])
        stds    = np.array([r[2] for r in results])
        fitness[~np.isfinite(fitness)] = float("-inf")
        print("")

    # result
    best_idx = int(np.argmax(fitness))
    best_df  = pop[best_idx]
    best_df2 = _with_item_id(best_df)

    print("=== Evolution Complete ===")
    print(f"Best fitness: {float(fitness[best_idx]):.6f}")
    print(f"Best number of items: {len(best_df)}")
    print("=======================================\n")

    return best_df2


