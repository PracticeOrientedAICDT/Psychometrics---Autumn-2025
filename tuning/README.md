# Tuning Module

This folder provides two complementary optimisation methods for refining **IRT item parameters**  
(`a`, `b`, optionally `c`) so that simulated test performance matches a desired distribution.

The tuning methods operate **after** you have:
1. Fit an IRT model using R/mirt → producing `abilities.csv` & `item_params.csv`
2. Loaded those into Python
3. Optionally computed real response distributions to match (for EA)

The two available tuners are:

- `cd.py` — **Coordinate Descent Tuner**  
  Adjusts item difficulties (`b`) and optionally discriminations (`a`) using a structured, deterministic search.

- `ea.py` — **Evolutionary Algorithm Tuner**  
  Searches over subsets of items and jittered parameters to match a real score distribution.

---

## Folder Contents

| File | Description |
|------|-------------|
| `cd.py` | Coordinate Descent tuner. Adjusts item parameters using b-shifts + fine-grained per-item perturbations. |
| `ea.py` | Evolutionary Algorithm tuner. Mutates, crosses over, adds, drops items, and jitters parameters. |

---

# 1. Coordinate Descent Tuner (`cd.py`)

The coordinate-descent tuner applies a structured optimisation in two phases:
1. A global search over a coarse difficulty shift.
2. Local per-item updates to `b` (and optionally `a`).

It supports monotonicity constraints, regularisation, and full normality objectives.

### Example Usage

```python
from tuning.cd import tune_item_params
import pandas as pd

abilities_df = pd.read_csv("data/<assessment_name>/modelling/abilities.csv")
items_df     = pd.read_csv("data/<assessment_name>/modelling/item_params.csv")

result = tune_item_params(
    **sim_config,
    normal="FULL",     # use the full KS + mean + std + skew + kurt objective
    max_score=100,
    target_std=10,
    verbose=True
)

tuned_items = result["items"]
```
### Arguments

**Core Inputs**
- **abilities_df**: DataFrame containing participant IDs and θ values.
- **item_params_df**: DataFrame containing initial item parameters (`a`, `b`, optional `c` which is preserved).
- **num_lives**: Number of lives available per item during simulation.
- **num_sub_levels**: Number of sub-levels per item (or a fixed value if not randomised).
- **item_scoring_df**: Optional per-item scoring lookup.
- **seed**: Random seed.

**Normality Objective**
- **target_mean**: Desired mean of simulated total scores.  
  If not provided, defaults to half of `max_score`.
- **target_std**: Desired standard deviation of scores.
- **normal**: If set to `"FULL"`, uses a full shape objective (KS + mean + std + skew + kurtosis).
- **lam_mean**: Legacy argument; typically unused.
- **lam_reg_b**: L2 penalty on per-item difficulty adjustments.
- **lam_reg_a**: L2 penalty on per-item discrimination adjustments.

**Global Coarse Search**
- **coarse_b_grid**: Range of global `b`-shift values to test.
- **a_global_scale**: Optional multiplicative scale to apply globally to all `a`.

**Local Coordinate Descent**
- **tune_a**: Whether to optimise discriminations (`a`) as well as difficulties (`b`).
- **eps_b**: Step size when adjusting an individual `b`.
- **eps_a**: Step size when adjusting an individual `a`.
- **b_cap**: Maximum absolute change allowed for any single `b`.
- **a_cap**: Maximum absolute change allowed for any single `a`.
- **passes**: Number of full coordinate-descent passes.
- **enforce_monotonic_b**: If true, forces item difficulties to remain ordered.

**Other**
- **max_score**: Maximum possible simulated score (used to infer default target_mean).
- **verbose**: Whether to print intermediate optimisation progress.

The function returns a dict containing:
- items — tuned parameter DataFrame (item_id, a, b, c if present)
- sim — final simulated scores
- norm — normality score
- mean — final mean
- db, da — per-item parameter updates
- b_shift — optimal global difficulty shift
Note: if the original item params included a c column, it is kept unchanged.


# 2. Evolutionary Algorithm Tuner (`ea.py`)
The EA tuner searches over item parameter sets to make the simulated total scores look like a target Normal(μ, σ) distribution.

The EA tuner performs a population-based search over item banks.  
It can adjust `a` and `b`, preserve `c`, and add/drop items to achieve a better fit.

This method explores a much wider space than coordinate descent.

### Example Usage
```python
from tuning.ea import tune_item_params
import pandas as pd

abilities_df = pd.read_csv("data/<assessment_name>/modelling/abilities.csv")
items_df     = pd.read_csv("data/<assessment_name>/modelling/item_params.csv")

max_score   = 100
target_mean = max_score / 2      # e.g. 50
target_std  = max_score / 4      # e.g. 25

ea_config = {
            **sim_config,
            # --- data sources ---         
            "item_params_df": items_df__to_tune,                          

            # --- EA search controls ---
            "pop_size": 100,                        
            "gens": 10,                             
            "min_items": 12,                        
            "max_items": len(items_df),   

            "mut_rate": 0.5,                       
            "add_drop_rate": 0.35,     
            "sig_a": 0.25,          
            "sig_b": 0.40,         
            "anneal_min": 0.30,     
            "bounds_a": (0.05, 4.0),
            "bounds_b": (-3.0, 3.0),

            # --- target Normal(μ, σ) for simulated totals ---
            "target_mean": target_mean,                    
            "target_std": target_std,                    
        }
#Tune with EA
tuned_item_params_df = ea.tune_item_params(**ea_config)

# -> DataFrame with columns: item_id, a, b, (c if present — preserved)
```

### Arguments

**Sim Inputs**
- **abilities_df**: Participant θ values used for simulation.
- **item_params_df**: Starting item parameters (must include `a` and `b`; may include `c` which is preserved).
- **num_lives**: Number of lives per item for simulation.
- **num_sub_levels**: Number of sub-levels per item.
- **rand_sl**: Whether to randomise the number of sub-levels.
- **rand_sl_range**: Inclusive range for random sub-levels (e.g., (1,4)).
- **item_scoring_df**: Optional scoring table.
- **seed**: Random seed.

**Evolution Settings**
- **pop_size**: Number of candidate item banks per generation.
- **gens**: Number of evolutionary generations.
- **min_items**: Minimum allowed item count during search.
- **max_items**: Maximum allowed item count.
- **mut_rate**: Probability of mutating item parameters.
- **add_drop_rate**: Probability of adding or dropping an item.

**Mutation Controls**
- **sig_a**: Standard deviation of the jitter applied to discrimination (a) during mutation.
Controls how “wide” the EA explores the discrimination dimension.
- **sig_b**: Standard deviation of the jitter applied to difficulty (b) during mutation.
Larger values increase exploration of difficulty space.
- **anneal_min**: Minimum annealing factor; sets how “cool” the search gets near the final generations.
Higher values retain more exploration; lower values focus on fine-tuning.
- **bounds_a**: (min, max) clamp values for parameter a after mutation.
- **bounds_b**: (min, max) clamp values for parameter b after mutation.
These parameters govern the shape and intensity of mutation behaviour, allowing the EA to explore more aggressively or focus on local improvements.

**Normality Objective**
- **target_mean**: Desired mean of the simulated scores.  
  Typically set to `max_score / 2`.
- **target_std**: Desired standard deviation of simulated scores.  
  Typically `max_score / 4`.

**Mutation Behaviour**
- The EA **only modifies `a` and `b`**.
- If the input item parameters include `c`, it is **preserved exactly**.
- New items introduced during the search inherit `c` from the base set (if present).

---

# Summary

- **CD tuner** = deterministic, fine-grained, ordered, best when you want precise control and subtle adjustments.  
- **EA tuner** = broad exploratory search over both parameter values and item-set structure.  
- Both methods aim to create item parameters that yield **simulated scores** matching a chosen Normal distribution.

