import pandas as pd
from irt.process import simulate_data_determinstic, simulate_data_stochastic

# Paths
abilities_csv = "data/QuickCalc/abilities.csv"
items_csv     = "data/QuickCalc/item_params.csv"

# Load
abilities_df = pd.read_csv(abilities_csv)      # expects participant_id, theta
items_df     = pd.read_csv(items_csv)          # expects a, b (and optionally c)

# --- Deterministic (expected number correct) ---
det_df = simulate_data_determinstic(
    abilities_df=abilities_df,
    item_latents_df=items_df,
    account_col="participant_id",
    ability_col="theta",
    a_col="a",
    b_col="b",
    c_col="c"      # or None if your items_df has no 'c'
)
det_df.to_csv("data/QuickCalc/scores_deterministic.csv", index=False)

# --- Stochastic (simulate actual 0/1 outcomes and sum) ---
# n_simulations lets you generate multiple independent runs
stoch_df = simulate_data_stochastic(
    abilities_df=abilities_df,
    item_latents_df=items_df,
    account_col="participant_id",
    ability_col="theta",
    a_col="a",
    b_col="b",
    c_col="c",
    n_simulations=3,
    seed=123
)
stoch_df.to_csv("data/QuickCalc/scores_simulated.csv", index=False)

print("Wrote data/QuickCalc/scores_expected.csv and data/QuickCalc/scores_stochastic.csv")
