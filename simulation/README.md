# Simulating Data
The simulation module allows you to generate synthetic assessment results based on a given set of IRT item parameters.

You can run the simulation using either:
- **Real participant abilities** (estimated via IRT), or  
- **Synthetic abilities** sampled from a Normal distribution.

## Example Usage

```python
from simulation import simulate

# Configuration dictionary for running a simulation
sim_config = {
    # Ability source
    "abilities_df": abilities_df,          # Used only when use_abilities_df=True
    "use_abilities_df": False,             # If False, synthetic abilities are generated instead
    "n_synthetic_participants": 4500,      # Number of simulated participants
    "ability_mean": 0.0,                   # Mean of the synthetic ability distribution (θ)
    "ability_std": 1.0,                    # Standard deviation of θ

    # Item / assessment settings
    "item_params_df": items_df,            # IRT item parameters (a, b, c)

    # Assessment mechanics
    "num_lives": 5,                         # Total lives before the test ends
    "num_sub_levels": 3,                    # Attempts required per item
    "rand_sl": True,                        # Randomise the number of sub-level attempts
    "rand_sl_range": [1, 4],                # Range for sub-level randomisation (inclusive)
}

# Run the simulation
simulated_df = simulate.get_simulated_scores(**sim_config)
```
