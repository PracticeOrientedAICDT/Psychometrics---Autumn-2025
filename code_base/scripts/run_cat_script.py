# project_root/scripts/run_cat_script.py

from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Make src importable regardless of where you run from
ROOT = Path(__file__).resolve().parents[1]   # -> project_root
SRC_DIR = ROOT / "src"
sys.path.append(str(SRC_DIR))

from init_core.cat import run_cat, CATConfig, load_item_bank


def run_single_cat(
    items_csv: Path,
    true_theta: float = 0.0,
    mode: str = "cat",
    se_target: float = 0.30,
    max_items: int = 20,
    top_k_randomesque: int = 3,
    grid_lo: float = -4.0,
    grid_hi: float = 4.0,
    grid_pts: int = 61,
    prior_mu: float = 0.0,
    prior_sd: float = 1.0,
    item_repeats: int = 1,
    seed: int = 1234,
):
    """
    Run a single CAT (or fixed-length) session on a given item bank.

    This is meant to be called from Python, not as a CLI.
    """
    items_csv = Path(items_csv)
    df_items = load_item_bank(items_csv)

    cfg = CATConfig(
        mode=mode,
        se_target=se_target,
        max_items=max_items,
        top_k_randomesque=top_k_randomesque,
        grid_lo=grid_lo,
        grid_hi=grid_hi,
        grid_pts=grid_pts,
        prior_mu=prior_mu,
        prior_sd=prior_sd,
        item_repeats=item_repeats,
        # if you added verbose to CATConfig:
        # verbose=False,
    )

    records, posters = run_cat(
        df=df_items,
        cfg=cfg,
        true_theta=true_theta,
        interactive=False,
        seed=seed,
        save_csv=None,
    )

    # Turn records into a DataFrame for inspection
    rec_df = pd.DataFrame([r.__dict__ for r in records])
    final_posterior = posters[-1]

    print("=== CAT run finished ===")
    print(f"True θ      : {true_theta}")
    print(f"Estimated θ̂: {final_posterior.mean:.3f}")
    print(f"SE(θ̂)      : {final_posterior.se:.3f}")
    print(f"Items used  : {len(records)}")

    return rec_df, final_posterior


def main():
    """
    Hard-coded example runner so you can just do:
        python project_root/scripts/run_cat_script.py
    """
    # Example: point to some item bank in your project
    items_csv = ROOT / "data" / "tuning" / "QuickCalc" / "tuned_item_params.csv"

    rec_df, final_post = run_single_cat(
        items_csv=items_csv,
        true_theta=0.5,
        mode="cat",
        se_target=0.30,
        max_items=20,
        top_k_randomesque=3,
        seed=42,
    )

    # If you want to inspect:
    # print(rec_df.head())


if __name__ == "__main__":
    main()
