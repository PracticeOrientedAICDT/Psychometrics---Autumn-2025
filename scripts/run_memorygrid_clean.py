# scripts/run_memorygrid_clean.py
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from memorygrid.clean_mg import get_memorygrid_df
from memorygrid.irt_format_mg import create_irt_input, summarize_matrix

RAW_PATH     = "data/memorygrid/MemoryGrid_UserResults.csv"
IRT_LONG_OUT = "data/memorygrid/memorygrid_irt_long.csv"
MIRT_IN_OUT  = "data/memorygrid/memorygrid_mirt.csv"

def main():
    # Load raw data
    df_raw = pd.read_csv(RAW_PATH)
    print(f"[INFO] Loaded raw data with {len(df_raw)} rows")

    # Clean to long IRT format
    irt_long = get_memorygrid_df(df_raw, verbose=True)

    # --- Sanity checks for long-format data ---
    assert set(irt_long.columns) == {"participant_id", "item_id", "response"}
    assert irt_long["response"].isin([0, 1]).all(), "Responses must be 0/1"
    assert irt_long["item_id"].between(1, 15).all(), "Item IDs must be between 1 and 15"
    assert not irt_long.duplicated(subset=["participant_id", "item_id"]).any(), \
        "Duplicate (participant_id, item_id) pairs found"

    # Save long data
    irt_long.to_csv(IRT_LONG_OUT, index=False)
    print(f"[INFO] Saved cleaned long IRT data → {IRT_LONG_OUT}")

    # Convert to wide format (MIRT-ready)
    mirt_in = create_irt_input(irt_long)

    # ---Sanity checks for wide-format data ---
    item_cols = [int(c) for c in mirt_in.columns[1:]]
    assert item_cols == list(range(1, 16)), "Item columns must be 1..15 in order"
    assert mirt_in.iloc[:, 1:].isin([0, 1]).all().all(), "Matrix values must be 0/1 only"

    # Save wide data
    mirt_in.to_csv(MIRT_IN_OUT, index=False)
    print(f"[INFO] Saved MIRT input data → {MIRT_IN_OUT}")

    # Summary for quick overview
    stats = summarize_matrix(mirt_in)
    print("[INFO] Matrix summary:", stats)

if __name__ == "__main__":
    main()
