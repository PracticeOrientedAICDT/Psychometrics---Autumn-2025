import pandas as pd
from typing import List, Optional
import numpy as np
from pathlib import Path

RAW_RESPONSES = "/Users/tt25013/Documents/GitHub/Psychometrics---Autumn-2025/data/raw/MemoryGrid.csv"
CLEANED_RESPONSES =  "/Users/tt25013/Documents/GitHub/Psychometrics---Autumn-2025/data/interim/MemoryGrid/cleaned_responses.csv"
IRT_MATRIX = "/Users/tt25013/Documents/GitHub/Psychometrics---Autumn-2025/data/processed/MemoryGrid/IRT_Matrix.csv"

def get_cleaned_responses() -> pd.DataFrame:
 
    # --- START: cleaning logic goes here ---
    df = pd.read_csv(RAW_RESPONSES)
    cleaned = df.copy()
    print("Length:",len(cleaned))

    #cleaned = drop_dumb_columns(cleaned)
    print("Length:",len(cleaned))

    cleaned = collapse_account_rows(cleaned)
    cleaned.to_csv(CLEANED_RESPONSES, index=False)

    return cleaned

def drop_dumb_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove nonsense columns before modelling/IRT formatting.
    """
    cols_to_remove = ["ContentReferenceId", "Value", "ResourceDescription"]
    df = df.drop(columns=cols_to_remove)
    df = df.drop_duplicates()
    return df

def collapse_account_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each AccountId, keep a single row containing:
      - AccountId
      - one value (first non-null) for FailedLevels, Level, Score, AssessmentVersionId
    Values may appear on any row within that AccountId group.
    """

    cols_to_keep = ["FailedLevels", "Level", "Score", "AssessmentVersionId"]

    def first_non_null(s: pd.Series):
        s_non_null = s.dropna()
        return s_non_null.iloc[0] if len(s_non_null) > 0 else pd.NA

    agg_dict = {col: first_non_null for col in cols_to_keep}

    collapsed = (
        df
        .groupby("AccountId", as_index=False)
        .agg(agg_dict)
    )

    return collapsed

def build_irt_matrix_from_collapsed(
    df: pd.DataFrame,
    account_col: str = "AccountId",
    level_col: str = "Level",
    failed_col: str = "FailedLevels",
) -> pd.DataFrame:
    """
    Turn collapsed responses (one row per AccountId) into a wide IRT matrix.

    For each AccountId:
      - Determine max item index as max(Level) - 1 across the whole df.
      - Set all items 1..(Level-1) = 1 (passed),
      - Set all items >= Level = 0 (not reached / not passed),
      - Then override: the item indicated by the FIRST value in FailedLevels
        (e.g. '2' in '2:4:5') is forced to 0.
    """

    # Ensure Level is numeric
    level_vals = pd.to_numeric(df[level_col], errors="coerce")
    max_level = int(level_vals.max())
    max_item = max_level - 1

    if max_item < 1:
        raise ValueError(f"Computed max_item = {max_item}, check Level column.")

    item_cols = [str(i) for i in range(1, max_item + 1)]

    n = len(df)
    mat = np.zeros((n, max_item), dtype=int)

    df_reset = df.reset_index(drop=True)

    for idx, row in df_reset.iterrows():
        lvl = row[level_col]

        # Try to interpret Level as int
        try:
            lvl_int = int(lvl)
        except (TypeError, ValueError):
            lvl_int = None

        # Set all items < Level to 1
        if lvl_int is not None and lvl_int > 1:
            mat[idx, : (lvl_int - 1)] = 1

        # FailedLevels: only FIRST value before ':' is the failed item
        failed_raw = row[failed_col]

        if pd.notna(failed_raw):
            failed_str = str(failed_raw).strip()
            if failed_str:
                # take only the first number before ':'
                first_token = failed_str.split(":")[0].strip()
                try:
                    item_idx = int(first_token)
                except ValueError:
                    item_idx = None

                if item_idx is not None and 1 <= item_idx <= max_item:
                    mat[idx, item_idx - 1] = 0

    wide = pd.DataFrame(mat, columns=item_cols)
    wide.insert(0, account_col, df_reset[account_col].values)

    return wide

def get_irt_matrix():
    path = Path(IRT_MATRIX)
    # ✅ If the file doesn't exist yet, build and save it
    if not path.exists():
        irt = build_irt_matrix_from_collapsed(pd.read_csv(CLEANED_RESPONSES))
        irt = irt.drop(columns=["AccountId"])
        #irt = irt.drop(columns=["1"])
        irt.to_csv(path, index=False)
        print(f"Saved new IRT matrix to {path}")
    else:
        # ✅ If it already exists, just load it
        irt = pd.read_csv(path)

    return irt




if __name__ == "__main__":
    get_irt_matrix()
