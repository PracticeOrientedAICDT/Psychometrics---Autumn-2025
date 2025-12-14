from __future__ import annotations

from typing import Optional, List
import pandas as pd
from typing import Hashable
import numpy as np
    
def get_cleaned_responses(
    df: pd.DataFrame,
    attempt_mode: str = "first",          # 'first', 'last', 'best', 'all'
    filter_largest_version: bool = True,
    account_col: str = "AccountId",
    date_col: str = "CreationDate",
    keep_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    
    df = df.copy()

    # --- Example: still support largest version, even if trivial now ---
    if filter_largest_version and "AssessmentVersionId" in df.columns:
        df = filter_to_dominant_assessment_version(df)

    # ensure CreationDate is datetime for ordering
    if "CreationDate" in df.columns:
        df["CreationDate"] = pd.to_datetime(df["CreationDate"], errors="coerce")

    # basic cleaning: drop rows without Score
    if "Score" in df.columns:
        df = df.dropna(subset=["Score"])
        df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
        df = df.dropna(subset=["Score"])

    account_col = "AccountId"

    # --- Attempt selection ---
    if attempt_mode == "first":
        df = filter_for_first_creation_date(df, account_col=account_col, date_col="CreationDate")

    elif attempt_mode == "last":
        df = filter_to_last_attempt_per_account(df, account_col=account_col, date_col="CreationDate")

    elif attempt_mode == "best":
        df = filter_to_best_attempt_per_account(df, account_col=account_col, score_col="Score")

    elif attempt_mode == "all":
        # keep all attempts; do nothing extra
        pass

    else:
        raise ValueError(f"Unknown attempt_mode: {attempt_mode!r}. "
                         f"Expected one of ['first', 'last', 'best', 'all'].")

    return df

def filter_to_dominant_assessment_version(
        
    df: pd.DataFrame,
    version_col: str = "AssessmentVersionId",
) -> pd.DataFrame:
    if version_col not in df.columns:
        raise ValueError(f"Missing required column '{version_col}'")

    # Count frequency of each version (NaNs already dropped upstream if desired)
    version_counts = df[version_col].value_counts(dropna=False)
 
    if version_counts.empty:
        raise ValueError(f"No valid '{version_col}' values to compute a dominant version.")

    dominant_version = version_counts.idxmax()
    filtered = df[df[version_col] == dominant_version].copy()
    return filtered

def filter_for_first_creation_date(
    df: pd.DataFrame,
    account_col: Hashable = "AccountId",
    date_col: Hashable = "CreationDate",
) -> pd.DataFrame:

    if account_col not in df.columns:
        raise ValueError(f"Column '{account_col}' not found in DataFrame.")
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame.")

    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")

    # idxmin skips NaT, so accounts with all NaT will become NaT and will be dropped;
    # that’s usually what we want in a cleaned dataset.
    idx = tmp.groupby(account_col)[date_col].idxmin()
    result = tmp.loc[idx].copy()

    # optional: sort by account for readability
    result = result.sort_values(by=[account_col, date_col])

    return result

def filter_to_last_attempt_per_account(
    df: pd.DataFrame,
    account_col: Hashable = "AccountId",
    date_col: Hashable = "CreationDate",
) -> pd.DataFrame:
   
    if account_col not in df.columns:
        raise ValueError(f"Column '{account_col}' not found in DataFrame.")
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame.")

    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")

    idx = tmp.groupby(account_col)[date_col].idxmax()
    result = tmp.loc[idx].copy()
    result = result.sort_values(by=[account_col, date_col])

    return result

def filter_to_best_attempt_per_account(
    df: pd.DataFrame,
    account_col: Hashable = "AccountId",
    score_col: Hashable = "Score",
    date_col: Hashable = "CreationDate",
) -> pd.DataFrame:
    
    for col in (account_col, score_col, date_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    tmp = df.copy()
    tmp[score_col] = pd.to_numeric(tmp[score_col], errors="coerce")
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")

    # Drop rows without valid score
    tmp = tmp.dropna(subset=[score_col])

    # Sort so that within each account, higher score comes later,
    # and for equal scores, later date comes later.
    tmp = tmp.sort_values(
        by=[account_col, score_col, date_col],
        ascending=[True, True, True],
    )

    # Take the last row per account (max score, and latest date for ties)
    idx = tmp.groupby(account_col)[score_col].idxmax()
    result = tmp.loc[idx].copy()
    result = result.sort_values(by=[account_col, date_col])

    return result

def build_irt_matrix_from_collapsed(
    df: pd.DataFrame,
    account_col: str = "AccountId",
    level_col: str = "Level",
    failed_col: str = "FailedLevels",
    keep_account_col: bool = False
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
    if keep_account_col:
        wide = wide.drop(columns=["AccountId"])

    return wide

