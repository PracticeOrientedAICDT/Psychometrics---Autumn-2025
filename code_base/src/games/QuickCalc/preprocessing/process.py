import os
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

#  Compute project directories
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

DATA_DIR = PROJECT_ROOT / "data"

QUICKCALC_CLEANED_RESPONSES = DATA_DIR / "interim" / "QuickCalc" / "user_responses.csv"

from games.QuickCalc.preprocessing import analyse

# ---------------------------------------------------------------------
# Internal helper: build wide IRT matrix for QuickCalc
# ---------------------------------------------------------------------
def _create_quickcalc_irt_matrix(
    df: pd.DataFrame,
    n_items: int,
    account_col: str = "AccountId",
    level_col: str = "Level",
    failed_col: str = "FailedLevels",
    date_col: str = "CreationDate",
    use_first_attempt: bool = True,   # if False, uses latest by CreationDate
) -> pd.DataFrame:
    """
    Build a wide IRT-style matrix for QuickCalc where:
      - rows = participants
      - columns = items 1..n_items (binary 0/1)
    """

    # ensure datetime for ordering
    dd = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(dd[date_col]):
        dd[date_col] = pd.to_datetime(dd[date_col], errors="coerce")

    # pick first/last row per participant by CreationDate
    sort_order = [account_col, date_col]
    dd = dd.sort_values(sort_order, kind="mergesort")
    picker = "first" if use_first_attempt else "last"
    picked = getattr(dd.groupby(account_col, as_index=False), picker)()

    # output skeleton: all NAs
    pids = picked[account_col].astype(str).values
    cols = ["participant_id"] + list(range(1, n_items + 1))
    irt_df = pd.DataFrame({c: pd.Series([pd.NA] * len(pids), dtype="Int64") for c in cols})
    irt_df["participant_id"] = pids  # keep id as str

    # parse FailedLevels like "1:foo,3:bar" or "2,7"
    def _parse_failed_levels(s):
        if pd.isna(s):
            return set()
        out = set()
        for chunk in str(s).split(","):
            head = chunk.strip().split(":")[0].strip()
            if head.isdigit():
                k = int(head)
                if 1 <= k <= n_items:
                    out.add(k)
        return out

    # fill rows
    for i, row in enumerate(picked.itertuples(index=False)):
        # Level (how far they reached)
        lvl_val = getattr(row, level_col, 0)
        try:
            lvl = int(lvl_val)
        except (TypeError, ValueError):
            lvl = 0
        lvl = max(0, min(n_items, lvl))

        # unseen items → 0
        for k in range(lvl + 1, n_items + 1):
            irt_df.iat[i, k] = 0

        # seen & (by default) succeeded → 1
        if lvl > 0:
            for k in range(1, lvl + 1):
                # i row, col index k (since col 0 is participant_id)
                irt_df.iat[i, k] = 1

        # failures override to 0
        fails = _parse_failed_levels(getattr(row, failed_col, None))
        for k in fails:
            irt_df.iat[i, k] = 0

    # ensure dtype Int64 for item columns
    for k in range(1, n_items + 1):
        irt_df[k] = irt_df[k].astype("Int64")

    return irt_df


# ---------------------------------------------------------------------
# PUBLIC API 1: get_cleaned_responses
# ---------------------------------------------------------------------
def get_cleaned_responses(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Take the raw QuickCalc export and return a cleaned dataframe
    with a well-defined Level column (0-based, clipped at 0) and
    only the relevant columns kept.
    """

    # Optional EDA (can comment out if noisy)
    # analyse.count_all_unique_groups(raw_df)

    columns_to_keep = [
        "AccountId",
        "AssessmentId",
        "AssessmentVersionId",
        "Score",
        "Percentile",
        "Level",
        "FailedLevels",
        "CreationDate"
    ]

    df_subset = analyse.filter_df_columns(raw_df, columns=columns_to_keep)
    cleaned_df = analyse.remove_nan_rows(df_subset, ["Level"])

    # Make Level integer and 0-based (and clip at 0)
    cleaned_df["Level"] = cleaned_df["Level"].astype(int) - 1
    cleaned_df["Level"] = cleaned_df["Level"].clip(lower=0)

    return cleaned_df


# ---------------------------------------------------------------------
# PUBLIC API 2: get_irt_format
# ---------------------------------------------------------------------
def get_irt_matrix(
    raw_df: pd.DataFrame = None,
    use_first_attempt: bool = True,
    return_wide: bool = False,
    cleaned_responses: pd.DataFrame = None
) -> pd.DataFrame:

    if cleaned_responses is None:
        cleaned_df = get_cleaned_responses(raw_df)
    else:
        cleaned_df = cleaned_responses

    # Number of items inferred from maximum Level reached
    num_items = int(cleaned_df["Level"].max())

    # Build wide IRT matrix (participant_id + item 1..num_items)
    wide_irt_df = _create_quickcalc_irt_matrix(
        cleaned_df,
        n_items=num_items,
        use_first_attempt=use_first_attempt,
        account_col="AccountId",
        level_col="Level",
        failed_col="FailedLevels",
        date_col="CreationDate",
    )

    if return_wide:
        # Ensure participant_id is a nice string type if you like
        wide_irt_df["participant_id"] = wide_irt_df["participant_id"].astype(str)
        return wide_irt_df

    # Convert wide → long: participant_id, item_id, response
    long_irt_df = (
        wide_irt_df
        .melt(
            id_vars="participant_id",
            var_name="item_id",
            value_name="response",
        )
        .dropna(subset=["response"])
        .reset_index(drop=True)
    )

    # Ensure nice types
    long_irt_df["participant_id"] = long_irt_df["participant_id"].astype(str)
    long_irt_df["item_id"] = long_irt_df["item_id"].astype(int)
    long_irt_df["response"] = long_irt_df["response"].astype(int)

    return long_irt_df



