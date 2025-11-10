import pandas as pd
from typing import Optional
from typing import Tuple
import numpy as np
try:
    from .analyse import (
        filter_df_columns,
        remove_nan_rows,
        filter_by_group,
        drop_column,
        validate,
    )
except ImportError:
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add project root
    from wordmatch.analyse import (
        filter_df_columns,
        remove_nan_rows,
        filter_by_group,
        drop_column,
        validate,
    )
from tqdm import tqdm

"""
ASSUMPTIONS: 

1. Questions that don't have answers i.e harder questions 
that weren't reached - pad with 0's.

2. One answer per question: If question attempt is unsucessful 
and then after retrial is then succesful, take only the successful 
attempt.

3. Multiple entries for an AccountID: Take the entry with the highest 
score.

4. The AssessmentVersion: 115 (newest, as most populated)

"""

def summarise_dataframe(df: pd.DataFrame, group_col: str = None) -> None:
    """
    Print useful summary information about a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to summarise.
    group_col : str, optional
        Name of a column to count unique values for.
        For example, 'AccountId' will print how many unique AccountIds exist.
    """

    print("\n📊 DATAFRAME SUMMARY 📊")
    print("-" * 40)

    # Shape
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Column names
    print("\nColumns:")
    print(df.columns.tolist())

    # Head
    print("\nFirst 5 rows:")
    print(df.head())

    # Basic info
    print("\nData types:")
    print(df.dtypes)

    # Optional unique count
    if group_col:
        if group_col in df.columns:
            unique_count = df[group_col].nunique()
            print(f"\nUnique values in '{group_col}': {unique_count}")
            print("\nTop 10 most frequent values:")
            print(df[group_col].value_counts().head(10))
        else:
            print(f"\nColumn '{group_col}' not found in DataFrame.")

    print("-" * 40)

def keep_max_score_per_account(
        
    df: pd.DataFrame,
    account_col: str = "AccountId",
    score_col: str = "Score",
    print= False,
) -> pd.DataFrame:
    """
    For groups where `Score` differs within an `AccountId`, keep only the rows
    with the highest score for that account. Groups with a consistent score
    are left unchanged.

    - Coerces `Score` to numeric (non-numeric become NaN).
    - Computes per-account max over non-NaN scores.
    - In inconsistent groups (multiple distinct non-NaN scores), drops rows
      whose score is less than the group's max. Ties at the max are kept.
    - In consistent groups, keeps all rows.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    account_col : str, default "AccountId"
        Grouping column.
    score_col : str, default "Score"
        Score column.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    # Validate columns
    for col in (account_col, score_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    # Work on a copy
    out = df.copy()

    # Coerce to numeric for robust comparison
    score_num = pd.to_numeric(out[score_col], errors="coerce")

    # Per-account diagnostics
    # Count distinct *non-NaN* scores
    uniq_counts = (
        score_num.groupby(out[account_col])
        .nunique(dropna=True)
    )
    # Flag rows in accounts where there are multiple distinct scores
    inconsistent_flag = out[account_col].map(uniq_counts) > 1

    # Compute per-account max (ignoring NaNs)
    group_max = score_num.groupby(out[account_col]).transform("max")

    # Build keep mask:
    # - If account is inconsistent -> keep rows where score == group_max
    # - If account is consistent -> keep everything
    keep_mask = (~inconsistent_flag) | (score_num == group_max)

    # Count removals for reporting
    removed = int((~keep_mask).sum())
    affected_accounts = int((uniq_counts > 1).sum())

    # Apply filter
    filtered = out.loc[keep_mask].copy()
    if print:
        print(
            f"Accounts with differing '{score_col}': {affected_accounts}. "
            f"Removed rows: {removed}. Remaining rows: {len(filtered)}."
        )

    return filtered

def keep_best_attempts_per_question(
    df: pd.DataFrame,
    account_col: str = "AccountId",
    question_col: str = "QuestionWordDifficulty",
    score_col: str = "AnswerScoreBinary",
    verbose = False
) -> pd.DataFrame:
    """
    For each (AccountId, QuestionWordDifficulty):
      - If any row has AnswerScoreBinary == 1, keep one successful row (earliest occurrence).
      - Else keep one failed attempt (earliest occurrence).

    Implemented via stable sort + groupby.head(1) for broad pandas compatibility.
    """
    for col in (account_col, question_col, score_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    # Preserve original order to break ties deterministically
    df = df.copy()
    df["_order"] = np.arange(len(df))

    # Sort so that successes (1) come before failures (0); stable sort keeps earliest among equals
    df_sorted = df.sort_values(
        by=[account_col, question_col, score_col, "_order"],
        ascending=[True, True, False, True],  # score desc -> 1 before 0
        kind="mergesort",                      # stable
    )

    # Take the first row per (account, question)
    out = (
        df_sorted
        .groupby([account_col, question_col], as_index=False, sort=False)
        .head(1)
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )
    if verbose:
        print(f"Reduced from {len(df)} → {len(out)} rows ({len(df) - len(out)} duplicates dropped).")
    return out

# =========================
# CLEAN
# =========================

def wordmatch(df, verbose=False, keep_assementid_versions=False):
    steps = [
        "Load CSV",
        "Filter columns",
        "Remove NaN rows",
        "Keep max score per account",
        "Keep best attempts per question",
        "Filter by assessment version",
        "Validate dataframe",
        "Drop AssessmentVersionId column",
        "Summarise (if verbose)"
    ]
    pbar = tqdm(total=len(steps), desc="🧩 Processing WordMatch", ncols=100)

    # 1
    
    pbar.update(1)

    # 2
    keep_cols = [
        "AccountId", "AssessmentId", "AssessmentVersionId",
        "QuestionWordDifficulty", "AnswerScoreBinary", "Score",
    ]
    filtered_df = filter_df_columns(df, keep_cols)
    pbar.update(1)

    # 3
    filtered_df = remove_nan_rows(filtered_df)
    pbar.update(1)

    # 4
    filtered_df = keep_max_score_per_account(filtered_df, account_col="AccountId", score_col="Score")
    pbar.update(1)

    # 5
    filtered_df = keep_best_attempts_per_question(filtered_df)
    pbar.update(1)

    # 6
    if not keep_assementid_versions:
        filtered_df = filter_by_group(filtered_df, "AssessmentVersionId", 115)
    pbar.update(1)

    # 7 (quiet when verbose=False)
    ok, summary = validate(filtered_df, verbose=verbose)
    if verbose and not ok:
        print(summary)
    pbar.update(1)

    # 8
    filtered_df = drop_column(filtered_df, "AssessmentVersionId")
    pbar.update(1)

    # 9
    if verbose:
        summarise_dataframe(filtered_df, group_col="AccountId")
    pbar.update(1)

    pbar.close()
    print("✅ WordMatch processing complete.\n")
    return filtered_df

def get_wordmatch_df(df,keep_assementid_versions=False,verbose=False):
    df = wordmatch(df,verbose=verbose,
                              keep_assementid_versions=keep_assementid_versions)

    return df


if __name__ == "__main__":
    csv_path = "data/WordMatch/Binary_WordMatch.csv"
    df = pd.read_csv(csv_path)
    df = get_wordmatch_df(df,verbose=False)