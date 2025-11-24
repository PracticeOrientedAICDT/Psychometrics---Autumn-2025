from __future__ import annotations

from typing import Optional, List
import pandas as pd


def get_cleaned_responses(
    df: pd.DataFrame,
    account_col: str = "AccountId",
    date_col: str = "CreationDate",
    version_col: str = "AssessmentVersionId",
    keep_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    
    # Basic presence checks
    required = [account_col, date_col, version_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cleaned = df.copy()

    # 1) Drop rows missing any key fields
    cleaned = cleaned.dropna(subset=[account_col, date_col, version_col])

    # 2) Keep only dominant assessment version
    cleaned = filter_to_dominant_assessment_version(
        cleaned,
        version_col=version_col,
    )

    # 3) Keep earliest attempt per account
    cleaned = filter_to_first_attempt_per_account(
        cleaned,
        account_col=account_col,
        date_col=date_col,
    )

    # 4) Optionally restrict to a subset of columns
    if keep_cols is not None:
        missing_keep = [c for c in keep_cols if c not in cleaned.columns]
        if missing_keep:
            raise ValueError(
                f"Columns requested in keep_cols not present after cleaning: {missing_keep}"
            )
        cleaned = cleaned[keep_cols].copy()

    return cleaned

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

def filter_to_first_attempt_per_account(
    df: pd.DataFrame,
    account_col: str = "AccountId",
    date_col: str = "CreationDate",
) -> pd.DataFrame:

    missing = [c for c in (account_col, date_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    tmp = df.copy()

    # Coerce CreationDate to datetime; drop invalid
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col])

    # Sort so that the earliest date per account appears first
    tmp = tmp.sort_values([account_col, date_col], kind="mergesort")

    # Group by account and take the first row (earliest CreationDate)
    first_attempt_df = tmp.groupby(account_col, as_index=False).first()
    return first_attempt_df