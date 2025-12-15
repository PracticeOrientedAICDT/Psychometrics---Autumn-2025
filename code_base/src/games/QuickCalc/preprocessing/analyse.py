import io
import inspect
import numpy as np 
from typing import Optional  
from contextlib import redirect_stdout
import pandas as pd  
from contextlib import nullcontext


"""
CHECKS:

1. AccountID's should have one entry.
2. Item responses are binary.
3. No NaN data.
4. All items should have one reponse.
5. All AssesmentVersions should be the same.
6. All items should increase in QuestionDifficulty the same. i.e 1,2,3 not 1,3,6

"""

# =========================
# TOOLS
# =========================
import pandas as pd

def count_all_unique_groups(df: pd.DataFrame) -> None:

    """
    For each column in the DataFrame, print:
      - the number of unique values
      - the top 10 most frequent values and their counts

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to inspect.
    """
    print(f"\n🔍 DataFrame shape: {df.shape}\n")

    for col in df.columns:
        unique_count = df[col].nunique(dropna=True)
        print(f"🧩 Column: '{col}'")
        print(f"→ Unique values: {unique_count}")

        # Compute top 10 most frequent values
        top_vals = df[col].value_counts(dropna=False).head(10)

        # Format and print
        print("→ Top 10 most frequent values:")
        for val, count in top_vals.items():
            val_display = "<NA>" if pd.isna(val) else val
            print(f"   {val_display}: {count}")

        print("-" * 40)

def summarise_df(df: pd.DataFrame, group_col: str = None,verbose=False) -> None:
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
    if verbose:
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

def filter_df_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Return a new DataFrame containing only the specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list
        List of column names to keep.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the requested columns.
    """
    # Validate input
    if not isinstance(columns, list):
        raise TypeError("columns must be provided as a list of column names.")

    # Check for missing columns
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"The following columns are missing from the DataFrame: {missing}")

    # Filter and return
    filtered_df = df[columns].copy()
    return filtered_df

def remove_nan_rows(df: pd.DataFrame,required_cols,print=False) -> pd.DataFrame:
    """
    Remove rows from a DataFrame where 'QuestionWordDifficulty' or
    'AnswerScoreBinary' contain NaN values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with NaN rows in the specified columns removed.
    """
    if required_cols is None:
        required_cols = ["QuestionWordDifficulty", "AnswerScoreBinary"]

    # Check required columns exist
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Remove rows with NaN in either column
    cleaned_df = df.dropna(subset=required_cols).copy()
    if print:
        print(f"Removed {len(df) - len(cleaned_df)} rows with NaN values "
            f"in {required_cols}. Remaining rows: {len(cleaned_df)}")

    return cleaned_df

def count_unique_groups(df: pd.DataFrame, column: str, max_display: int = 20,print=False) -> int:

    """
    Count and display unique values (groups) in a specified column,
    along with how many rows belong to each group.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    column : str
        The column name to analyze.
    max_display : int, optional
        Maximum number of unique values/groups to display (default is 20).

    Returns
    -------
    int
        Number of unique values (groups) in the specified column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    # Drop NaNs and get value counts
    value_counts = df[column].value_counts(dropna=True)
    unique_count = len(value_counts)
    if print:
        print(f"\nColumn '{column}' has {unique_count} unique value(s).")

    # Display value counts
    if unique_count <= max_display:
        if print:
            print("\nGroup counts:")
            print(value_counts)
    else:
        if print:
            print(f"\nShowing top {max_display} most frequent groups:")
            print(value_counts.head(max_display))

    return unique_count

def print_value_mapping(df: pd.DataFrame, key_col: str, value_col: str, ascending: bool = True):
    """
    Print all unique key values (from key_col) and their associated unique values (from value_col).
    - If key_col is numeric, it is sorted ascending.
    - For each key, show all unique associated value_col entries.
    - If a key has multiple distinct value_col entries, they are shown comma-separated.

    Example:
        🧩 Column: 'QuestionReferenceId'
        → Unique numeric values (3), sorted:
        606: 1,2
        607: 1
        608: 2,3
    """
    # --- validation ---
    if key_col not in df.columns:
        raise ValueError(f"Column '{key_col}' not found in DataFrame.")
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in DataFrame.")

    s_key = df[key_col].dropna()
    s_val = df[value_col]

    # determine if key column is numeric
    coerced = pd.to_numeric(s_key, errors="coerce")
    all_numeric = coerced.notna().all()

    # unique sorted keys
    if all_numeric:
        unique_keys = np.sort(df[key_col].dropna().unique().astype(float))
        if not ascending:
            unique_keys = unique_keys[::-1]
    else:
        unique_keys = pd.unique(df[key_col].dropna())

    print(f"\n🧩 Column: '{key_col}'")
    print(f"→ Unique keys ({len(unique_keys)}), sorted:\n")

    # loop through keys and collect unique values
    for key in unique_keys:
        vals = df.loc[df[key_col] == key, value_col].dropna().unique()
        vals_list = [str(v) for v in np.sort(vals)]
        joined = ", ".join(vals_list)
        print(f"{int(key) if isinstance(key, (int, float)) and key.is_integer() else key}: {joined}")

    print("\n✅ Done.\n")

def filter_by_group(df: pd.DataFrame, column: str, group_value,print=False) -> pd.DataFrame:
    """
    Return only the rows from a DataFrame that belong to a specified group
    in a given column.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    column : str
        The column to filter on.
    group_value : any
        The specific group value to keep (e.g., an AccountId number).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only rows where df[column] == group_value.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    filtered_df = df[df[column] == group_value].copy()

    if filtered_df.empty:
        if print:
            print(f"No rows found for {column} = {group_value}")
    else:
        if print:
            print(f"Filtered DataFrame: {len(filtered_df)} rows where {column} = {group_value}")

    return filtered_df

def drop_column(df: pd.DataFrame, column: str,print=False) -> pd.DataFrame:
    """
    Drop a specified column from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    column : str
        The column name to drop.

    Returns
    -------
    pd.DataFrame
        A new DataFrame without the specified column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    df_dropped = df.drop(columns=[column]).copy()
    if print:
        print(f"Dropped column: '{column}'. Remaining columns: {len(df_dropped.columns)}")

    return df_dropped

# =========================
# VALIDATORS
# =========================

def validate_one_entry_per_account(
    df: pd.DataFrame,
    account_col: str = "AccountId",
    score_col: str = "Score",
    count_na_as_value: bool = True,
    verbose= False
) -> bool:
    """
    True if each AccountId has a single (unique) Score value.
    Flags accounts only when the set of scores within an account has size > 1.

    Parameters
    ----------
    account_col : grouping column (AccountId)
    score_col   : the score column to check for consistency
    count_na_as_value : if True, NaN counts as a distinct value; if False, NaN is ignored
    """
    for col in (account_col, score_col):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}'")

    # Number of unique scores per account
    uniq_counts = (
        df.groupby(account_col)[score_col]
          .nunique(dropna=not count_na_as_value)
    )

    # Offenders: accounts with >1 distinct score
    bad_accounts = uniq_counts[uniq_counts > 1]
    if len(bad_accounts) == 0:
        print("✅ Each AccountId has a single consistent Score.")
        return True

    # Show a few examples with their unique scores
    def _unique_vals(g):
        vals = g[score_col].unique()
        return [("NaN" if pd.isna(v) else v) for v in vals]

    examples = (
        df[df[account_col].isin(bad_accounts.index)]
        .groupby(account_col)
        .apply(_unique_vals)
        .head(10)
        .to_dict()
    )
    if verbose:
        print(f"❌ Accounts with differing '{score_col}': {len(bad_accounts)} (examples: {examples})")
    return False

def validate_no_nan(df: pd.DataFrame, cols: Optional[list] = None) -> bool:
    """
    True if no NaN in selected columns (or entire df if cols=None).
    """
    sub = df if cols is None else df[cols]
    n = sub.isna().sum().sum()
    if n > 0:
        print(f"❌ Found {n} NaN values in {list(sub.columns)}")
        return False
    print("✅ No NaN values.")
    return True

def validate_one_response_per_item(
    df: pd.DataFrame,
    account_col: str = "AccountId",
    item_col: str = "QuestionWordDifficulty",
) -> bool:
    """
    True if there is at most one row per (AccountId, item).
    """
    for c in (account_col, item_col):
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}'")
    dup_counts = df.groupby([account_col, item_col]).size()
    bad = dup_counts[dup_counts > 1]
    if len(bad) > 0:
        print(f"❌ Duplicate responses per ({account_col}, {item_col}): {len(bad)} pairs")
        return False
    print(f"✅ At most one response per ({account_col}, {item_col}).")
    return True

def validate_single_assessment_version(df: pd.DataFrame, col: str = "AssessmentVersionId") -> bool:
    """
    True if there is exactly one assessment version present.
    """
    if col not in df.columns:
        raise ValueError(f"Missing column '{col}'")
    n = df[col].nunique(dropna=True)
    if n != 1:
        print(f"❌ Multiple assessment versions found in '{col}': {n}")
        return False
    print("✅ Single assessment version.")
    return True

def validate_difficulty_progression_uniform(
    df: pd.DataFrame,
    difficulty_col: str = "QuestionWordDifficulty",
    item_group_col: Optional[str] = "QuestionReferenceId",
    verbose = False
) -> bool:
    """
    True if the sorted unique difficulties form a consecutive range and (if item_group_col provided)
    each item group shares the same difficulty sequence.

    - If item_group_col provided: check per item and also they all match a reference sequence.
    - If not provided: check the global sequence is consecutive.
    """
    if difficulty_col not in df.columns:
        raise ValueError(f"Missing column '{difficulty_col}'")

    def is_consecutive(seq: list) -> bool:
        return list(range(int(min(seq)), int(max(seq)) + 1)) == list(map(int, seq))

    if item_group_col and item_group_col in df.columns:
        sequences = {}
        for key, g in df.groupby(item_group_col):
            vals = sorted(pd.to_numeric(g[difficulty_col], errors="coerce").dropna().unique().astype(int).tolist())
            if not vals:
                continue
            sequences[key] = vals
            if not is_consecutive(vals):
                if verbose:
                    print(f"❌ Non-consecutive difficulties for {item_group_col}={key}: {vals}")
                return False

        # Compare all sequences to the first non-empty one
        ref = None
        for v in sequences.values():
            if v:
                ref = v
                break
        if ref is None:
            if verbose:
                print("⚠️ No valid difficulty values to compare.")
            return True

        mismatches = {k: v for k, v in sequences.items() if v != ref}
        if mismatches:
            if verbose:
                print(f"❌ Not all items share the same difficulty sequence. Reference {ref}, mismatches: {list(mismatches.items())[:5]}")
            return False
        if verbose:
            print(f"✅ All items share the same consecutive difficulty sequence: {ref}")
        return True

    # Global-only check
    vals = sorted(pd.to_numeric(df[difficulty_col], errors="coerce").dropna().unique().astype(int).tolist())
    if not vals:
        if verbose:
            print("⚠️ No valid difficulty values to check.")
        return True
    if not is_consecutive(vals):
        if verbose:
            print(f"❌ Global difficulty not consecutive: {vals}")
        return False
    if verbose:
        print(f"✅ Global difficulty consecutive: {vals}")
    return True

def validate(df: pd.DataFrame, verbose: bool = True):
    """
    Auto-runs all validate_* functions in this module.
    - If verbose=False, suppresses all print output inside validators.
    Returns: (all_ok: bool, summary: str)
    """
    funcs = []
    for name, fn in inspect.getmembers(__import__(__name__), inspect.isfunction):
        if name.startswith("validate_") and name != "validate":
            funcs.append((name, fn))
    funcs.sort(key=lambda x: x[0])

    lines = []
    all_ok = True

    # capture prints if not verbose
    sink = io.StringIO()
    ctx = redirect_stdout(sink) if not verbose else nullcontext()

    with (ctx):
        for name, fn in funcs:
            try:
                ok = fn(df)
            except Exception as e:
                ok = False
                if verbose:
                    print(f"💥 {name} raised {e}")
            all_ok &= bool(ok)
            lines.append(f"{'✅' if ok else '❌'} {name}")

    # if verbose, the validators already printed details; still return a concise summary
    return all_ok, "\n".join(lines)

# =========================
# DEBUG
# =========================

def export_inconsistent_scores(
    df: pd.DataFrame,
    csv_debug_path: str,
    account_col: str = "AccountId",
    score_col: str = "Score",
    count_na_as_value: bool = True,
    extra_sort_cols: Optional[list] = None,
) -> pd.DataFrame:
    """
    Find groups (by `account_col`) where `score_col` is not constant and
    export all rows from those groups to a CSV for debugging.
    """
    for col in (account_col, score_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    uniq_counts = df.groupby(account_col)[score_col].nunique(dropna=not count_na_as_value)
    bad_accounts = uniq_counts[uniq_counts > 1].index

    if len(bad_accounts) == 0:
        print("✅ All groups have consistent scores. No CSV written.")
        return pd.DataFrame(columns=df.columns)

    inconsistent_df = df[df[account_col].isin(bad_accounts)].copy()

    def _unique_vals(series: pd.Series) -> str:
        vals = series.unique()
        as_str = ["NaN" if pd.isna(v) else str(v) for v in vals]
        return ", ".join(as_str)

    per_group_uniques = inconsistent_df.groupby(account_col)[score_col].apply(_unique_vals).to_dict()
    per_group_counts = inconsistent_df.groupby(account_col)[score_col].nunique(dropna=not count_na_as_value).to_dict()

    inconsistent_df["Score_Uniques_In_Group"] = inconsistent_df[account_col].map(per_group_uniques)
    inconsistent_df["Score_UniqueCount_In_Group"] = inconsistent_df[account_col].map(per_group_counts)

    sort_cols = [account_col, score_col]
    if extra_sort_cols:
        sort_cols.extend([c for c in extra_sort_cols if c in inconsistent_df.columns])

    try:
        inconsistent_df = inconsistent_df.sort_values(sort_cols)
    except Exception:
        inconsistent_df = inconsistent_df.sort_values([account_col])

    inconsistent_df.to_csv(csv_debug_path, index=False)
    print(
        f"⚠️ Found {len(bad_accounts)} group(s) with inconsistent '{score_col}'. "
        f"Exported {len(inconsistent_df)} row(s) to: {csv_debug_path}"
    )
    return inconsistent_df

