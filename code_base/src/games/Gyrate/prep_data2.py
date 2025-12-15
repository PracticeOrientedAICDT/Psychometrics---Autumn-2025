import re
import numpy as np
import pandas as pd

DATA_PATH = "../Gyrate_UserResults.xlsx"
KEEP_ATTEMPT = "first"   # options: "first", "latest", "best"

# Main function

def main(path=DATA_PATH, keep_attempt=KEEP_ATTEMPT,
         file1="IRTMatrix.csv", file2="IRTMatrixExtended.csv"):
    """
    Receives as input the excel file to process, the criteria to follow when getting rid of 
    multiple attempts by participants, and creates two excel files with suitable IRT matrices.
    """
    df = load_sessions_dev(path, keep_attempt)                 # Loads data and cleans repeated rows and empty columns
    df = remove_repeated_attempts(df, keep_attempt)            # Removes repeated participants
    irtM = make_item_response_matrix_binary(df)                # Creates IRT matrix 1 level = 1 item
    irtM_extended = make_item_response_matrix_3attempts(df)    # Creates IRT matrix 1 level = 3 items

    create_IRT_matrix(irtM, file1)             # <-- save the binary matrix
    create_IRT_matrix_extended(irtM_extended, file2)  # <-- save the 3-attempts matrix


def create_IRT_matrix(df, file):
    df.to_csv(file, index=True)

def create_IRT_matrix_extended(df, file):
    df.to_csv(file, index=True)


def parse_failed_levels(s: str | float | None) -> dict[int, int]:
    """
    Parse the 'FailedLevels' column into {level -> total_failures_at_that_level}.
    Each token looks like 'A:B:C'. We keep max(B) per A (usually equals the count).
    """
    fails: dict[int, int] = {}
    if pd.isna(s):
        return fails
    s = str(s).strip()
    if not s:
        return fails
    for tok in s.split(","):
        parts = tok.strip().split(":")
        if len(parts) < 2:
            continue
        a, b = parts[0].strip(), parts[1].strip()
        if a.isdigit() and b.isdigit():
            A = int(a)
            B = int(b)
            fails[A] = max(fails.get(A, 0), B)
    return fails

def load_sessions_dev(path=DATA_PATH, keep_attempt=KEEP_ATTEMPT):
    """
    Read the raw summary rows and coalesce each group of duplicate-key rows into one row
    by taking the first non-null value inside each group for the data columns.
    """
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]

    # Drop unused columns if present
    df = df.drop(columns=[c for c in ["Percentile", "BonusAwards", "EarlyResponses", "LateResponses"] if c in df.columns],
                 errors="ignore")

    # Types we rely on
    for c in ["Level", "CorrectResponses", "TotalResponses", "ReactionTime"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "CreationDate" in df.columns:
        df["CreationDate"] = pd.to_datetime(df["CreationDate"], errors="coerce", infer_datetime_format=True)

    # Columns that define identical groups (these are equal within each 4-row block)
    key_cols = [
        "AccountId", "AssessmentId", "AssessmentVersionId", "Score",
        "DisplayScore", "Percentage", "Locale", "CreationDate"
    ]
    key_cols = [c for c in key_cols if c in df.columns]  # guard

    # Columns where only one row in the block has the value; pick the first non-null
    value_cols = [
        "TotalResponses", "CorrectResponses", "ReactionTime", "Level", "FailedLevels"
    ]
    value_cols = [c for c in value_cols if c in df.columns]  # guard

    # Helper: first non-null in the group
    def first_valid(s):
        s = s.dropna()
        return s.iloc[0] if not s.empty else np.nan

    # Group by the invariant columns and coalesce value columns
    if key_cols:
        agg_map = {c: first_valid for c in value_cols}
        collapsed = (
            df.groupby(key_cols, as_index=False)
              .agg(agg_map)  # returns one row per unique key
        )
    else:
        # Fallback if keys missing: just take first non-null per entire df (unlikely)
        collapsed = df[value_cols].agg(first_valid).to_frame().T

    return collapsed


def remove_repeated_attempts(df,  keep_attempt=KEEP_ATTEMPT): 

    """
    Read the raw summary rows and keep ONE row per AccountId.
    - 'first'  : earliest CreationDate per AccountId
    - 'latest' : latest CreationDate per AccountId
    - 'best'   : highest Level, tie-broken by latest time
    """

    if keep_attempt == "first":
        df = (df.sort_values(["AccountId", "CreationDate"], ascending=[True, True])
                .drop_duplicates(subset=["AccountId"], keep="first"))
    elif keep_attempt == "latest":
        df = (df.sort_values(["AccountId", "CreationDate"], ascending=[True, False])
                .drop_duplicates(subset=["AccountId"], keep="first"))
    elif keep_attempt == "best":
        df = (df.sort_values(["AccountId", "Level", "CreationDate"],
                             ascending=[True, False, False])
                .drop_duplicates(subset=["AccountId"], keep="first"))
    else:
        raise ValueError("KEEP_ATTEMPT must be 'first', 'latest', or 'best'.")

    return df


def _pass_attempts_from_failed_dict(fails: dict[int, int], n_levels: int = 21) -> list[int]:
    """
    Internal helper.
    Returns a list pass_attempt[1..n], where for each level L:
      - 1,2,3 = passed on that attempt (after that many failures)
      - 0     = failed all 3 attempts at that level
      - -1    = not reached (due to earlier two consecutive 3-fails); treated as fail
    """
    # Normalize and clamp failure counts for levels 1..n
    level_fails = [0] * (n_levels + 1)  # index by level
    for L, cnt in fails.items():
        if 1 <= L <= n_levels:
            c = int(cnt)
            if c < 0: c = 0
            if c > 3: c = 3
            level_fails[L] = c

    # Find first occurrence of two consecutive 3-fails (termination point)
    terminate_at = None  # the second level of the two consecutive fails
    for L in range(1, n_levels):
        if level_fails[L] == 3 and level_fails[L + 1] == 3:
            terminate_at = L + 1
            break

    # Build pass attempts
    pass_attempt = [None] * (n_levels + 1)
    for L in range(1, n_levels + 1):
        if terminate_at is not None and L > terminate_at:
            pass_attempt[L] = -1  # not reached; treated as fail
        else:
            f = level_fails[L]
            if f >= 3:
                pass_attempt[L] = 0   # failed all
            else:
                pass_attempt[L] = f + 1  # passed on attempt 1/2/3
    return pass_attempt[1:]  # 1..n

def make_item_response_matrix_binary(
    df: pd.DataFrame,
    id_col: str = "AccountId",
    failed_col: str = "FailedLevels",
    n_levels: int = 21
) -> pd.DataFrame:
    """
    One column per level (21 columns). Entry is:
      - 1 if participant passed the level (any attempt),
      - 0 if participant failed all 3 attempts,
      - 0 if the level was not reached due to two consecutive prior fails.
    Returns a DataFrame indexed by AccountId with columns L1..L{n}.
    """
    rows = []
    ids = []

    for _, row in df[[id_col, failed_col]].iterrows():
        pid = row[id_col]
        fails = parse_failed_levels(row[failed_col])
        attempts = _pass_attempts_from_failed_dict(fails, n_levels=n_levels)
        # Map attempts to binary score
        # 1/2/3 -> 1 (passed), 0 or -1 -> 0 (failed/not reached)
        binary = [1 if a in (1, 2, 3) else 0 for a in attempts]
        rows.append(binary)
        ids.append(pid)

    cols = [f"L{L}" for L in range(1, n_levels + 1)]
    M = pd.DataFrame(rows, index=ids, columns=cols).astype(int)
    return M


def make_item_response_matrix_3attempts(
    df: pd.DataFrame,
    id_col: str = "AccountId",
    failed_col: str = "FailedLevels",
    n_levels: int = 21
) -> pd.DataFrame:
    """
    Three columns per level (total 3*n_levels). For each level:
      - Passed on 1st attempt: [1, 1, 1]
      - Passed on 2nd attempt: [0, 1, 1]
      - Passed on 3rd attempt: [0, 0, 1]
      - Failed all / Not reached: [0, 0, 0]
    Returns a DataFrame indexed by AccountId with columns:
      L1_A1, L1_A2, L1_A3, ..., L{n}_A1, L{n}_A2, L{n}_A3
    """
    rows = []
    ids = []

    for _, row in df[[id_col, failed_col]].iterrows():
        pid = row[id_col]
        fails = parse_failed_levels(row[failed_col])
        attempts = _pass_attempts_from_failed_dict(fails, n_levels=n_levels)

        triple = []
        for a in attempts:
            if a == 1:
                triple.extend([1, 1, 1])
            elif a == 2:
                triple.extend([0, 1, 1])
            elif a == 3:
                triple.extend([0, 0, 1])
            else:  # a == 0 (failed all) or a == -1 (not reached)
                triple.extend([0, 0, 0])
        rows.append(triple)
        ids.append(pid)

    cols = [f"L{L}_A{A}" for L in range(1, n_levels + 1) for A in (1, 2, 3)]
    M = pd.DataFrame(rows, index=ids, columns=cols).astype(int)
    return M

if __name__ == "__main__":
    main()