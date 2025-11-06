import os
import pandas as pd
from cleaning.QuickCalc import analyse

def create_irt_input(
    df: pd.DataFrame,
    n_items: int = 28,
    account_col: str = "AccountId",
    level_col: str = "Level",
    failed_col: str = "FailedLevels",
    date_col: str = "CreationDate",
    use_first_attempt: bool = True,   # if False, uses latest by CreationDate
) -> pd.DataFrame:

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
        if pd.isna(s): return set()
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
        for k in range(lvl+1, n_items + 1):
            irt_df.iat[i, k] = 0


        # seen & (by default) succeeded → 1
        if lvl > 0:
            # columns 1..lvl get 1
            for k in range(1, lvl + 1):
                irt_df.iat[i, k] = 1  # i row, col index k (since col 0 is participant_id)

        # failures override to 0
        fails = _parse_failed_levels(getattr(row, failed_col, None))
        for k in fails:
            irt_df.iat[i, k] = 0

    # ensure dtype Int64 for item columns
    for k in range(1, n_items + 1):
        irt_df[k] = irt_df[k].astype("Int64")

    return irt_df

def run(df):   
    
    columns_to_keep = ["AccountId","AssessmentId","AssessmentVersionId","Score","Percentile","Level","FailedLevels","CreationDate"
    ]
    df = analyse.filter_df_columns(df,columns=columns_to_keep)
    cleaned_df = analyse.remove_nan_rows(df,["Level"])
    cleaned_df["Level"] = cleaned_df["Level"].astype(int) - 1
    cleaned_df["Level"] = cleaned_df["Level"].clip(lower=0)

    irt_df = create_irt_input(df)
    irt_df = analyse.drop_column(irt_df,"participant_id")

    
    return irt_df,cleaned_df



if __name__ == "__main__":
    #run()
    print()