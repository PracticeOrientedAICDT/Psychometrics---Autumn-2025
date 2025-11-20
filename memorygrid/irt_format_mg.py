# memorygrid/irt_format_mg.py
import pandas as pd

def _validate_long_df(df, participant_col, item_col, response_col):
    need = {participant_col, item_col, response_col}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    bad = ~df[response_col].isin([0, 1])
    if bad.any():
        raise ValueError(f"{int(bad.sum())} response values are not 0/1.")

def create_irt_input(
    df_long: pd.DataFrame,
    participant_col: str = "participant_id",
    item_col: str = "item_id",
    response_col: str = "response",
    valid_item_range: tuple | None = (1, 15),
    sort_items: bool = True,
    sort_participants: bool = True,
) -> pd.DataFrame:
    """
    Long → wide matrix:
      rows = participants
      cols = items (levels)
      values = 0/1
    """
    _validate_long_df(df_long, participant_col, item_col, response_col)
    df = df_long.copy()

    # enforce level range if requested
    if valid_item_range is not None:
        lo, hi = valid_item_range
        df = df[df[item_col].between(lo, hi)]

    # collapse just in case (should already be unique from cleaner)
    df = df.groupby([participant_col, item_col], as_index=False)[response_col].max()

    # pivot to wide
    wide = df.pivot(index=participant_col, columns=item_col, values=response_col)

    # fill missing with 0, coerce to small int
    wide = wide.fillna(0).astype("int8")

    # ensure columns are sorted numerically
    if sort_items:
        wide = wide.reindex(sorted(wide.columns), axis=1)
    if sort_participants:
        wide = wide.sort_index()

    # return with participant_id as first column
    wide.reset_index(inplace=True)
    return wide

def summarize_matrix(df_wide: pd.DataFrame, participant_col: str = "participant_id") -> dict:
    item_cols = [c for c in df_wide.columns if c != participant_col]
    if not item_cols:
        return {"n_participants": len(df_wide), "n_items": 0, "mean_score": 0.0, "items": []}
    mean_score = float(df_wide[item_cols].mean().mean())
    return {
        "n_participants": df_wide.shape[0],
        "n_items": len(item_cols),
        "mean_score": round(mean_score, 3),
        "items": item_cols[:10] + (["..."] if len(item_cols) > 10 else []),
    }