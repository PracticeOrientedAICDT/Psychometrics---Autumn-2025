import pandas as pd

def parse_failed_levels_max_level(failed_str: str) -> int | None:
    """
    Return the highest level that appears in FailedLevels (or None).
    FailedLevels example: '3:1:2,3:2:1,4:1:0'
    """
    if pd.isna(failed_str) or not failed_str:
        return None
    max_lvl = None
    for chunk in str(failed_str).split(","):
        parts = chunk.split(":")
        if not parts:
            continue
        try:
            lvl = int(parts[0])
        except ValueError:
            continue
        max_lvl = lvl if (max_lvl is None or lvl > max_lvl) else max_lvl
    return max_lvl

def get_memorygrid_df(df_raw: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Build long IRT data: (participant_id, item_id=level, response in {0,1}).

    Rules:
      - 'Level' = highest level PASSED (had a correct trial).
      - You can still ATTEMPT levels > highest_passed (after failing previous levels),
        but those attempts never had a correct trial.
      - Therefore: response = 1 iff level ≤ highest_passed, else 0 for any attempted higher level.
      - Levels are 1..15; attempts are sequential from 1 up to highest_attempted.
    """
    LEVEL_MIN, LEVEL_MAX = 1, 15
    rows: list[tuple[str, int, int]] = []

    for _, row in df_raw.iterrows():
        pid = row.get("AccountId")

        # Highest PASSED level (may be NaN if none)
        highest_passed = row.get("Level", None)
        try:
            h = int(highest_passed) if pd.notna(highest_passed) else None
        except (ValueError, TypeError):
            h = None

        # Highest level that appears in the failures string (attempted but not necessarily passed)
        failed_levels_raw = row.get("FailedLevels", "")
        max_failed_level = parse_failed_levels_max_level(failed_levels_raw)

        # Highest ATTEMPTED is the max of passed and failed levels we can see
        candidates = [x for x in (h, max_failed_level) if x is not None]
        highest_attempted = max(candidates) if candidates else None
        if highest_attempted is None:
            # No evidence of any attempt; skip participant
            continue

        # Clamp to valid design range
        highest_attempted = max(LEVEL_MIN, min(highest_attempted, LEVEL_MAX))
        h_clamped = max(LEVEL_MIN, min(h, LEVEL_MAX)) if h is not None else None

        # Assign response for each ATTEMPTED level:
        # - 1 if level ≤ highest_passed (they must have had a correct on that level)
        # - 0 otherwise (they attempted it but never had a correct there)
        for lvl in range(LEVEL_MIN, highest_attempted + 1):
            if h_clamped is not None and lvl <= h_clamped:
                resp = 1
            else:
                resp = 0
            rows.append((pid, lvl, resp))

    # Build DataFrame, enforce types, collapse dupes (shouldn’t be any but safe)
    df = pd.DataFrame(rows, columns=["participant_id", "item_id", "response"])
    if not df.empty:
        df = df[(df["item_id"] >= LEVEL_MIN) & (df["item_id"] <= LEVEL_MAX)].copy()
        df["participant_id"] = df["participant_id"].astype(str)
        df["item_id"] = df["item_id"].astype(int)
        df["response"] = df["response"].astype(int)
        df = df.groupby(["participant_id", "item_id"], as_index=False)["response"].max()

    if verbose:
        n_p = df["participant_id"].nunique() if not df.empty else 0
        n_l = df["item_id"].nunique() if not df.empty else 0
        print(f"[MemoryGrid] Created IRT DataFrame with {len(df)} rows "
              f"for {n_p} participants and {n_l} levels (range 1–15).")
    return df