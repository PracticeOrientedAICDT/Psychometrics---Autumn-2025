import pandas as pd
import os

REQUIRED_COLUMNS = [
    "item_id",
    "a",
    "b",
    "difficulty",
    "speedup",
    "releaseInterval",
    "levelUpHits"
]


def load_mechanics(csv_path: str) -> pd.DataFrame:
    """
    Load the mechanics CSV that the user provides at runtime.

    Expected columns:
        item_id, a, b, difficulty, speedup, releaseInterval, levelUpHits

    This function:
    - checks that the file exists
    - reads CSV
    - validates required columns
    - sorts by item_id (levels follow 1..N order)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Mechanics CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Mechanics CSV is missing columns: {missing}")

    # Sort by item_id (this becomes level index)
    df = df.sort_values("item_id").reset_index(drop=True)

    # Ensure types
    df["item_id"] = df["item_id"].astype(int)
    df["difficulty"] = df["difficulty"].astype(int)
    df["levelUpHits"] = df["levelUpHits"].astype(int)

    # Floats
    df["a"] = df["a"].astype(float)
    df["b"] = df["b"].astype(float)
    df["speedup"] = df["speedup"].astype(float)
    df["releaseInterval"] = df["releaseInterval"].astype(float)

    return df
