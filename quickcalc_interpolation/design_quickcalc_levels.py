# quickcalc_interpolation/design_quickcalc_levels.py
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

def ask_float(prompt, default):
    """
    Ask the user for a float value in the terminal.
    If they just press Enter, use the default.
    """
    raw = input(f"{prompt} [default={default}]: ").strip()
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print("  Could not parse number, using default.")
        return default

# -----------------------
# Data loading and preprocessing
# -----------------------

def load_mechanics(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # operations is stored as a string representation of a list; parse it
    def parse_ops(x):
        if pd.isna(x) or x == "":
            return []
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    df["operations_parsed"] = df["operations"].apply(parse_ops)

    op_types = ["addition", "subtraction", "multiplication", "division", "fraction", "percentage"]
    for op in op_types:
        colname = f"has_{op}"
        df[colname] = df["operations_parsed"].apply(lambda ops: int(op in ops))

    df["level"] = pd.to_numeric(df["level"], errors="coerce")
    df["speedup"] = df["speedup"].fillna(0)
    df["bonus"] = df["bonus"].fillna(0)

    return df

def load_irt(path: str) -> pd.DataFrame:
    irt = pd.read_csv(path)
    irt["level"] = pd.to_numeric(irt["item_id"], errors="coerce")
    return irt[["level", "a", "b"]]

def join_mechanics_irt(mech: pd.DataFrame, irt: pd.DataFrame) -> pd.DataFrame:
    return mech.merge(irt, on="level", how="inner")

def get_feature_matrix(df: pd.DataFrame):
    feature_cols = [
        "n_functions",
        "releaseInterval",
        "speedup",
        "bonus",
        "balloon_lifetime_min",
        "balloon_lifetime_max",
        "has_addition",
        "has_subtraction",
        "has_multiplication",
        "has_division",
        "has_fraction",
        "has_percentage",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].fillna(0.0)
    return X, feature_cols

def fit_ridge_model(X: pd.DataFrame, y: pd.Series):
    model = Ridge(alpha=1.0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return model, {"r2": r2, "mae": mae}

# -----------------------
# Design utility
# -----------------------

def design_quickcalc_levels(
    model_b,
    feature_cols,
    target_b,
    top_k=15,
):
    """
    Suggest QuickCalc level mechanics that approximate a target difficulty b.
    Returns a DataFrame with the top_k closest designs.
    """

    speedup_vals         = [0, 100, 200, 300, 400, 500, 600]
    release_intervals    = [2200, 2500, 2800, 3000, 3200]
    operation_choices    = [
        ("addition",),
        ("subtraction",),
        ("addition", "subtraction"),
        ("addition", "multiplication"),
        ("multiplication", "division"),
        ("fraction",),
        ("percentage",),
        ("addition", "percentage"),
        ("addition", "fraction"),
    ]

    candidates = []

    for ops in operation_choices:
        for speed in speedup_vals:
            for interval in release_intervals:

                row = {
                    "n_functions": 3,
                    "releaseInterval": interval,
                    "speedup": speed,
                    "bonus": 0,
                    "balloon_lifetime_min": 10000 - speed,
                    "balloon_lifetime_max": 10000 - speed + 2000,
                    "has_addition": int("addition" in ops),
                    "has_subtraction": int("subtraction" in ops),
                    "has_multiplication": int("multiplication" in ops),
                    "has_division": int("division" in ops),
                    "has_fraction": int("fraction" in ops),
                    "has_percentage": int("percentage" in ops),
                }

                X = pd.DataFrame([row])[feature_cols].fillna(0.0)
                pred_b = model_b.predict(X)[0]
                abs_err = abs(pred_b - target_b)

                candidates.append({
                    **row,
                    "operations": ops,
                    "pred_b": float(pred_b),
                    "abs_err": float(abs_err),
                })

    out = pd.DataFrame(candidates).sort_values("abs_err").head(top_k)
    return out

# -----------------------
# main()
# -----------------------

def main():
    project_root = Path(__file__).resolve().parents[1]
    script_dir   = Path(__file__).resolve().parent

    mech_path = script_dir / "quickcalc_mechanics.csv"
    irt_path  = project_root / "out_quickcalc" / "quickcalc_item_params.csv"

    print("Script dir:", script_dir)
    print("Mechanics CSV:", mech_path)
    print("IRT CSV:", irt_path)

    mech = load_mechanics(str(mech_path))
    irt  = load_irt(str(irt_path))
    joined = join_mechanics_irt(mech, irt)

    print(f"Mechanics rows: {len(mech)}")
    print(f"IRT rows:       {len(irt)}")
    print(f"Joined rows:    {len(joined)}")

    X, feature_cols = get_feature_matrix(joined)
    model_b, stats_b = fit_ridge_model(X, joined["b"])

    print("\nModel for b (difficulty):")
    print("  R² :", round(stats_b["r2"], 3))
    print("  MAE:", round(stats_b["mae"], 3))

        # --- interactive bits: ask the user what they want ---
    target_b = ask_float("\nEnter target difficulty b (e.g., -1.5, 0, 1.0)", 0.0)
    top_k    = int(ask_float("How many designs to return (top_k)", 20))

    print(f"\nDesigning levels for target b = {target_b} (top_k = {top_k})...")


    suggestions = design_quickcalc_levels(
        model_b=model_b,
        feature_cols=feature_cols,
        target_b=target_b,
        top_k=top_k
    )

    out_file = script_dir / "quickcalc_suggested_designs.csv"
    suggestions.to_csv(out_file, index=False)

    print("\nSaved design suggestions →", out_file)
    print(suggestions.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
