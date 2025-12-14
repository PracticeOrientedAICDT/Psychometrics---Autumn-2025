#!/usr/bin/env python3
"""
local_independence_test.py

Compute Yen's Q3 local independence statistic for multiple tests,
each defined by three CSV files:
    - responses: item response matrix (0/1)
    - abilities: participant_id, theta
    - items: item_id, a, b, c (c can be 0 for 2PL)

Configuration structure:
- A master file "LoclaIndependencePaths.txt" containing one config file path per line.
- Each config file contains lines like:
      responses= /path/to/IRTMatrix_test1.csv
      abilities= /path/to/IRTMatrix_2PL_abilities_test1.csv
      items= /path/to/IRTMatrix_2PL_items_test1.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration parsing
# ---------------------------------------------------------------------

def load_test_configs(master_file: Path):
    """
    Read the master file (e.g. LoclaIndependencePaths.txt) and
    return a list of test config dicts:
        {
          "name": <config file stem>,
          "responses": Path(...),
          "abilities": Path(...),
          "items": Path(...)
        }
    """
    configs = []
    with master_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cfg_path = Path(line)
            cfg = parse_single_config(cfg_path)
            configs.append(cfg)
    return configs


def parse_single_config(cfg_path: Path):
    """
    Parse a single config file with lines of the form key=value.
    Required keys: responses, abilities, items.
    """
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    data = {}
    with cfg_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(
                    f"Invalid line in config {cfg_path}: {line!r}. "
                    f"Expected 'key = value'."
                )
            key, value = line.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            data[key] = value

    for required in ("responses", "abilities", "items"):
        if required not in data:
            raise ValueError(
                f"Missing '{required}' entry in config file {cfg_path}"
            )

    return {
        "name": cfg_path.stem,
        "responses": Path(data["responses"]),
        "abilities": Path(data["abilities"]),
        "items": Path(data["items"]),
    }


# ---------------------------------------------------------------------
# Data loading and probability computation
# ---------------------------------------------------------------------

def load_data_for_test(config: dict):
    """
    Load responses, abilities, and item parameters for a single test.

    Returns:
        item_names : list of item column names in order
        observed   : ndarray (N, J) of 0/1 responses
        theta      : ndarray (N,) of ability estimates
        a, b, c    : ndarrays (J,) of item parameters aligned with item_names
    """
    resp_path = config["responses"]
    abil_path = config["abilities"]
    items_path = config["items"]

    print(f"\n--- Loading data for test: {config['name']} ---")
    print(f"Responses : {resp_path}")
    print(f"Abilities : {abil_path}")
    print(f"Items     : {items_path}")

    if not resp_path.exists():
        raise FileNotFoundError(f"Responses file not found: {resp_path}")
    if not abil_path.exists():
        raise FileNotFoundError(f"Abilities file not found: {abil_path}")
    if not items_path.exists():
        raise FileNotFoundError(f"Items file not found: {items_path}")

    # Load data
    df_resp = pd.read_csv(resp_path)
    df_abil = pd.read_csv(abil_path)
    df_items = pd.read_csv(items_path)

    # Merge responses and abilities by participant_id
    if "participant_id" not in df_resp.columns or "participant_id" not in df_abil.columns:
        raise ValueError(
            "Both responses and abilities files must have a 'participant_id' column."
        )

    df = df_resp.merge(df_abil, on="participant_id", how="inner")

    if df.empty:
        raise ValueError(
            f"No overlapping participant_id between {resp_path} and {abil_path}."
        )

    # Item columns: all columns in df_resp except participant_id
    item_names = [c for c in df_resp.columns if c != "participant_id"]
    if not item_names:
        raise ValueError(f"No item columns found in responses file: {resp_path}")

    # Ensure items file has matching item_ids
    if "item_id" not in df_items.columns:
        raise ValueError(
            f"Items file must have an 'item_id' column: {items_path}"
        )

    # Check that all item_names appear in item_id
    item_params = df_items.set_index("item_id")

    missing = [name for name in item_names if name not in item_params.index]
    if missing:
        raise ValueError(
            f"The following item columns in responses are missing in the items file: "
            f"{missing}"
        )

    # Order item parameters to match item_names
    item_params = item_params.loc[item_names]

    # Extract observed responses (N x J)
    observed = df[item_names].to_numpy(dtype=float)

    # Ability estimates
    if "theta" not in df.columns:
        raise ValueError(
            f"Abilities file must have a 'theta' column: {abil_path}"
        )
    theta = df["theta"].to_numpy(dtype=float)

    # Item parameters a, b, c (if c is missing, assume 0)
    if "a" not in item_params.columns or "b" not in item_params.columns:
        raise ValueError(
            f"Items file must have at least 'a' and 'b' columns: {items_path}"
        )

    a = item_params["a"].to_numpy(dtype=float)
    b = item_params["b"].to_numpy(dtype=float)
    if "c" in item_params.columns:
        c = item_params["c"].to_numpy(dtype=float)
    else:
        c = np.zeros_like(a)

    return item_names, observed, theta, a, b, c


def compute_predicted_probs(theta, a, b, c, D=1.702):
    """
    Compute model-predicted probabilities P_ij under a 2PL/3PL form:

        P_ij = c_j + (1 - c_j) * 1 / (1 + exp(-D * a_j * (theta_i - b_j)))

    theta : (N,)
    a, b, c : (J,)
    Returns P : (N, J)
    """
    theta = np.asarray(theta).reshape(-1, 1)  # N x 1
    a = np.asarray(a).reshape(1, -1)          # 1 x J
    b = np.asarray(b).reshape(1, -1)          # 1 x J
    c = np.asarray(c).reshape(1, -1)          # 1 x J

    z = D * a * (theta - b)          # N x J
    logistic = 1.0 / (1.0 + np.exp(-z))
    P = c + (1.0 - c) * logistic     # N x J
    return P


# ---------------------------------------------------------------------
# Q3 computation
# ---------------------------------------------------------------------

def compute_q3(observed: np.ndarray, predicted: np.ndarray):
    """
    Compute Yen's Q3 matrix and bias-corrected aQ3.

    observed : (N, J)
    predicted: (N, J)

    Returns:
        q3          : (J, J) matrix of Q3 correlations (NaN on diagonal)
        aq3         : (J, J) matrix of bias-corrected Q3
        mean_offdiag: float, mean of off-diagonal Q3 values
    """
    if observed.shape != predicted.shape:
        raise ValueError("Observed and predicted must have the same shape.")

    residuals = observed - predicted  # N x J

    # Correlation between columns (items)
    q3 = np.corrcoef(residuals, rowvar=False)

    # Remove diagonal (self-correlation)
    np.fill_diagonal(q3, np.nan)

    mean_offdiag = np.nanmean(q3)
    aq3 = q3 - mean_offdiag

    return q3, aq3, mean_offdiag


def q3_matrix_to_long_form(item_names, q3_matrix, aq3_matrix):
    """
    Convert Q3 and aQ3 matrices to a long-format DataFrame with columns:
        item_i, item_j, Q3, aQ3
    Only pairs with i < j are kept.
    """
    J = len(item_names)
    records = []
    for j in range(J):
        for k in range(j + 1, J):
            records.append({
                "item_i": item_names[j],
                "item_j": item_names[k],
                "Q3": q3_matrix[j, k],
                "aQ3": aq3_matrix[j, k],
            })
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------
# Analysis for a single test
# ---------------------------------------------------------------------

def analyze_test(config: dict, D: float, top_n: int, threshold: float):
    """
    Run Q3 analysis for a single test defined by config dict.
    """
    item_names, observed, theta, a, b, c = load_data_for_test(config)

    # Compute predicted probabilities from 2PL/3PL parameters
    predicted = compute_predicted_probs(theta, a, b, c, D=D)

    # Compute Q3
    q3, aq3, mean_offdiag = compute_q3(observed, predicted)

    # DataFrames with names
    q3_df = pd.DataFrame(q3, index=item_names, columns=item_names)
    aq3_df = pd.DataFrame(aq3, index=item_names, columns=item_names)
    pairs_df = q3_matrix_to_long_form(item_names, q3, aq3)
    pairs_df_sorted = pairs_df.sort_values("aQ3", ascending=False)

    # Output filenames: use config name
    out_dir = config["responses"].parent
    base = config["name"]

    q3_matrix_file = out_dir / f"q3_{base}.csv"
    aq3_matrix_file = out_dir / f"aq3_{base}.csv"
    pairs_file = out_dir / f"q3_pairs_{base}.csv"

    q3_df.to_csv(q3_matrix_file, index=True)
    aq3_df.to_csv(aq3_matrix_file, index=True)
    pairs_df_sorted.to_csv(pairs_file, index=False)

    # Console summary
    print(f"\n=== Results for test: {base} ===")
    print(f"Mean off-diagonal Q3: {mean_offdiag:.4f}")
    print(f"\nTop {top_n} item pairs by aQ3:")
    print(pairs_df_sorted.head(top_n).to_string(index=False))

    flagged = pairs_df_sorted[pairs_df_sorted["aQ3"] > threshold]
    print(f"\nNumber of item pairs with aQ3 > {threshold}: {len(flagged)}")
    if not flagged.empty:
        print("Flagged pairs (first 10):")
        print(flagged.head(10).to_string(index=False))

    print(f"\nSaved Q3 matrix to:  {q3_matrix_file}")
    print(f"Saved aQ3 matrix to: {aq3_matrix_file}")
    print(f"Saved Q3 pairs to:   {pairs_file}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute Yen's Q3 for multiple tests using 2PL/3PL item parameters."
    )
    parser.add_argument(
        "--master-file",
        type=str,
        default="LoclaIndependencePaths.txt",
        help="Master paths file listing per-test config files "
             "(default: LoclaIndependencePaths.txt)",
    )
    parser.add_argument(
        "--D",
        type=float,
        default=1.702,
        help="Scaling constant D in the logistic model (default: 1.702). "
             "Use the same value as in your IRT estimation software.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top item pairs (by aQ3) to show in the console (default: 10).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.20,
        help="Threshold on aQ3 for flagging local dependence (default: 0.20).",
    )

    args = parser.parse_args()

    master_file = Path(args.master_file)
    if not master_file.exists():
        raise FileNotFoundError(f"Master file not found: {master_file}")

    test_configs = load_test_configs(master_file)
    if not test_configs:
        raise ValueError(f"No test configs found in {master_file}")

    for cfg in test_configs:
        analyze_test(cfg, D=args.D, top_n=args.top_n, threshold=args.threshold)


if __name__ == "__main__":
    main()
