import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# -------------------------------------------------------------------
# Configuration (hard-coded)
# -------------------------------------------------------------------
LIST_FILE = "irt_data_paths.txt"  # text file listing CSV paths
DELIMITER = ","                   # CSV field separator
SKIP_HEADER = 1                   # <-- now 1, to skip the header row
# -------------------------------------------------------------------

def load_csv_paths(list_path):
    """
    Read CSV file paths from a text file, ignoring empty lines and comments.
    """
    csv_paths = []
    try:
        with open(list_path, "r") as f:
            for line in f:
                line = line.strip()
                # Ignore empty lines and lines starting with '#'
                if not line or line.startswith("#"):
                    continue
                csv_paths.append(line)
    except Exception as e:
        print(f"Error reading list file '{list_path}': {e}", file=sys.stderr)
        sys.exit(1)

    if not csv_paths:
        print(f"No valid CSV paths found in '{list_path}'.", file=sys.stderr)
        sys.exit(1)

    return csv_paths


def analyze_file(csv_path,
                 assessment_name= None,
                 save_path = None):
    """
    Load one CSV file, compute eigenvalues of the item correlation matrix,
    print diagnostics, and show a scree plot.
    """
    print("=" * 80)
    print(f"Analyzing file: {csv_path}")
    print("=" * 80)

    # --------------------------------------
    # 1. Load response matrix X from CSV
    # --------------------------------------
    try:
        X = np.loadtxt(csv_path, delimiter=DELIMITER, skiprows=SKIP_HEADER)
    except Exception as e:
        print(f"  Error loading file '{csv_path}': {e}", file=sys.stderr)
        return

    # Ensure X is 2D (loadtxt can return 1D if only one column)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # IMPORTANT: drop first column (participant_id), keep only items
    if X.shape[1] > 1:
        X = X[:, 1:]

    n_persons, n_items = X.shape
    print(f"  Response matrix shape (after dropping ID column): {X.shape} (persons x items)")

    if n_items < 2:
        print("  Only one item found; unidimensionality cannot be assessed.\n")
        return

    # --------------------------------------
    # 2. Compute item correlation matrix
    # --------------------------------------
    R = np.corrcoef(X, rowvar=False)  # shape (n_items, n_items)
    print("  Correlation matrix shape:", R.shape)

    # --------------------------------------
    # 3. Eigen-decomposition
    # --------------------------------------
    eigenvalues, eigenvectors = np.linalg.eigh(R)

    # Sort eigenvalues (and eigenvectors) in descending order
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    print("\n  Eigenvalues (sorted descending):")
    print("  ", eigenvalues)

    # --------------------------------------
    # 4. Proportion of variance explained
    # --------------------------------------
    total_variance = n_items  # for a correlation matrix, trace = number of items
    prop_variance = eigenvalues / total_variance
    cum_prop_variance = np.cumsum(prop_variance)

    print("\n  Proportion of variance explained by each factor:")
    for i, (ev, pv, cpv) in enumerate(
        zip(eigenvalues, prop_variance, cum_prop_variance), start=1
    ):
        print(
            f"    Factor {i:2d}: "
            f"Eigenvalue = {ev:6.3f}, "
            f"Prop Var = {pv:6.3f}, "
            f"Cum Prop Var = {cpv:6.3f}"
        )

    # --------------------------------------
    # 5. Simple unidimensionality diagnostics
    # --------------------------------------
    lambda1 = eigenvalues[0]
    lambda2 = eigenvalues[1]
    ratio_1_2 = lambda1 / lambda2

    print("\n  First eigenvalue: ", lambda1)
    print("  Second eigenvalue:", lambda2)
    print("  Ratio λ1 / λ2:    ", ratio_1_2)

    print("\n  Heuristic interpretation:")
    print("  - If λ1 is much larger than λ2, and others are small, that supports unidimensionality.")
    print("  - Inspect the scree plot to see if there is a clear 'elbow' after the first eigenvalue.\n")

    # --------------------------------------
    # 6. Scree plot
    # --------------------------------------
    fig, ax = plt.subplots()   # <-- capture fig & ax
    ax.plot(range(1, n_items + 1), eigenvalues, marker="o")
    ax.set_xlabel("Factor number")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"{assessment_name} Scree plot: {csv_path.name}")
    ax.grid(True)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)  # ensure it's a Path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    plt.show()


def main():
    print(f"Reading CSV paths from: {LIST_FILE}\n")
    csv_paths = load_csv_paths(LIST_FILE)

    print(f"Found {len(csv_paths)} CSV files to analyze.\n")

    for csv_path in csv_paths:
        analyze_file(csv_path)


if __name__ == "__main__":
    main()
