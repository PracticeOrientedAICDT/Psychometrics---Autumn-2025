from pathlib import Path
from typing import Iterable, Dict



def validate_csv_paths(paths: Iterable[Path]) -> Dict[Path, bool]:
    """
    Validate that each given path exists, is a file, and ends with .csv.

    Parameters
    ----------
    paths : Iterable[Path]
        List/tuple of Path objects to validate.

    Returns
    -------
    Dict[Path, bool]
        A dictionary mapping each path -> True (valid) or False (invalid)
    """
    results = {}

    for p in paths:
        if not isinstance(p, Path):
            results[p] = False
            continue

        is_valid = p.exists() and p.is_file() and p.suffix.lower() == ".csv"
        results[p] = is_valid

    return results

def get_all_irt_matrix_paths(DATA_DIR) -> list[Path]:
    """
    Search inside:
        PROJECT_ROOT / "data" / "processed" / <any folder>
    and return all CSV paths whose filename starts with 'IRTMatrix'.

    Returns
    -------
    list[Path]
        List of full paths to matching CSV files.
    """

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Processed data directory does not exist: {DATA_DIR}")

    matches = []

    # iterate through all subfolders inside processed/
    for subdir in DATA_DIR.iterdir():
        if subdir.is_dir():
            for file in subdir.iterdir():
                if file.is_file() and file.suffix.lower() == ".csv":
                    if file.name.startswith("IRTMatrix"):
                        matches.append(file)

    return matches
