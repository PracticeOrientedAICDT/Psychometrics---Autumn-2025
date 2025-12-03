import sys
import subprocess
from pathlib import Path
import argparse
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent 
SRC_DIR = PROJECT_ROOT / "src" 
sys.path.append(str(SRC_DIR)) 
R_SCRIPT = PROJECT_ROOT / "R" / "fit_irt_mirt.r"

DATA_DIR = PROJECT_ROOT / "data" 
IRT_DATA_DIR = DATA_DIR / "processed"
DIAGNOSTIC = DATA_DIR / "diagnostics"


EYEBALL_IRT = DIAGNOSTIC / "labelled_irt" / "EyeBall.csv"
MEMORYGRID_IRT  = DIAGNOSTIC / "labelled_irt" / "MemoryGrid.csv"
PYRAMIDS_IRT  = DIAGNOSTIC / "labelled_irt" / "Pyramids.csv"
QUICKCALC_IRT  = DIAGNOSTIC / "labelled_irt" / "QuickCalc.csv"
GYRATE_IRT  = DIAGNOSTIC / "labelled_irt" / "Gyrate.csv"


OUT_DIR = DATA_DIR / "diagnostics" 
# ===== Function to run the R script =====
def run_fit_irt_mirt_r(input_csv: str,
           
                       out_abilities_csv: str = "abilities.csv",
                       out_items_csv: str = "item_params.csv",
                       n_factors: int = 1,
                       itemtype: str = "3PL",
                       method: str = "EM",
                       verbose: bool = False,
                       max_cycles: Optional[int] = None):

    """
    Runs fit_irt_mirt.r via Rscript from within an IDE.
    Prints R output to console and reports errors clearly.

    Parameters
    ----------
    input_csv : str
        Path to the input response matrix CSV.
    out_abilities_csv : str
        Output filename for the abilities CSV.
    out_items_csv : str
        Output filename for the item-parameter CSV.
    n_factors : int
        Number of latent factors (default=1).
    itemtype : str
        IRT model type, e.g. "3PL", "2PL".
    method : str
        Fit method, typically "EM".
    verbose : bool
        Whether to run Rscript in verbose mode.
    """

    # Ensure R script exists
    if not R_SCRIPT.exists():
        raise FileNotFoundError(f"R script not found at: {R_SCRIPT}")

    # Build the Rscript command
    cmd = [
        "Rscript",
        str(R_SCRIPT),
        "--input", input_csv,
        "--out_abilities", out_abilities_csv,
        "--out_items", out_items_csv,
        "--factors", str(n_factors),
        "--itemtype", itemtype,
        "--method", method
    ]

    if max_cycles is not None:
        cmd.extend(["--max_cycles", str(max_cycles)])

    if verbose:
        cmd.append("--verbose")

    # Run the R script
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Stream output back to Android Studio console
        for line in process.stdout:
            print(line.strip())

        process.wait()

        # If R failed, show error output
        if process.returncode != 0:
            err = process.stderr.read()
            raise RuntimeError(f"Rscript failed with exit code {process.returncode}:\n{err}")

        print("✅ R script executed successfully!")

    except Exception as e:
        print("⚠️ Error while running R script:")
        raise e
    
def main():
    run_fit_irt_mirt_r(
        input_csv=str(GYRATE_IRT),
        out_abilities_csv=str(OUT_DIR /"labelled_irt"/ "gyrate_abilities_labeled.csv"),
        out_items_csv=str(OUT_DIR /"labelled_irt"/ "gyrate_items.csv"),
        n_factors=1,
        itemtype="2PL",
        method="EM",
        verbose=False,
        max_cycles=200000
    )

main()