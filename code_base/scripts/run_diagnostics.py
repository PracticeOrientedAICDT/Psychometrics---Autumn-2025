from pathlib import Path
import sys
import numpy as np
import pandas as pd

#  Compute project directories
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIAGNOSTIC_DIR = PROJECT_ROOT / "data" / "diagnostics"

from utils.io_utils import get_all_irt_matrix_paths
from init_core.diagnostics import irt_unidimensionality_eigen as irt_un

def run():
    irt_matrix_paths = get_all_irt_matrix_paths(DATA_DIR)
    for p in irt_matrix_paths:
        assessment = p.parent.name              
        stem = p.stem                          
        out_name = f"{assessment}_{stem}.png"  
        save_path = OUTPUT_DIAGNOSTIC_DIR / out_name
        
        irt_un.analyze_file(p, 
                            assessment_name=assessment,
                            save_path=save_path)

if __name__ == "__main__":
    run()
