import base64
import os
from pathlib import Path


# ------------------------------------------------------------
# CLAMP
# ------------------------------------------------------------
def clamp(x, min_val, max_val):
    """Clamp x to range [min_val, max_val]."""
    return max(min_val, min(x, max_val))


# ------------------------------------------------------------
# LOAD BALLOON IMAGE AS BASE64
# ------------------------------------------------------------
def load_image_base64():
    """
    Loads balloon.png if available in ./assets
    Otherwise falls back to a built-in tiny 1-colour balloon icon.

    Returns: base64-encoded PNG image (string)
    """

    # Try to load from assets/
    local_path = Path(__file__).parent / "assets" / "balloon.png"
    if local_path.exists():
        with open(local_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Fallback: embedded ~50×50 px simple balloon PNG (red-ish)
    # This is a minimal placeholder balloon so the game ALWAYS works.
    fallback_png_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAADIAAAAyCAYAAAAeP4ixAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz"
        "AAALEgAACxIB0t1+/AAAABl0RVh0Q3JlYXRpb24gVGltZQAwNS8wMy8xNqKCb3UAAAEPSURBVFiF"
        "7ZexDcIwEERHn7/0E3Uf4DqgHcoHGwEq+yLkOlLXTr4Wr9L5xRcmk6VwBEmcLkAMPx/+qvOB0AhOk"
        "gkAMPADrABtABewAWwAFsABrAAewAfesF2fzGEGZm4G6dkcnnrIALlTyBC0bYKT1eZqUPVHtVHCnQ"
        "fmHzWmvMRGsiE9zraFMvx6bMpiKFFnzvolG/GpNZgbf+5Q5e0siJmq9hw3rroWAtxEvsC0BpvyEuk"
        "hhZCvGSqtEtFPi0PfrxGX2YeliYg+6TSGT+b/xGaxzwoMPmxjSPdZRhqUTrWyd4InENcy+XUG6u+Y"
        "GZNVVRFl1FdYyqS/fWznuaCG2lLBVmOAXoAWwAGsAB7ABbAAVsABbAAbAAP77AH2Fne5+5ZTCDwAA"
        "AABJRU5ErkJggg=="
    )

    return fallback_png_base64
