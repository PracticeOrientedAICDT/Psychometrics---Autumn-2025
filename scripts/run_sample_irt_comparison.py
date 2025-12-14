
from pathlib import Path
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams["font.serif"] = ["Times New Roman"]

#  Compute project directories
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))


DATA_DIR = PROJECT_ROOT / "data" 

IRT_DATA_DIR = DATA_DIR / "processed"

EYEBALL_IRT = IRT_DATA_DIR / "EyeBall"/ "IRTMatrix.csv"
MEMORYGRID_IRT  = IRT_DATA_DIR / "MemoryGrid"/ "IRT_Matrix.csv"
PYRAMIDS_IRT  = IRT_DATA_DIR / "Pyramids"/ "IRT_Matrix.csv"
QUICKCALC_IRT  = IRT_DATA_DIR / "QuickCalc"/ "IRTMatrix_27items.csv"
GYRATE_IRT  = IRT_DATA_DIR / "Gyrate"/ "IRTMatrix_EXT.csv"

EXPERIMENT_DIR = DATA_DIR / "experimental" 


def save_sample_variations(df,sample_sizes,assessment_name):
    OUT_DIR = EXPERIMENT_DIR / f"{assessment_name}" / "interim" 
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for n in sample_sizes:
        if n > len(df):
            print(f"Skipping n={n}, only {len(df)} rows available.")
            continue

        # Randomly sample n rows
        sub_df = df.sample(n=n, random_state=42)

        # Save in the same format (AccountId preserved as row index)
        out_path = OUT_DIR / f"{n}_IRTMatrix.csv"

        sub_df = sub_df.drop(sub_df.columns[-1], axis=1)
        sub_df.to_csv(out_path, index=False)

        print(f"Assessment: {assessment_name} Saved sample size {n} → {out_path}")
def save_sample_var_all():
    IRT = pd.read_csv(MEMORYGRID_IRT)
    print(len(IRT))

    sample_sizes = [120,140,160]
    save_sample_variations(IRT,
                           sample_sizes=sample_sizes,
                           assessment_name="MemoryGrid")
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def plot_irt_feasibility():
    plt.rcParams["font.serif"] = ["Times New Roman"]
    irt_requirements = [
        {"assessment": "Gyrate",     "items": 21, "min_samples": 200},
        {"assessment": "Pyramids",   "items":  9, "min_samples": 20},
        {"assessment": "MemoryGrid", "items": 14, "min_samples": 160},
        {"assessment": "EyeBall",    "items":  9, "min_samples": 40},
        {"assessment": "QuickCalc",  "items": 27, "min_samples": 800},
    ]

    df = pd.DataFrame(irt_requirements)
    df = df[df["min_samples"] > 0]  # ensure no zero values

    x = df["items"].values.astype(float)
    y = df["min_samples"].values.astype(float)

    # Exponential regression function
    def exp_func(x, k, m):
        return k * np.exp(m * x)

    popt, _ = curve_fit(exp_func, x, y, p0=(1, 0.05))
    k_fit, m_fit = popt

    # Extend curve past last point
    x_ext = x.max() * 1.4  # 40% beyond max items for visual continuation
    x_fit = np.linspace(0, x_ext, 400)
    y_fit = k_fit * np.exp(m_fit * x_fit)

    plt.figure(figsize=(6,4))

    # Plot scatter points with legend labels attached to each dot
    for _, row in df.iterrows():
        plt.scatter(row["items"], row["min_samples"], s=70, label=row["assessment"])

    # Plot exponential curve
    plt.plot(x_fit, y_fit, linestyle="--", linewidth=1.3, label="Exponential fit")

    # Add a fine grid without cluttering axis ticks
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(50))
    plt.tick_params(which="minor", length=0)
    plt.grid(True, which="minor", linewidth=0.35)

    plt.xlim(0, x_ext)
    plt.ylim(0, y.max() * 1.2)

    plt.title("2PL Sample Scaling by Item Count")
    plt.xlabel("Number of Categorised Items (J)")
    plt.ylabel("Minimum Samples Needed (N)")
    plt.legend(title="Assessment", fontsize=7)
    plt.tight_layout()
    plt.show()

    print(f"\nFitted exponential: y = {k_fit:.3f} * e^({m_fit:.5f} * x)")


if __name__ == "__main__":
   plot_irt_feasibility()