# IRT Modelling (R / mirt)

This module contains tools for fitting 1PL/2PL/3PL and multidimensional IRT models using the **mirt** package in R.  
The core entry point is the `fit_irt()` wrapper function, which handles data loading, model fitting, fallback strategies, and exporting results for downstream analysis in Python.

---

## What `fit_irt()` Does

The wrapper is designed to:

- Fit **unidimensional or multidimensional** IRT models  
- Support **Rasch**, **2PL**, **3PL**, or mixed item types  
- Export clean, analysis-ready CSVs:
  - `abilities.csv` → participant ability estimates (θ, θ₂, …)
  - `item_params.csv` → item parameters (a, b, c)
- Automatically retry **3PL** models with a stable 2PL-based starting configuration if the first attempt fails  
- Normalise ID columns, handle missing values, and extract item names consistently  

This ensures reproducible and consistent fitting across different assessments.

---

## Input Format

`fit_irt()` expects a **wide-format** CSV of 0/1/NA item responses:

| participant_id | item 1 | item 2 | item 3 | ... |
|----------------|--------|--------|--------|-----|
| 1028           | 1      | 0      | NA     | ... |
| 1097           | 1      | 1      | 0      | ... |
| ...            | ...    | ...    | ...    | ... |

- An ID column is optional.  
  - If present, names such as `participant_id` or `AccountId` will be detected.  
  - If missing, IDs are auto-generated as `P1`, `P2`, …  
- All item columns must be numeric and represent correct/incorrect responses.

Save this file as: data/<assessment_name>/mirt_in.csv

# 1. Installing Dependencies
```r
install.packages("mirt")
source("path/to/project/irt/MIRT.r")   # loads fit_irt()
```

# 2. The fit_irt() Function

#### Key Arguments

| Argument | Description |
|---------|-------------|
| `input_csv` | Path to the wide-format response file used as model input. |

| `out_abilities_csv` | Output file for estimated participant ability values (θ). |
| `out_items_csv` | Output file for estimated item parameters (a, b, c). |

| `itemtype` | IRT model: `"Rasch"`, `"2PL"`, `"3PL"`, etc. |
| `n_factors` | Number of latent dimensions (default 1). |
| `method` | Estimation method: `"EM"`, `"QMCEM"`, or `"MHRM"`. |
| `technical` | Algorithm controls (e.g., EM cycles, MHRM draws). |
| `mirt_args` | Additional mirt settings (e.g., quadrature points for EM). |
| `fscores_method` | Person scoring method: `"EAP"` (default), `"MAP"`, `"ML"`, `"WLE"`. |


#### When to Use Each Model
- **Rasch**  
  - All items share the same discrimination.  
  - Useful for simple difficulty-only modelling.
- **2PL (default)**  
  - Estimates discrimination (a) and difficulty (b).  
  - More flexible and stable than 3PL for small datasets.

- **3PL**  
  - Adds a guessing parameter (c).  
  - Useful for multiple-choice tasks where random guessing is plausible.  
  - `fit_irt()` includes an automatic fallback:  
    if the initial 3PL fit fails, it first fits a 2PL model to obtain good starting values, then retries 3PL using those starts.

- **Multidimensional (n_factors > 1)**  
  - For tasks with multiple latent skills (e.g., speed + accuracy, or quantity + spatial).  
  - Outputs θ₁, θ₂, … in the abilities file.

## Full Function Signature
```r
fit_irt(
  input_csv,
  id_cols           = c("participant_id", "AccountId"),
  n_factors         = 1,
  itemtype          = "3PL",
  method            = "EM",
  verbose           = FALSE,
  technical         = NULL,
  mirt_args         = list(),
  fscores_method    = "EAP",
  fscores_args      = list()
)
```
This produces:
- abilities.csv containing estimated participant θ values
- item_params.csv containing item discrimination (a), difficulty (b), and optionally guessing (c)

This data is saved to: data/<assessment_name>/modelling/...

## 3. Load IRT outputs in Python
```python
from io_utils import load_csv_into_df

abilities_df = load_csv_into_df("data/<assessment_name>/abilities.csv")
items_df = load_csv_into_df("data/<assessment_name>/item_params.csv")
```

