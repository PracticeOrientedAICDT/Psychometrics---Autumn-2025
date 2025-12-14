# Quick Calc Workspace

Orgainised workspace for IRT analysis for the QuickCalc game in Python and R, including data, preprocessing, visualization, simulation, and Computer Adaptive Testing (CAT).

##  Folder Structure

├── data/ # Item parameters & abilities ONLY

├── preprocessing/ # Convert raw data to IRT format

├── visualisation/ # ICC plots & distribution plots

├── simulation/ # Simulations using IRT parameters

└── computer_adaptive_testing/ # CAT for IRT 

### `data/`
**Purpose:** Store only the IRT inputs/outputs:
- `item_params.*` (e.g., `item_id, a, b, c`)
- `abilities.*` (e.g., `participant_id, theta`)

### `preprocessing/`
**Purpose:** Convert raw assessment exports to IRT-ready data.

**Typical outputs:**
- Long format: `participant_id, item_id (or level), response (0/1)`
- Wide matrix (items as columns): 0/1 values for MIRT

### `visualisation/`
**Purpose:** Diagnostics & plots.
- Item Characteristic Curves (ICCs)
- Item/Test scores
- Distributions of scores

**Outputs:** `.png`, `.pdf`  

### `simulation/`
**Purpose:** Simulate responses/tests using known IRT parameters.

### `computer_adaptive_testing/`
**Purpose:** Prototype CAT pipelines.
- **Ability updates:** EAP
