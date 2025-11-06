# Data Pre-Processing
## Cleaning
- Clean data  
- Change to IRT format:

  | participant_id | item_id | response |
  |----------------|----------|-----------|
  | 1028 | 1 | 1 |
  | 1028 | 2 | 0 |
  | ... | ... | ... |

## IRT Processing: with MIRT
- Prepare data for MIRT and save to csv

  | participant_id | item_id 1 | item_id 2 | ... |
  |----------------|----------|-----------|-----------|
  | participant_id 1 | response | response | ... |
  | participant_id 2 | response | response | ... |
  |  ... | ... | ... | ... |


- Open R console
```r
install.packages("mirt")
```
```r
source("path/to/project/irt/MIRT.r") 
```
```r
fit_irt(
  input_csv = "path_to_project/data/assessment_name/mirt_in.csv",
  out_abilities_csv = "path_to_project/data/assessment_name/modelling/abilities.csv",
  out_items_csv = "path_to_project/data/assessment_name/modelling/item_params.csv",
  itemtype = "2PL",
  method = "EM",
  mirt_args = list(SE = TRUE,quadpts = 41),
  technical = list(NCYCLES = 1500)  
)
```
- Then:
```python
import io_utils

abilities_df = load_csv_into_df("data/assessment_name/abilities.csv")
items_df = load_csv_into_df("data/assessment_name/item_params.csv")
```

# Simulating Data

-Example:

```python
import simulate

 simulated_df = simulate.get_simulated_scores(abilities_df=abilities_df,
                                                 item_params_df=items_df,
                                                 num_lives=5,
                                                 num_sub_levels=3)
```

# Visualise 
## Compare Real and Simulated Data
```python
from irt.visualise import simulated_plot_comparison

fig = simulated_plot_comparison(scores_df=df,
                          simulated_scores_df=simulated_df,
                          assessment_name="Assessment Name")

fig.savefig("path/to/project/data/{assessmenet_name}/modelling/Simulation_Comparison.png", dpi=300, bbox_inches="tight")
```

## ICC curves
```python
from irt.visualise import plot_icc

fig = plot_icc(scores_df=df,
        simulated_scores_df=simulated_df,
        assessment_name="Assessment Name")


fig.savefig("path/to/project/data/{assessmenet_name}/modelling/ICC.png", dpi=300, bbox_inches="tight")
```



    

