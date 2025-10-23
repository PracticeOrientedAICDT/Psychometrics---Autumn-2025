# Testing WordMatch

## Data Pre-Processing
- Clean data  
- Change to IRT format:

  | participant_id | item_id | response |
  |----------------|----------|-----------|
  | 1028 | 1 | 1 |
  | 1028 | 2 | 0 |
  | ... | ... | ... |

```python
from wordmatch import clean,irt_format

csv_path = "data/WordMatch/Binary_WordMatch.csv"

raw_df = load_csv_into_df(csv_path)
df = clean.get_wordmatch_df(raw_df,verbose=False)
irt_df = irt_format.create_irt_input(df)
```

## IRT Processing: with MIRT
- Prepare data for MIRT and save to csv

  | participant_id | item_id 1 | item_id 2 | ... |
  |----------------|----------|-----------|-----------|
  | participant_id 1 | response | response | ... |
  | participant_id 2 | response | response | ... |
  |  ... | ... | ... | ... |

```python
from irt import process
import io_utils

irt_df = process.prepare_mirt_input(irt_df)
irt_in_csv = "data/WordMatch/mirt_in.csv"
save_df_as_csv(irt_df,irt_in_csv)

```
- Open R console
```r
install.packages("mirt")
```
```r
source("path/to/project/fit_irt.r") 
```
```r
fit_irt(
  input_csv = "path/to/project/data/WordMatch/mirt_in.csv",
  out_abilities_csv = "/path/to/project/data/WordMatch/abilities.csv",
  out_items_csv = "path/to/project/data/WordMatch/item_params.csv"
)
```
- Then:
```python
import io_utils

abilities_df = load_csv_into_df("data/WordMatch/abilities.csv")
items_df = load_csv_into_df("data/WordMatch/item_params.csv")
```

## IRT Processing: with Girth
```python
from irt import process
import io_utils

abilities_df, items_df = process.fit_with_girth(
    irt_df,
    epochs=10000000,
    model_spec="3PL",
    quadrature_n= 100)

save_df_as_csv(abilities_df,"abilities.csv") #optional
save_df_as_csv(items_df,"items.csv") #optional
```

## Simulating Data
```python
from irt import process

simulated_df = process.simulate_data_stochastic(
    abilities_df=abilities_df,
    item_latents_df=items_df) 
```

## (Optional) Visualise and Compare Real and Simulated Data
```python
import matplotlib.pyplot as plt
from irt import visualise 

bins = 25
groups = [
        [1, 2],        # subplot 1
        [3, 4, 5],     # subplot 2
        [6, 7, 8, 9],  # subplot 3
    ]

    # build one plotter per subplot
    plotters = [
        lambda ax: visualise.plot_percentile_distribution(
            df, ax=ax, title="Percentiles: Observed",bins=bins
        ),
        lambda ax: visualise.plot_percentile_distribution(
            simulated_df, ax=ax, title="Percentiles: Simulated",bins=bins
        ),
        lambda ax: visualise.plot_icc(
            items_df, ax=ax, title="Item Characteristic Curves"
        ),
    ]

    # 1 row × 3 columns (one window)
    fig, axes = visualise.compose_plots(
        plotters,
        ncols=3,              # 3 separate subplots side-by-side
        figsize=(14, 4),
        suptitle="IRT Overview",
        tight=True,
    )
    plt.show()
```
    

