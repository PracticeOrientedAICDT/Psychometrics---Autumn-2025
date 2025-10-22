# Psychometrics---Autumn-2025
 Psychometrics practial project 

## Testing WordMatch

### Data Pre-Processing

```python
csv_path = "data/WordMatch/Binary_WordMatch.csv"

raw_df = load_csv_into_df(csv_path)
df = clean.get_wordmatch_df(raw_df,verbose=False)
irt_df = irt_format.create_irt_input(df)
```

### IRT Processing: with Girth
```python
abilities_df, items_df = process.fit_with_girth(
    irt_df,
    epochs=10000000,
    model_spec="3PL",
    quadrature_n= 100)

save_df_as_csv(abilities_df,"abilities.csv") #optional
save_df_as_csv(items_df,"items.csv") #optional
```

### Simulating Data
```python
simulated_df = process.simulate_irt_scores(
    abilities_df=abilities_df,
    item_latents_df=items_df) 
```

### (Optional) Visualise and Compare Real and Simulated Data
```python
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
    

