import pandas as pd

# Long
long_df = pd.read_csv("data/memorygrid/memorygrid_irt_long.csv")
assert set(long_df.columns) == {"participant_id", "item_id", "response"}
assert long_df["response"].isin([0,1]).all()
assert long_df["item_id"].between(1,15).all()
assert not long_df.duplicated(subset=["participant_id","item_id"]).any()

# Wide
wide_df = pd.read_csv("data/memorygrid/memorygrid_mirt.csv")

# Cast header names to int (except the first column)
item_cols = [int(c) for c in wide_df.columns[1:]]
assert item_cols == list(range(1,16)), "Item columns must be 1..15 in order"

# Ensure values are binary
assert wide_df.iloc[:,1:].isin([0,1]).all().all()

# Difficulty sanity: means should generally decrease with level (allow small noise)
means = long_df.groupby("item_id")["response"].mean()
roughly_decreasing = all(means.iloc[i] >= means.iloc[i+1] - 0.02 for i in range(len(means)-1))
print("Roughly decreasing by level:", roughly_decreasing)
print(means.round(3))
