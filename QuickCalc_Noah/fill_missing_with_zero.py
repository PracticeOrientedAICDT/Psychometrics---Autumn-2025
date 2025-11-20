import pandas as pd

# Load the input CSV file
input_file = "data/QuickCalc/mirt_in.csv"
data = pd.read_csv(input_file)

# Replace all missing values with zero
data_filled = data.fillna(0)

# Save the updated data to a new CSV file
output_file = "data/QuickCalc/mirt_in_full.csv"
data_filled.to_csv(output_file, index=False)
print(f"File with missing values filled saved to {output_file}")