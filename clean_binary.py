import pandas as pd

# Load the binary.csv file
input_file = "data/EyeBall/binary.csv"
data = pd.read_csv(input_file)

# Remove the AccountId column
data = data.drop(columns=['AccountId'])

# Remove columns with only one unique value
data = data.loc[:, data.nunique() > 1]

# Save the cleaned file
output_file = "clean_binary.csv"
data.to_csv(output_file, index=False)
print(f"Cleaned binary.csv saved to {output_file}")