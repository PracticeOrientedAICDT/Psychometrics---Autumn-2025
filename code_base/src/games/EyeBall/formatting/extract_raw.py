import pandas as pd

# Function to extract specific rows and columns from the file
def extract_specific_rows_and_columns(input_file, output_file):
    # Read the CSV file
    data = pd.read_csv(input_file)

    # Generate the row indices: start at 95, add 124 each time
    row_indices = list(range(281, len(data) + 1, 372))

    # Extract the specific rows
    extracted_data = data.iloc[row_indices]

    # Select only the desired columns
    columns_to_keep = ["AccountId", "Score", "Level", "FailedLevels"]
    extracted_data = extracted_data[columns_to_keep]

    # Handle duplicate AccountIDs by appending _2, _3, etc.
    extracted_data['AccountId'] = extracted_data['AccountId'].astype(str)  # Ensure AccountId is a string
    extracted_data['AccountId'] = extracted_data['AccountId'] + extracted_data.groupby('AccountId').cumcount().add(1).astype(str).replace('1', '', regex=False).apply(lambda x: f"_{x}" if x != '' else '')

    # Save the extracted rows and columns to a new CSV file
    extracted_data.to_csv(output_file, index=False)
    print(f"Extracted rows and columns saved to {output_file}")

# Example usage
input_file = "Pyramids/raw.csv"
output_file = "Pyramids/raw_short.csv"
extract_specific_rows_and_columns(input_file, output_file)