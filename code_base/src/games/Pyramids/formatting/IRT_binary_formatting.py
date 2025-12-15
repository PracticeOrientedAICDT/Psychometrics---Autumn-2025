import pandas as pd
import numpy as np

# Function to transform the raw short file into a binary format
def transform_to_binary(input_file, output_file):
    # Read the input CSV file
    data = pd.read_csv(input_file)

    # Get unique AccountIDs
    unique_ids = data['AccountId'].unique()

    # Create an empty DataFrame for the output
    columns = [f"{i}:{j}" for i in range(1, 11) for j in range(1, 4)]  # 10 questions, 3 subquestions each
    binary_data = pd.DataFrame(index=unique_ids, columns=columns)

    # Fill the DataFrame with default values (all entries marked as 1)
    binary_data.loc[:, :] = 1

    # Process the FailedLevels column to override default values
    for _, row in data.iterrows():
        account_id = row['AccountId']
        failed_levels = row['FailedLevels']
        if pd.notna(failed_levels):
            max_fail = 0  # Track the last failed column
            for entry in failed_levels.split(","):
                question, subquestion, _ = map(int, entry.split(":"))  # Ignore the third number
                column_name = f"{question}:{subquestion}"
                binary_data.loc[account_id, column_name] = 0
                max_fail = max(max_fail, columns.index(column_name))

            # Set all columns after the last fail to 0
            for col in columns[max_fail + 1:]:
                binary_data.loc[account_id, col] = 0

    # Reset the index to make AccountId a column
    binary_data.reset_index(inplace=True)
    binary_data.rename(columns={'index': 'AccountId'}, inplace=True)

    # Save the binary data to a new CSV file
    binary_data.to_csv(output_file, index=False)
    print(f"Binary data saved to {output_file}")

# Example usage
input_file = "Pyramids/raw_short.csv"
output_file = "Pyramids/binary.csv"
transform_to_binary(input_file, output_file)