import pandas as pd
import numpy as np

# Function to transform the short raw file into the desired format
def transform_short_raw(input_file, output_file):
    # Read the input CSV file
    data = pd.read_csv(input_file)

    # Get unique AccountIDs
    unique_ids = data['AccountId'].unique()

    # Create an empty DataFrame for the output
    columns = [f"{i}:{j}" for i in range(1, 11) for j in range(1, 4)]  # 10 questions, 3 subquestions each
    transformed_data = pd.DataFrame(index=unique_ids, columns=columns)

    # Fill the DataFrame with default values (question numbers)
    for i in range(1, 11):
        for j in range(1, 3 + 1):
            transformed_data[f"{i}:{j}"] = i

    # Process the FailedLevels column to override default values
    for _, row in data.iterrows():
        account_id = row['AccountId']
        failed_levels = row['FailedLevels']
        if pd.notna(failed_levels):
            max_fail = 0  # Track the last failed column
            for entry in failed_levels.split(","):
                question, subquestion, value = map(int, entry.split(":"))
                column_name = f"{question}:{subquestion}"
                transformed_data.loc[account_id, column_name] = value
                max_fail = max(max_fail, columns.index(column_name))

            # Set all columns after the last fail to 0
            for col in columns[max_fail + 1:]:
                transformed_data.loc[account_id, col] = 0

    # Reset the index to make AccountId a column
    transformed_data.reset_index(inplace=True)
    transformed_data.rename(columns={'index': 'AccountId'}, inplace=True)

    # Save the transformed data to a new CSV file
    transformed_data.to_csv(output_file, index=False)
    print(f"Transformed data saved to {output_file}")

    # Generate binary file
    binary_data = transformed_data.copy()
    for col in columns:
        question, subquestion = map(int, col.split(":"))
        binary_data[col] = (transformed_data[col] == question).astype(int)

    binary_output_file = output_file.replace("numerical", "binary")
    binary_data.to_csv(binary_output_file, index=False)
    print(f"Binary data saved to {binary_output_file}")

# Example usage
input_file = "EyeBall_short_raw.csv"
output_file = "EyeBall_numerical.csv"
transform_short_raw(input_file, output_file)