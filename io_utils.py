import pandas as pd
import os
import json

#IMPORT DATA
def load_csv_into_df(file_path: str) -> pd.DataFrame:
    """
    Load a pandas DataFrame from a CSV file.

    Parameters
    ----------
    file_path : str
        Full path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")

    df = pd.read_csv(file_path)
    print(f"✅ Loaded DataFrame from: {file_path} (shape: {df.shape})")
    return df


#EXPORT DATA
def save_df_as_csv(df: pd.DataFrame, file_path: str, index: bool = False):
    """
    Save a pandas DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save.
    file_path : str
        Full path (including filename) where the CSV will be saved.
    index : bool, default False
        Whether to include the DataFrame index as a column in the CSV.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save CSV
    df.to_csv(file_path, index=index)
    print(f"✅ DataFrame saved to: {file_path} (shape: {df.shape})")
