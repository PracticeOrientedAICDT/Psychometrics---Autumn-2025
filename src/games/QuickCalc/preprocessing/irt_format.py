import numpy as np
import pandas as pd
from typing import Tuple


"""
Each QuestionDiffucylty is moved into a column, structure then
looks like:

participant_id  |   item_id    |    reponse     |


1097            |    1         |   1            |
1097            |    2         |   0            |

Columns with no answers are padded with 0s. 
"""

def pad_missing_questions_with_zeros(
    irt_df: pd.DataFrame,
    account_col: str = "AccountId",
    verbose = False
) -> pd.DataFrame:
    """
    Fill NaN (missing) values in an IRT-style DataFrame with 0s.
    Leaves the AccountId column untouched.

    Parameters
    ----------
    irt_df : pd.DataFrame
        Wide IRT-style DataFrame with AccountId and question difficulty columns.
    account_col : str, default 'AccountId'
        Column identifying each participant.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame where all NaNs in question columns are replaced with 0s.
    """
    if account_col not in irt_df.columns:
        raise ValueError(f"Column '{account_col}' not found in DataFrame.")

    # Identify question (difficulty) columns
    question_cols = [c for c in irt_df.columns if c != account_col]

    # Copy to avoid modifying original
    padded_df = irt_df.copy()

    # Fill NaN with 0 in all question columns
    padded_df[question_cols] = padded_df[question_cols].fillna(0)
    if verbose:
        print(f"Replaced NaN values with 0s in {len(question_cols)} question columns.")

    return padded_df


def convert(df):
    # Make a copy to avoid modifying the original DataFrame
    formatted_df = df.copy()

    # Rename columns
    formatted_df = formatted_df.rename(columns={
        'AccountId': 'participant_id',
        'QuestionWordDifficulty': 'item_id',
        'AnswerScoreBinary': 'response'
    })

    # Select only the relevant columns
    formatted_df = formatted_df[['participant_id', 'item_id', 'response']]

    # Ensure correct data types
    formatted_df['participant_id'] = formatted_df['participant_id'].astype(str)
    formatted_df['item_id'] = formatted_df['item_id'].astype(str)
    formatted_df['response'] = formatted_df['response'].astype(int)

    # Drop rows where 'response' isn't 0 or 1
    formatted_df = formatted_df[formatted_df['response'].isin([0, 1])]
    # Reset index
    formatted_df = formatted_df.reset_index(drop=True)


    return formatted_df

def create_irt_input(df):
    pad_df = pad_missing_questions_with_zeros(df)
    irt_df = convert(pad_df)
    return irt_df

