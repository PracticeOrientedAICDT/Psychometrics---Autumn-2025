import numpy as np
import pandas as pd
from typing import Optional,Tuple

def generate_abilities_df(
    n_participants: int,
    mean: float = 0.0,
    std: float = 1.0,
    seed: Optional[int] = None,
    id_prefix: str = "S"
) -> pd.DataFrame:
    """
    Create a synthetic abilities dataframe with columns ['participant_id','theta'].
    IDs are strings like S1, S2, ...
    """
    rng = np.random.default_rng(seed)
    thetas = rng.normal(loc=mean, scale=std, size=n_participants)
    participant_ids = [f"{id_prefix}{i+1}" for i in range(n_participants)]
    return pd.DataFrame({"participant_id": participant_ids, "theta": thetas})

def get_simulated_scores(abilities_df: Optional[pd.DataFrame],
                         item_params_df,
                         num_lives: int,
                         num_sub_levels: int,
                         item_scoring_df = None,
                         seed: Optional[int] = None,
      
                        rand_sl:bool = False,
                        rand_sl_range: Tuple[Optional[int], Optional[int]] = (None, None),
                        
                        # NEW controls for synthetic abilities when use_abilities_df=False
                        use_abilities_df: bool = True,
                        n_synthetic_participants: int = 1000,
                        ability_mean: float = 0.0,
                        ability_std: float = 1.0,
                        ability_id_prefix: str = "S",
                        ):

    """
    abilities_df: columns ['participant_id','theta']
    item_params_df: columns ['item_id','a','b','c']
    item_scoring_df: DataFrame with columns ['item_id', 'item_score']
    
    returns 
    simulated_results_df: columns ['AccountId','Score']

    """
    rng = np.random.default_rng(seed)

    simulated_results = []

    # pick abilities source
    if use_abilities_df:
        if abilities_df is None or abilities_df.empty:
            raise ValueError("use_abilities_df=True but abilities_df is None/empty.")
        abilities_use = abilities_df[["participant_id", "theta"]].copy()
    else:
        abilities_use = generate_abilities_df(
            n_participants=n_synthetic_participants,
            mean=ability_mean,
            std=ability_std,
            seed=seed,
            id_prefix=ability_id_prefix,
        )

    # --- Resolve random sub-level range ---
    lo_raw, hi_raw = (list(rand_sl_range) + [None, None])[:2]
    lo = 0 if lo_raw is None else int(max(0, lo_raw))
    hi = 4 if hi_raw is None else int(max(lo, hi_raw))  # ensure hi >= lo

    # ensure item params have the needed columns
    needed = {"item_id", "a", "b"}
    missing = needed - set(item_params_df.columns)
    if missing:
        raise ValueError(f"item_params_df missing columns: {missing}")

    # normalise optional 'c'
    has_c = "c" in item_params_df.columns

    simulated_results = []

    for _, person in abilities_use.iterrows():
        participant_id = person["participant_id"]  # keep as string; do not cast to int
        theta = float(person["theta"])

        remaining_lives = num_lives
        total_score = 0

        # iterate over items
        for _, row in item_params_df.iterrows():
            # choose sub-level count
            if rand_sl:
                n_sub = int(rng.integers(lo, hi + 1))
                #print(f"Level {_}: n_sub_levels = {n_sub}")
            else:
                n_sub = int(num_sub_levels)

            item_id = str(row["item_id"])
            a = float(row["a"])
            b = float(row["b"])
            c = float(row["c"]) if has_c and pd.notna(row["c"]) else 0.0

            # iterate sub-levels
            for _ in range(n_sub):
                is_correct = False
                # retry until success or out of lives
                while (not is_correct) and (remaining_lives > 0):
                    is_correct = simulate_question(theta, a, b, c, rng)

                    if not is_correct:
                        score = 0
                        remaining_lives -= 1
                    else:
                        score = (
                            get_question_score(item_id, item_scoring_df)
                            if item_scoring_df is not None
                            else 1
                        )

                    total_score += score
                    if remaining_lives <= 0:
                        break

            if remaining_lives <= 0:
                break

        simulated_results.append({"AccountId": participant_id, "Score": total_score})

    return pd.DataFrame(simulated_results)
          
def simulate_question(theta, a, b, c, rng):
    p = c + (1.0 - c) / (1.0 + np.exp(-a * (theta - b)))
    return rng.random() < p

def get_question_score(item_id: int, item_scoring_df: Optional[pd.DataFrame]) -> int:
    """
    Return the score for a given item_id using item_scoring_df.

    If item_scoring_df is None, or the item_id is not found,
    defaults to a score of 1.
    """
    if item_scoring_df is None or item_scoring_df.empty:
        return 1

    # normalise dtypes
    try:
        df = item_scoring_df.copy()
        df["item_id"] = df["item_id"].astype(int)
    except Exception:
        return 1

    row = df.loc[df["item_id"] == item_id, "item_score"]

    if row.empty:
        return 1

    score = row.iloc[0]
    if pd.isna(score):
        return 1

    return int(score)

def generate_simulated_irt_matrix(
    abilities_df: pd.DataFrame,
    item_params_df: pd.DataFrame,
    account_col: str = "AccountId",
    theta_col: str = "theta",
    item_id_col: str = "item_id",   # or "QuestionID"
    a_col: str = "a",
    b_col: str = "b",
    c_col: str = "c",
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate a simulated item response matrix using IRT parameters and abilities.

    Parameters
    ----------
    abilities_df : DataFrame
        Must contain [account_col, theta_col].
    item_params_df : DataFrame
        Must contain [item_id_col, a_col, b_col]. c_col is optional; if missing, assumes 0.
    account_col : str
        Column name for person identifier.
    theta_col : str
        Column name for ability θ.
    item_id_col : str
        Column name for item identifier.
    a_col, b_col, c_col : str
        Column names for IRT parameters.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    irt_matrix : DataFrame
        Rows = people (account_col), columns = item IDs, values = 0/1 responses.
    """

    rng = np.random.default_rng(seed)

    # Ensure c-column
    items_df = item_params_df.copy()
    if c_col not in items_df.columns:
        items_df[c_col] = 0.0

    # Extract arrays
    people = abilities_df[account_col].to_numpy()
    thetas = abilities_df[theta_col].to_numpy()

    item_ids = items_df[item_id_col].to_numpy()
    a_vals = items_df[a_col].to_numpy()
    b_vals = items_df[b_col].to_numpy()
    c_vals = items_df[c_col].to_numpy()

    n_people = len(people)
    n_items = len(item_ids)

    # Response matrix (people × items)
    resp = np.zeros((n_people, n_items), dtype=int)

    for i in range(n_people):
        theta = thetas[i]
        for j in range(n_items):
            u = simulate_question(theta, a_vals[j], b_vals[j], c_vals[j], rng)
            resp[i, j] = int(u)

    # Build DataFrame: first column = account id, rest = items
    irt_matrix = pd.DataFrame(resp, columns=item_ids)
    irt_matrix.insert(0, account_col, people)

    return irt_matrix

def add_score_column(item_response_df, score_col_name="Score"):
    """
    Add a score column to an item-response matrix by counting the number
    of correct responses (1s) per participant.

    Parameters
    ----------
    item_response_df : pd.DataFrame
        Rows = participants  
        Columns = item responses (binary 0/1). 
        Non-item columns (like participant_id) are preserved.
    
    score_col_name : str
        Name of the new score column. Default = "Score".

    Returns
    -------
    pd.DataFrame
        A new DataFrame with an added score column.
    """

    df = item_response_df.copy()

    # Identify item columns = numeric columns with only 0/1 values
    item_cols = [
        col for col in df.columns
        if df[col].dropna().isin([0, 1]).all()
    ]

    # Compute row-wise sum of correct answers
    df[score_col_name] = df[item_cols].sum(axis=1)

    return df
