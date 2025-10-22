
from typing import Tuple
import pandas as pd
import numpy as np
from girth import onepl_mml, twopl_mml, threepl_mml,ability_eap
from typing import Optional

def simulate_irt_scores(
    abilities_df: pd.DataFrame,
    item_latents_df: pd.DataFrame,
    account_col: str = "participant_id",
    ability_col: str = "theta",
    a_col: str = "a",
    b_col: str = "b",
    c_col: Optional[str] = "c",
) -> pd.DataFrame:
    """
    Compute expected total score per participant using 3PL form:
        P = c + (1 - c) / (1 + exp[-a*(theta - b)])
    Falls back to 2PL if `c` missing, Rasch if `a` missing.
    """
    import numpy as np
    import pandas as pd

    if account_col not in abilities_df or ability_col not in abilities_df:
        raise ValueError(f"abilities_df must have columns [{account_col}, {ability_col}]")
    if b_col not in item_latents_df:
        raise ValueError(f"item_latents_df must include '{b_col}'")
    
    items = item_latents_df.copy()
    if a_col not in items:
        items[a_col] = 1.0
    if c_col is None or c_col not in items:
        c_col = "__c__"
        items[c_col] = 0.0

    thetas = abilities_df[ability_col].to_numpy(dtype=float)
    a = items[a_col].to_numpy(dtype=float)
    b = items[b_col].to_numpy(dtype=float)
    c = items[c_col].to_numpy(dtype=float)

    P = c[None, :] + (1.0 - c[None, :]) / (1.0 + np.exp(-a[None, :] * (thetas[:, None] - b[None, :])))
    expected_scores = P.sum(axis=1)

    return pd.DataFrame({
        account_col: abilities_df[account_col].values,
        "Score": expected_scores
    })


'''
TRYING TO USE GIRTH
'''
def prepare_girth_input(df_long: pd.DataFrame):
    """
    Input (long): columns = ['participant_id','item_id','response'] with 0/1.
    Output:
      X_tag   : np.ndarray (participants x items), with GIRTH-missing tags
      row_ids : list[str] participant ids (row order)
      col_ids : list[str] item ids (col order)
    """

    df = df_long[['participant_id','item_id','response']].copy()


    """
    Turn Particpand Ids and question Ids into strings,  
    """
    df['participant_id'] = df['participant_id'].astype(str)
    df['item_id'] = df['item_id'].astype(float).astype(int).astype(str)

    df['response'] = df['response'].astype(int)

    # Pivot; collapse accidental duplicates deterministically
    mat = df.pivot_table(index='participant_id', columns='item_id',
                         values='response', aggfunc='max')

    # Your rule: treat all missing as 0 (incorrect) and ensure integer dtype
    mat = mat.fillna(0)
    X = mat.to_numpy(dtype=np.int64)

    row_ids = mat.index.astype(str).tolist()
    col_ids = mat.columns.astype(str).tolist()

    return X, row_ids, col_ids

def fit_with_girth(irt_df, model_spec="1PL", epochs=10, quadrature_n=21):
    """
    Fits an IRT model (1PL or 2PL) using GIRTH.
    Returns (persons_df, items_df).
    """
    print("Preparing girth data...")
    X_tag, row_ids, col_ids = prepare_girth_input(irt_df)

    #DEBUG
    #print(f"X_tag shape: {X_tag.shape}")
    #print("Unique values:", np.unique(X_tag)[:10])
    #print("Valid count:", np.count_nonzero(X_tag != -1))

    # Transpose to items × participants
    X_ip = X_tag.T
    n_items = X_ip.shape[0]

    ms = model_spec.upper().strip()
    print(f"Estimating {ms} parameters...")

    if ms == "1PL":
        est = onepl_mml(
            X_ip,
            options={"max_iteration": epochs, "quadrature_n": quadrature_n}
        )
        a = np.ones(n_items)
        b = est["Difficulty"]
        c = None  # no guessing in 1PL

    elif ms == "2PL":
        est = twopl_mml(
            X_ip,
            options={"max_iteration": epochs, "quadrature_n": quadrature_n}
        )
        a = est["Discrimination"]
        b = est["Difficulty"]
        c = None  # no guessing in 2PL

    elif ms == "3PL":
        est = threepl_mml(
            X_ip,
            options={"max_iteration": epochs, "quadrature_n": quadrature_n}
        )
        a = est["Discrimination"]
        b = est["Difficulty"]
        c = est["Guessing"]  # lower asymptote
        print(
            "⚠︎ 3PL selected: GIRTH's ability_eap doesn’t use guessing (c). "
            "Abilities will be EAP under a 2PL likelihood."
        )


    else:
        raise ValueError("model_spec must be '1PL', '2PL', or '3PL'")


    print("Estimating ability (EAP)...")
    # ability_eap accepts guessing=None for 1PL/2PL, or a vector for 3PL
    theta = ability_eap(
            X_ip,
            difficulty=b,
            discrimination=a,
            options={"quadrature_n": quadrature_n}
    )

    print("Done.")

    persons_df = pd.DataFrame({"participant_id": row_ids, "theta": theta})
    items_df = pd.DataFrame({"item_id": col_ids, "a": a, "b": b})
    if c is not None:
        items_df["c"] = c

    return persons_df, items_df
