import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from typing import Iterable, Optional, Dict, Tuple, List


class MechanicsPredictor:
    """
    KNN multi-output regressor mapping (a, b) -> mechanics.

    - Trains on rows where a, b, and all target mechanics columns are present.
    - By default learns all non-['a','b'] columns as mechanics, or you can
      explicitly pass `mech_cols`.
    - Can optionally:
        * round selected mechanics to integers (int_mech_cols)
        * clip mechanics to specified ranges (clip_config)

    Parameters
    ----------
    k : int
        Number of neighbours for KNN.
    weights : str
        Weighting scheme for KNN ('uniform' or 'distance').
    int_mech_cols : Iterable[str], optional
        Columns that should be rounded to ints on prediction.
    clip_config : Dict[str, Tuple[float, float]], optional
        Mapping col_name -> (min, max) for clipping predictions.
    """

    def __init__(
        self,
        k: int = 5,
        weights: str = "distance",
        int_mech_cols: Optional[Iterable[str]] = None,
        clip_config: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.k = k
        self.weights = weights
        self.model = MultiOutputRegressor(
            KNeighborsRegressor(n_neighbors=k, weights=weights)
        )

        self.trained_cols_: Optional[List[str]] = None   # mechanics cols actually used
        self.int_mech_cols_ = list(int_mech_cols) if int_mech_cols is not None else []
        self.clip_config_ = clip_config if clip_config is not None else {}

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame, mech_cols: Optional[Iterable[str]] = None):
        """
        Fit the KNN model on a DataFrame with columns ['a','b', ...mechanics...]

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing item parameters and mechanics.
        mech_cols : Iterable[str], optional
            Names of mechanics columns to predict. If None, use all columns
            except 'a' and 'b'.

        Returns
        -------
        self
        """
        if mech_cols is None:
            # Infer mechanics as "everything except a,b"
            mech_cols = [c for c in df.columns if c not in ("a", "b")]
            if not mech_cols:
                raise ValueError("No mechanic columns found (cols other than 'a','b').")

        mech_cols = list(mech_cols)

        # Keep only rows that have a, b, and all mechanics present (no NaNs)
        cols_needed = ["a", "b"] + mech_cols
        ok = df[cols_needed].dropna(how="any")
        if ok.empty:
            raise ValueError("No complete rows with a,b and mechanics to fit on.")

        X = ok[["a", "b"]].to_numpy(dtype=float)
        y = ok[mech_cols].to_numpy(dtype=float)

        self.model.fit(X, y)
        self.trained_cols_ = mech_cols
        return self

    # ------------------------------------------------------------------ #
    # Predict
    # ------------------------------------------------------------------ #
    def predict(self, df_with_ab: pd.DataFrame) -> pd.DataFrame:
        """
        Predict mechanics for rows with 'a' and 'b' columns.

        Parameters
        ----------
        df_with_ab : pd.DataFrame
            Must contain columns 'a' and 'b'.

        Returns
        -------
        pd.DataFrame
            DataFrame with one column per trained mechanic.
        """
        if self.trained_cols_ is None:
            raise RuntimeError("Call fit(...) before predict(...).")

        if not {"a", "b"}.issubset(df_with_ab.columns):
            raise ValueError("df_with_ab must contain columns 'a' and 'b'.")

        X = df_with_ab[["a", "b"]].to_numpy(dtype=float)
        pred = self.model.predict(X)

        out = pd.DataFrame(pred, columns=self.trained_cols_, index=df_with_ab.index)

        # Round integer mechanics
        for c in self.int_mech_cols_:
            if c in out.columns:
                out[c] = np.rint(out[c]).astype(int)

        # Apply clipping rules
        for col, (lo, hi) in self.clip_config_.items():
            if col in out.columns:
                out[col] = out[col].clip(lower=lo, upper=hi)

        return out
