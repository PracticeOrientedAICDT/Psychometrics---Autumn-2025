import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from typing import Iterable, Optional, Dict, Tuple, List


class MechanicsRegressor:
    """
    Multi-output regression mapping feature_cols -> mechanics.

    - By default uses feature_cols = ["a", "b"], but you can pass ["b"] or any
      numeric subset of columns.
    - Uses a polynomial regression model with Ridge regularisation to learn a
      smooth global mapping f(a, b) -> mechanics.
    - Works better than KNN for extrapolation when new (a, b) fall outside the
      range seen in the training data.
    - Can optionally:
        * round selected mechanics to integers (int_mech_cols)
        * clip mechanics to specified ranges (clip_config)
    """

    def __init__(
        self,
        degree: int = 2,
        alpha: float = 1.0,
        feature_cols: Optional[Iterable[str]] = None,
        int_mech_cols: Optional[Iterable[str]] = None,
        clip_config: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        Parameters
        ----------
        degree : int, default 2
            Polynomial degree for the regression surface (1 = linear, 2 = quadratic).
        alpha : float, default 1.0
            Ridge regularisation strength (higher = smoother, less flexible).
        feature_cols : Iterable[str], optional
            Names of feature columns, e.g. ["a", "b"] or just ["b"].
        int_mech_cols : Iterable[str], optional
            Mechanics columns to round to integers after prediction.
        clip_config : dict, optional
            Dict mapping column name -> (min, max) to clip mechanics into ranges.
        """
        # which columns to use as inputs
        if feature_cols is None:
            feature_cols = ["a", "b"]
        self.feature_cols_: List[str] = list(feature_cols)

        # polynomial + ridge model wrapped in MultiOutputRegressor
        base_model = make_pipeline(
            PolynomialFeatures(degree=degree),
            Ridge(alpha=alpha)
        )
        self.model = MultiOutputRegressor(base_model)

        self.trained_cols_: Optional[List[str]] = None  # mechanics cols actually used
        self.int_mech_cols_ = list(int_mech_cols) if int_mech_cols is not None else []
        self.clip_config_ = clip_config if clip_config is not None else {}

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame, mech_cols: Optional[Iterable[str]] = None):
        """
        Fit the regression model on a DataFrame with columns feature_cols_ + mechanics.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing item parameters and mechanics.
        mech_cols : Iterable[str], optional
            Names of mechanics columns to predict. If None, use all columns
            except feature_cols_.
        """
        # Ensure feature columns exist
        missing_feats = [c for c in self.feature_cols_ if c not in df.columns]
        if missing_feats:
            raise ValueError(f"Missing feature columns in df: {missing_feats}")

        if mech_cols is None:
            mech_cols = [c for c in df.columns if c not in self.feature_cols_]
            if not mech_cols:
                raise ValueError(
                    f"No mechanic columns found (cols other than {self.feature_cols_})."
                )

        mech_cols = list(mech_cols)

        # Keep only rows that have all features + all mechanics present (no NaNs)
        cols_needed = self.feature_cols_ + mech_cols
        ok = df[cols_needed].dropna(how="any")
        if ok.empty:
            raise ValueError("No complete rows with features and mechanics to fit on.")

        X = ok[self.feature_cols_].to_numpy(dtype=float)
        y = ok[mech_cols].to_numpy(dtype=float)

        self.model.fit(X, y)
        self.trained_cols_ = mech_cols
        return self

    # ------------------------------------------------------------------ #
    # Predict
    # ------------------------------------------------------------------ #
    def predict(self, df_with_feats: pd.DataFrame) -> pd.DataFrame:
        """
        Predict mechanics for rows containing the feature columns (e.g. 'a','b' or just 'b').

        Parameters
        ----------
        df_with_feats : pd.DataFrame
            Must contain all feature_cols_ used at fit time.

        Returns
        -------
        pd.DataFrame
            DataFrame with one column per trained mechanic.
        """
        if self.trained_cols_ is None:
            raise RuntimeError("Call fit(...) before predict(...).")

        missing_feats = [c for c in self.feature_cols_ if c not in df_with_feats.columns]
        if missing_feats:
            raise ValueError(f"df_with_feats is missing feature columns: {missing_feats}")

        X = df_with_feats[self.feature_cols_].to_numpy(dtype=float)
        pred = self.model.predict(X)

        out = pd.DataFrame(pred, columns=self.trained_cols_, index=df_with_feats.index)

        # Round integer mechanics
        for c in self.int_mech_cols_:
            if c in out.columns:
                out[c] = np.rint(out[c]).astype(int)

        # Apply clipping rules
        for col, (lo, hi) in self.clip_config_.items():
            if col in out.columns:
                out[col] = out[col].clip(lower=lo, upper=hi)

        return out
