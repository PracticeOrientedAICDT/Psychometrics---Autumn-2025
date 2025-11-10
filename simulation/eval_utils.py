import numpy as np
import pandas as pd
from scipy.stats import kstest, norm, skew, kurtosis

def full_normality_score(values, target_mean=None, target_std=None,
                         w_ks=1.0, w_mean=0.5, w_std=0.5,
                         w_skew=0.3, w_kurt=0.2):
    """
    Compute a composite score indicating how close a distribution is to
    a target normal distribution.

    Higher = better match.
    Lower = worse match.

    Components:
        - KS similarity
        - mean deviation
        - std deviation
        - skew penalty
        - kurtosis penalty
    """
    x = np.asarray(values, dtype=float)

    # --- Basic stats
    mu = x.mean()
    sigma = x.std()

    # --- KS goodness-of-fit to normal(mu, sigma)
    ks_stat, _ = kstest(x, 'norm', args=(mu, sigma))
    score_ks = 1 - ks_stat  # higher is better

    # --- Default targets
    if target_mean is None:
        target_mean = x.max() / 2

    if target_std is None:
        target_std = sigma

    # --- Penalties
    mean_penalty = abs(mu - target_mean)
    std_penalty  = abs(sigma - target_std)
    skew_penalty = abs(skew(x))
    kurt_penalty = abs(kurtosis(x))

    # --- Weighted combined score
    return (
        w_ks * score_ks
        - w_mean * mean_penalty
        - w_std  * std_penalty
        - w_skew * skew_penalty
        - w_kurt * kurt_penalty
    )


