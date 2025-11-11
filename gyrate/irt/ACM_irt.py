import numpy as np
import pandas as pd

# ---------- Utilities ----------

def sigmoid(x):
    # numerically stable logistic
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1 / (1 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1 + ex)
    return out

def safe_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def total_loglik(y, theta, a, b):
    """
    Total log-likelihood (ignores NaNs).
    2PL: p_ij = σ(a_j (θ_i - b_j))
    """
    eta = np.outer(theta, a) - a * b  # shape (n_person, n_item)
    p = sigmoid(eta)
    mask = ~np.isnan(y)
    y_m = y[mask]
    p_m = p[mask]
    # avoid log(0)
    p_m = np.clip(p_m, 1e-12, 1 - 1e-12)
    return np.sum(y_m * np.log(p_m) + (1 - y_m) * np.log(1 - p_m))

# ---------- Step 0: Initialization ----------

def initialisation(df):
    """
    Initialize 2PL parameters following ACM/JMLE:
      - theta_i: standardized logit of each person's proportion correct
      - b_j:      difficulty = -logit(item proportion correct)
      - a_j:      discrimination initialized to 1
    """
    Y = df.to_numpy().astype(float)          # shape: (n_person, n_item)

    # Person proportions (ignore NaNs)
    p_i = np.nanmean(Y, axis=1)
    theta = safe_logit(p_i)
    theta = (theta - theta.mean()) / theta.std()

    # Item proportions (ignore NaNs)
    p_j = np.nanmean(Y, axis=0)
    b = -safe_logit(p_j)                     # difficulty
    a = np.ones_like(b, dtype=float)         # discrimination

    return Y, theta, a, b

# ---------- Step 1: Update abilities (θ) given (a, b) ----------

def update_theta(Y, theta, a, b, max_inner=10, tol=1e-6, step_shrink=0.5, clip=6.0):
    """
    One ACM/JMLE 'ability' update sweep:
      For each person i, maximize their log-likelihood with a few Newton steps.
      Missing responses (NaN) are ignored.
    """
    n_person, n_item = Y.shape
    a = np.asarray(a)        # (n_item,)
    b = np.asarray(b)        # (n_item,)
    theta_new = theta.copy()

    for i in range(n_person):
        yi = Y[i, :]
        mask = ~np.isnan(yi)
        if not np.any(mask):
            continue  # no data for this person

        a_m = a[mask]
        b_m = b[mask]
        y_m = yi[mask]

        th = theta_new[i]

        for _ in range(max_inner):
            # p_ij = σ(a_j (th - b_j))
            z = a_m * (th - b_m)
            p = sigmoid(z)

            # gradient and Hessian
            g = np.sum(a_m * (y_m - p))
            h = -np.sum((a_m ** 2) * p * (1 - p))

            if h >= 0 or np.isclose(h, 0.0):
                break  # pathological; skip update

            delta = -g / h
            # simple damping if step explodes
            step = 1.0
            th_try = th + step * delta
            # optional backtracking to avoid huge jumps
            while np.abs(th_try - th) > 1.0 and step > 1e-3:
                step *= step_shrink
                th_try = th + step * delta

            if np.abs(th_try - th) < tol:
                th = th_try
                break
            th = th_try

        # clip to keep numeric stability
        theta_new[i] = np.clip(th, -clip, clip)

    return theta_new


# ---------- Step 2: Re-identification / scaling ----------

def rescale_theta_and_items(theta, a, b):
    """
    Standardize theta to mean 0, sd 1, and transform items to keep likelihood invariant:
      θ' = (θ - m)/s
      a' = a * s
      b' = (b + m)/s
    """
    m = theta.mean()
    s = theta.std()
    if s <= 0:
        return theta.copy(), a.copy(), b.copy()
    theta_p = (theta - m) / s
    a_p = a * s
    b_p = (b + m) / s
    return theta_p, a_p, b_p

import numpy as np

# ---- Step 3: Update items (a, b) given theta ----

def update_items(Y, theta, a, b, max_inner=25, tol=1e-6, l2=1e-6, clip_beta=12.0):
    """
    Per-item logistic regressions with θ as the sole predictor.
    Missing responses (NaN) are ignored.

    Parameters
    ----------
    Y : array (n_person, n_item) with 0/1/NaN
    theta : array (n_person,)
    a, b : current item params (used only for warm starts)
    max_inner : Newton iterations per item
    tol : convergence tolerance on the beta step
    l2 : small ridge penalty for stability
    clip_beta : bounds to keep betas numerically safe

    Returns
    -------
    a_new : array (n_item,)
    b_new : array (n_item,)
    """
    n_person, n_item = Y.shape
    theta = np.asarray(theta, dtype=float)

    a_new = np.asarray(a, dtype=float).copy()
    b_new = np.asarray(b, dtype=float).copy()

    for j in range(n_item):
        yj = Y[:, j]
        mask = ~np.isnan(yj)
        if not np.any(mask):
            # no data for this item; keep previous params
            continue

        y = yj[mask]
        th = theta[mask]

        # --- warm start from current (a,b) ---
        beta1 = a_new[j]
        if not np.isfinite(beta1) or beta1 == 0.0:
            beta1 = 1.0
        beta0 = -beta1 * b_new[j]
        if not np.isfinite(beta0):
            beta0 = 0.0

        # --- Newton–Raphson for 2-parameter logistic regression ---
        for _ in range(max_inner):
            eta = beta0 + beta1 * th
            # numerically stable σ
            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -40, 40)))
            w = p * (1.0 - p)                      # weights

            # Gradients (with L2 ridge)
            g0 = np.sum(y - p) - l2 * beta0
            g1 = np.sum((y - p) * th) - l2 * beta1

            # Hessian (negative-definite)
            H00 = -np.sum(w) - l2
            H01 = -np.sum(w * th)
            H11 = -np.sum(w * th * th) - l2

            H = np.array([[H00, H01],
                          [H01, H11]], dtype=float)
            g = np.array([g0, g1], dtype=float)

            # Solve H * delta = -g
            try:
                delta = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                # fall back to a tiny step in the gradient direction
                delta = 0.1 * g

            beta0 += delta[0]
            beta1 += delta[1]

            # keep betas in a safe numeric range
            beta0 = float(np.clip(beta0, -clip_beta, clip_beta))
            beta1 = float(np.clip(beta1, -clip_beta, clip_beta))

            if np.linalg.norm(delta, ord=2) < tol:
                break

        # --- map back to 2PL ---
        if np.isclose(beta1, 0.0):
            # if slope ~0, make a tiny slope and push difficulty far away
            beta1 = np.sign(beta1) * 1e-6 if beta1 != 0 else 1e-6

        a_new[j] = beta1                      # allow negative during iteration
        b_new[j] = -beta0 / beta1

    return a_new, b_new


def fit_2pl(
    df,
    max_iters=100,
    tol=1e-4,                 # stop if relative LL improvement < tol
    theta_inner=10,           # Newton steps per person
    item_inner=25,            # Newton steps per item
    l2=1e-6,                  # small ridge in item updates
    verbose=True,
):
    """
    Alternating Conditional Maximization (ACM/JMLE) for 2PL.

    Steps per outer iteration:
      1) update_theta
      2) rescale_theta_and_items
      3) update_items
    Converges when relative LL improvement falls below `tol`.
    """
    # --- Step 0: init
    Y, theta, a, b = initialisation(df)
    ll = total_loglik(Y, theta, a, b)
    history = [ll]

    if verbose:
        print(f"[init]   loglik = {ll:.6f}")

    for it in range(1, max_iters + 1):
        # 1) abilities
        theta = update_theta(
            Y, theta, a, b,
            max_inner=theta_inner, tol=1e-6
        )

        # 2) re-identify (standardize theta; adjust items)
        theta, a, b = rescale_theta_and_items(theta, a, b)

        # 3) items
        a, b = update_items(
            Y, theta, a, b,
            max_inner=item_inner, tol=1e-6, l2=l2
        )

        # evaluate
        ll_new = total_loglik(Y, theta, a, b)
        history.append(ll_new)

        rel_imp = (ll_new - ll) / (abs(ll) + 1e-12)
        if verbose:
            print(f"[iter {it:02d}] loglik = {ll_new:.6f}  Δrel = {rel_imp:.3e}")

        if rel_imp < tol:
            if verbose:
                print(f"Converged at iter {it} (Δrel={rel_imp:.3e} < {tol}).")
            break
        ll = ll_new

    out = {
        "theta": theta,       # (n_person,)
        "a": a,               # (n_item,)
        "b": b,               # (n_item,)
        "loglik": ll_new,
        "history": np.array(history),
        "n_iter": it,
        "converged": rel_imp < tol
    }
    return out

# ---------- (optional) helpers ----------

def irt_prob(theta, a, b):
    """
    Return matrix of probabilities p_ij = σ(a_j (θ_i - b_j))
    theta: (n_person,), a: (n_item,), b: (n_item,)
    """
    eta = np.outer(theta, a) - a * b
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -40, 40)))