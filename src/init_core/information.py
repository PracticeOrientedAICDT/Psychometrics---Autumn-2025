#!/usr/bin/env python3

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


def read_item_params(csv_path):
    """
    Reads a CSV file with columns: item_id, a, b, c (c ignored for 2PL).
    Returns arrays of a, b.
    """
    df = pd.read_csv(csv_path)
    # Allow for slight variations in column names (case-insensitive)
    df.columns = [c.lower() for c in df.columns]
    if not {"a", "b"}.issubset(df.columns):
        raise ValueError("Input file must contain at least columns 'a' and 'b'.")
    a = df["a"].values.astype(float)
    b = df["b"].values.astype(float)
    return a, b


def irt_probability_2pl(theta, a, b):
    """
    2PL model probability with D = 1:
        P(theta) = 1 / (1 + exp(-a * (theta - b)))
    theta: array of ability values
    a, b: item parameters (can be scalars or arrays broadcastable to theta)
    """
    z = a * (theta - b)
    return 1.0 / (1.0 + np.exp(-z))


def item_information_2pl(theta, a, b):
    """
    Item information for 2PL with D = 1:
        I(theta) = a^2 * P(theta) * (1 - P(theta))
    """
    P = irt_probability_2pl(theta, a, b)
    return (a ** 2) * P * (1.0 - P)


def compute_marginal_reliability(theta_grid, test_info, theta_mean=0.0, theta_sd=1.0):
    """
    Computes marginal reliability:
        MR = 1 - E[SEM(theta)^2] / Var(theta)
           = 1 - E[1 / I(theta)] / Var(theta)

    where expectation is over the assumed theta distribution, here
    Normal(mean = theta_mean, sd = theta_sd).

    theta_grid: array of theta values
    test_info: array of test information values at theta_grid
    theta_mean: mean of theta distribution
    theta_sd: standard deviation of theta distribution
    """
    theta_var = theta_sd ** 2

    # Avoid division by zero or extremely tiny information
    eps = 1e-10
    safe_info = np.maximum(test_info, eps)

    sem2 = 1.0 / safe_info  # SEM^2

    # Normal pdf for the assumed theta distribution
    pdf = norm.pdf(theta_grid, loc=theta_mean, scale=theta_sd)

    # Numerical integration using trapezoidal rule
    expected_sem2 = np.trapz(sem2 * pdf, theta_grid)

    marginal_reliability = 1.0 - expected_sem2 / theta_var
    return marginal_reliability


def read_theta_file(theta_path):
    """
    Reads a CSV file with columns: participant_id, theta.
    Returns an array of theta values.
    """
    df = pd.read_csv(theta_path)
    df.columns = [c.lower() for c in df.columns]
    if "theta" not in df.columns:
        raise ValueError("Theta file must contain a 'theta' column.")
    theta_vals = df["theta"].values.astype(float)
    return theta_vals


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate ICC, item information, test information, SEM plots "
            "for a 2PL IRT item bank, and print marginal reliability. "
            "Optionally, plot a histogram of IRT theta estimates."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to CSV file with item parameters (columns: item_id, a, b, c).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Directory where the plots will be saved.",
    )
    parser.add_argument(
        "--theta-min",
        type=float,
        default=-3.0,
        help="Minimum theta value for plots and integration.",
    )
    parser.add_argument(
        "--theta-max",
        type=float,
        default=3.0,
        help="Maximum theta value for plots and integration.",
    )
    parser.add_argument(
        "--theta-step",
        type=float,
        default=0.01,
        help="Step size for theta grid (default: 0.01).",
    )
    parser.add_argument(
        "--theta-mean",
        type=float,
        default=0.0,
        help="Mean of the assumed theta distribution for marginal reliability (default: 0.0).",
    )
    parser.add_argument(
        "--theta-sd",
        type=float,
        default=1.0,
        help="Standard deviation of the assumed theta distribution for marginal reliability (default: 1.0).",
    )
    parser.add_argument(
        "--abilities",
        type=str,
        default=None,
        help=(
            "Optional path to CSV file containing IRT theta estimates "
            "(e.g., columns: participant_id, theta). If provided, "
            "a histogram of theta values will be saved as an extra plot."
        ),
    )

    args = parser.parse_args()

    input_path = args.input
    output_dir = args.output_dir
    theta_min = args.theta_min
    theta_max = args.theta_max
    theta_step = args.theta_step
    theta_mean = args.theta_mean
    theta_sd = args.theta_sd
    theta_file = args.abilities

    os.makedirs(output_dir, exist_ok=True)

    # Read item parameters
    a, b = read_item_params(input_path)

    # Theta grid
    theta = np.arange(theta_min, theta_max + theta_step, theta_step)

    # 1. ICC curves (all items)
    plt.figure(figsize=(8, 6))
    for ai, bi in zip(a, b):
        P = irt_probability_2pl(theta, ai, bi)
        plt.plot(theta, P, alpha=0.7)
    plt.xlabel(r"$\theta$")
    plt.ylabel("P(correct)")
    plt.title("Item Characteristic Curves (2PL)")
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.3)
    icc_path = os.path.join(output_dir, "icc_curves.png")
    plt.savefig(icc_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Item information curves (overlaid)
    plt.figure(figsize=(8, 6))
    for ai, bi in zip(a, b):
        info_i = item_information_2pl(theta, ai, bi)
        plt.plot(theta, info_i, alpha=0.7)
    plt.xlabel(r"$\theta$")
    plt.ylabel("Information")
    plt.title("Item Information Curves (2PL)")
    plt.grid(True, linestyle="--", alpha=0.3)
    item_info_path = os.path.join(output_dir, "item_information_curves.png")
    plt.savefig(item_info_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Total information curve
    item_info_matrix = np.array(
        [item_information_2pl(theta, ai, bi) for ai, bi in zip(a, b)]
    )
    test_info = item_info_matrix.sum(axis=0)

    plt.figure(figsize=(8, 6))
    plt.plot(theta, test_info)
    plt.xlabel(r"$\theta$")
    plt.ylabel("Total Information")
    plt.title("Test Information Curve (2PL)")
    plt.grid(True, linestyle="--", alpha=0.3)
    test_info_path = os.path.join(output_dir, "test_information_curve.png")
    plt.savefig(test_info_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 4. SEM curve (standard error of measurement)
    eps = 1e-10
    safe_test_info = np.maximum(test_info, eps)
    sem = 1.0 / np.sqrt(safe_test_info)

    plt.figure(figsize=(8, 6))
    plt.plot(theta, sem)
    plt.xlabel(r"$\theta$")
    plt.ylabel("SEM")
    plt.title("Standard Error of Measurement Curve (2PL,)")
    plt.grid(True, linestyle="--", alpha=0.3)
    sem_path = os.path.join(output_dir, "sem_curve.png")
    plt.savefig(sem_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Marginal reliability using assumed theta distribution
    marginal_reliability = compute_marginal_reliability(
        theta,
        test_info,
        theta_mean=theta_mean,
        theta_sd=theta_sd,
    )
    print(
        f"Marginal reliability (2PL, assumed N({theta_mean}, {theta_sd**2})): "
        f"{marginal_reliability:.4f}"
    )

    # 5. Optional: plot histogram of IRT theta estimates (no further calculations)
    if theta_file is not None:
        theta_vals = read_theta_file(theta_file)

        plt.figure(figsize=(8, 6))
        # Sensible choice for large N: histogram with a moderate number of bins
        plt.hist(theta_vals, bins=50, edgecolor="black", alpha=0.7)
        plt.xlabel(r"Estimated $\theta$")
        plt.ylabel("Count")
        plt.title("Histogram of IRT Theta Estimates")
        plt.grid(True, linestyle="--", alpha=0.3)
        theta_hist_path = os.path.join(output_dir, "theta_histogram.png")
        plt.savefig(theta_hist_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        theta_hist_path = None

    # Print where plots were saved
    print("\nPlots saved to:")
    print(f"  ICC curves:                {icc_path}")
    print(f"  Item information curves:   {item_info_path}")
    print(f"  Test information curve:    {test_info_path}")
    print(f"  SEM curve:                 {sem_path}")
    if theta_hist_path is not None:
        print(f"  Theta histogram:           {theta_hist_path}")


if __name__ == "__main__":
    main()
