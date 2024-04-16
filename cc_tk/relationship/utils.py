"""Utility functions to perform caracterisation."""

import numpy as np
import pandas as pd


def cut_influence(influence_values: pd.Series) -> pd.Series:
    """Cut influence values into categories.

    Influence values define the impact of a given feature on another one, this
    function helps categorizing these influences into 6 categories:
    - `--`: Strong negative influence.
    - `-`: Weak negative influence.
    - `` or ` `: No influence.
    - `+`: Weak positive influence.
    - `++`: Strong positive influence.

    Parameters
    ----------
    influence_values : pd.Series
        Influence values to cut.

    Returns
    -------
    pd.Series
        Categorical influence values.

    """
    # cut_values are used to determine the thresholds between categories
    cut_values = influence_values.quantile(np.linspace(0, 1, 7))

    # duplicate_threshold is used to determine if two cut_values are equal
    duplicate_threshold = 0.01

    # correction_factor is used to correct the cut_values in case of duplicates
    correction_factor = (
        cut_values.diff().mask(cut_values.diff() < duplicate_threshold).min()
        * duplicate_threshold
    )
    if np.isnan(correction_factor):
        correction_factor = duplicate_threshold**2

    # duplicate_correction is used to shift cut_values in case of duplicates
    # (i.e. when two or more cut_values are equal)
    # This shifts are centered so it does not add any bias
    duplicate_correction = (
        cut_values.diff().abs() < duplicate_threshold
    ).cumsum() * correction_factor
    duplicate_correction = (
        duplicate_correction - duplicate_correction.max() / 2
    )

    cut_values = cut_values + duplicate_correction

    # final_cuts are the categorical values
    final_cuts = pd.cut(
        influence_values,
        cut_values,
        labels=["--", "-", "", " ", "+", "++"],
        include_lowest=True,
    )
    return final_cuts


def influence_from_correlation(correlation_value: float) -> str:
    """Compute the influence from a correlation value.

    Parameters
    ----------
    correlation_value : float
        Correlation value to compute the influence from.

    Returns
    -------
    str
        Influence value.

    """
    strong_correlation = 0.6
    weak_correlation = 0.3
    if correlation_value < -strong_correlation:
        return "--"
    if correlation_value < -weak_correlation:
        return "-"
    if correlation_value < weak_correlation:
        return ""
    if correlation_value < strong_correlation:
        return "+"
    return "++"
