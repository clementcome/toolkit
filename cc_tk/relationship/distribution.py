"""Module for computing distribution of data."""
from typing import Tuple

import pandas as pd
import pandera as pa

from cc_tk.relationship.schema import (
    OnlyCategoricalSchema,
    OnlyNumericSchema,
    SeriesType,
    check_input_types,
)


@pa.check_input(OnlyNumericSchema)
def numeric_distribution(numeric_features: pd.DataFrame) -> pd.DataFrame:
    """Compute the distribution of all numeric features.

    Parameters
    ----------
    numeric_features : pd.DataFrame
        Numeric features to compute the distribution of.

    Returns
    -------
    pd.DataFrame
        Distribution of the features.
    """
    distribution_df = numeric_features.describe().T
    distribution_df.index.name = "Variable"
    distribution_df = distribution_df.reset_index()
    return distribution_df


@pa.check_input(OnlyCategoricalSchema)
def categorical_distribution(
    categorical_features: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the distribution of all categorical features.

    Parameters
    ----------
    categorical_features : pd.DataFrame
        Categorical features to compute the distribution of.

    Returns
    -------
    pd.DataFrame
        Distribution of the features.
    """
    distribution_dict = {}
    for feature in categorical_features.columns:
        distribution_dict[feature] = pd.concat(
            (
                categorical_features[feature].value_counts(),
                categorical_features[feature].value_counts(normalize=True),
            ),
            axis=1,
        )

    distribution_df = pd.concat(distribution_dict, axis=0)
    distribution_df.index.names = ["Variable", "Value"]
    distribution_df = distribution_df.reset_index()
    return distribution_df


@check_input_types(
    ("target", SeriesType.CATEGORICAL),
)
def summary_distribution_by_target(
    features: pd.DataFrame, target: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the distribution of all features by target group.

    Parameters
    ----------
    features : pd.DataFrame
        Features to compute the distribution of.
    target : pd.Series
        Target to group by. It must be categorical.

    Returns
    -------
    numeric_summary, catecorigal_summary : Tuple[pd.DataFrame, pd.DataFrame]
        Distribution of the features by target group.
    """
    # Compute the distribution of numeric features by target group
    numeric_features = features.select_dtypes(include="number")
    numeric_distribution_df = numeric_features.groupby(target).describe().T
    numeric_distribution_df.index.name = "Variable"
    numeric_distribution_df = numeric_distribution_df.reset_index()

    # Compute the distribution of categorical features by target group
    categorical_features = features.select_dtypes(exclude="number")
    categorical_distribution_dict = {}
    for feature in categorical_features.columns:
        categorical_distribution_dict[feature] = pd.concat(
            (
                categorical_features[feature].value_counts(),
                categorical_features[feature].value_counts(normalize=True),
            ),
            axis=1,
        )

    categorical_distribution_df = pd.concat(categorical_distribution_dict)

    return numeric_distribution_df, categorical_distribution_df
