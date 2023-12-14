"""Module for computing distribution of data."""
import pandas as pd
import pandera as pa

from cc_tk.caracterisation.schema import (
    OnlyCategoricalSchema,
    OnlyNumericSchema,
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
