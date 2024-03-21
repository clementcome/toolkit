"""Evaluate the significance of the relationship between 2 variables.

Usually this consists in evaluating the relationship between a feature and
the target variable.
"""
from enum import Enum, unique
from typing import Tuple

import pandas as pd
from pydantic import BaseModel, ConfigDict, validate_call
from scipy import stats

from cc_tk.relationship.schema import (
    SeriesType,
    check_input_index,
    check_input_types,
)
from cc_tk.relationship.utils import cut_influence, influence_from_correlation


class Constants:
    """
    Constants for the relationship functions
    """

    WEAK_THRESHOLD = 0.1
    STRONG_THRESHOLD = 0.05


@unique
class SignificanceEnum(str, Enum):
    WEAK_VALUE = "weak"
    MEDIUM_VALUE = "medium"
    STRONG_VALUE = "strong"


class SignificanceOutput(BaseModel):
    """
    Output of the significance functions
    """

    pvalue: float
    influence: pd.Series
    statistic: float
    message: str = ""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def significance(self) -> SignificanceEnum:
        """
        Computing significativity based on pvalue.
        """
        significance = SignificanceEnum.WEAK_VALUE
        if self.pvalue < Constants.WEAK_THRESHOLD:
            significance = SignificanceEnum.MEDIUM_VALUE
        if self.pvalue < Constants.STRONG_THRESHOLD:
            significance = SignificanceEnum.STRONG_VALUE

        return significance

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the output to a dataframe
        """
        return pd.DataFrame(
            {
                "influence": self.influence,
                "pvalue": self.pvalue,
                "statistic": self.statistic,
                "message": self.message,
                "significance": self.significance.value,
            }
        )


@check_input_types(
    ("numeric_values_1", SeriesType.NUMERIC),
    ("numeric_values_2", SeriesType.NUMERIC),
)
@check_input_index("numeric_values_1", "numeric_values_2")
@validate_call(config={"arbitrary_types_allowed": True})
def significance_numeric_numeric(
    numeric_values_1: pd.Series, numeric_values_2: pd.Series
) -> Tuple[str, float, float]:
    """
    Computes the correlation and the significance of this correlation to be
    non-zero

    Parameters
    ----------
    numeric_values_1 : pd.Series
        First numeric values
    numeric_values_2 : pd.Series
        Second numeric values

    Returns
    -------
    SignificanceOutput
        Output of the significance function
    """
    corr_results = stats.pearsonr(numeric_values_1, numeric_values_2)
    correlation = corr_results.statistic
    pvalue = corr_results.pvalue

    influence = influence_from_correlation(correlation)
    influence = pd.Series([influence])

    output = SignificanceOutput(
        pvalue=pvalue,
        influence=influence,
        statistic=correlation,
    )

    return output


@check_input_types(
    ("numeric_values", SeriesType.NUMERIC),
    ("categorical_values", SeriesType.CATEGORICAL),
)
@check_input_index("numeric_values", "categorical_values")
@validate_call(config={"arbitrary_types_allowed": True})
def significance_numeric_categorical(
    numeric_values: pd.Series, categorical_values: pd.Series
) -> SignificanceOutput:
    """
    Computes the significance of distibution difference of a numeric
    variable against categories

    Parameters
    ----------
    numeric_values : pd.Series
        Numeric values which we are interested in knowing if there is a
        difference in distribution
    categorical_values : pd.Series
        Categorical values to divide the numeric values in groups

    Returns
    -------
    SignificanceOutput
        Output of the significance function
    """
    group_info = _compute_group_info(numeric_values, categorical_values)
    ks_pvalue_series, bartlett_pvalue = _perform_tests(group_info)

    # If any of the groups is not gaussian: ks_pvalue_series < 0.05 OR
    # If any of the groups does not have equal variance: bartlett_pvalue < 0.05
    # Then we use Kruskal-Wallis test
    if (ks_pvalue_series < 0.05).any() or (bartlett_pvalue < 0.05):
        test = stats.kruskal(*[info["values"] for info in group_info.values()])
        message = (
            f"{numeric_values.name} grouped by {categorical_values.name} "
            f"are not gaussians with equal variances. "
            f"Computing Kruskal-Wallis p-value."
        )

    else:
        # If all the groups are gaussian and have equal variances
        # we use ANOVA test
        test = stats.f_oneway(*[info["values"] for info in group_info.values()])
        message = (
            f"{numeric_values.name} grouped by {categorical_values.name} "
            f"are gaussians and have equal variances. Computing "
            f"ANOVA p-value."
        )

    statistic = test.statistic
    pvalue = test.pvalue

    mean_by_group = pd.Series({key: info["mean"] for key, info in group_info.items()})
    influence = cut_influence(mean_by_group)

    output = SignificanceOutput(
        pvalue=pvalue,
        influence=influence,
        statistic=statistic,
        message=message,
    )

    return output


def _compute_group_info(
    numeric_values: pd.Series, categorical_values: pd.Series
) -> dict:
    """
    Compute the mean, std, normalized values and Kolmogorov-Smirnov test for
    each group of values

    Parameters
    ----------
    numeric_values : pd.Series
        Numeric values to divide in groups
    categorical_values : pd.Series
        Categorical values to divide the numeric values in groups

    Returns
    -------
    dict
        Dictionary with the information of each group
    """
    group_info = {}
    for categorical_value in categorical_values.unique():
        group_values = numeric_values[categorical_values == categorical_value]
        group_mean = group_values.mean()
        group_std = group_values.std()
        group_normalized_values = (group_values - group_mean) / group_std
        group_test_ks = stats.kstest(group_normalized_values, "norm")
        group_info[categorical_value] = {
            "values": group_values,
            "normalized_values": group_normalized_values,
            "mean": group_mean,
            "std": group_std,
            "ks_test": group_test_ks,
        }
    return group_info


def _perform_tests(group_info: dict) -> Tuple[pd.Series, float]:
    """
    Perform the Kolmogorov-Smirnov test and the Bartlett test for the
    groups of values
    - Kolmogorov-Smirnov test is used to check if the groups are gaussian
    - Bartlett test is used to check if the groups have equal variances
    """
    ks_pvalue_series = pd.Series(
        [group_info[key]["ks_test"].pvalue for key in group_info.keys()]
    )
    bartlett_pvalue = stats.bartlett(
        *[info["values"] for info in group_info.values()]
    ).pvalue
    return ks_pvalue_series, bartlett_pvalue


@check_input_types(
    ("categorical_values_1", SeriesType.CATEGORICAL),
    ("categorical_values_2", SeriesType.CATEGORICAL),
)
@check_input_index(
    "categorical_values_1",
    "categorical_values_2",
)
@validate_call(config={"arbitrary_types_allowed": True})
def significance_categorical_categorical(
    categorical_values_1: pd.Series, categorical_values_2: pd.Series
) -> SignificanceOutput:
    """
    Computes the significance of the difference between 2 categorical
    variables

    Parameters
    ----------
    categorical_values_1 : pd.Series
        First categorical series
    categorical_values_2 : pd.Series
        Second categorical series

    Returns
    -------
    SignificanceOutput
        Output of the significance function
    """
    contingency_table = pd.crosstab(categorical_values_1, categorical_values_2)
    if (contingency_table < 5).any().any():
        hypotheses_verified = False
    else:
        hypotheses_verified = True

    chi2_results = stats.chi2_contingency(contingency_table)
    statistic = chi2_results.statistic
    pvalue = chi2_results.pvalue

    # Influence is computed based on the relative difference between actual
    # values and expected frequencies
    influence = cut_influence(
        (contingency_table - chi2_results.expected_freq)
        .divide(chi2_results.expected_freq)
        .unstack()
    )

    if hypotheses_verified:
        message = "Hypotheses verified."
    else:
        message = "Hypotheses not verified."

    output = SignificanceOutput(
        pvalue=pvalue,
        influence=influence,
        statistic=statistic,
        message=message,
    )

    return output
