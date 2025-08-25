"""Significance tests based on statistical methods."""

from typing import Optional, Tuple

import pandas as pd
from scipy import stats

from cc_tk.relationship.significance.base import (
    Constants,
    SignificanceCategoricalCategorical,
    SignificanceCategoricalNumeric,
    SignificanceNumericCategorical,
    SignificanceNumericNumeric,
    SignificanceOutput,
)
from cc_tk.relationship.utils import cut_influence, influence_from_correlation


class PearsonSignificance(SignificanceNumericNumeric):
    """Significance test based on Pearson correlation."""

    def _evaluate(
        self, numeric_values_1: pd.Series, numeric_values_2: pd.Series
    ) -> SignificanceOutput:
        """Pearson correlation-based significance.

        The significance is based on the test of pearson correlation not being
        null.

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


class AnovaSignificance:
    """Base class for ANOVA significance tests."""

    def compute_significance(
        self, numeric_values: pd.Series, categorical_values: pd.Series
    ) -> SignificanceOutput:
        """Anova or Kruskal-Wallis significance test.

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
        group_info = self._compute_group_info(
            numeric_values, categorical_values
        )
        ks_pvalue_series, bartlett_pvalue = self._perform_tests(group_info)

        # If any of the groups is not gaussian: ks_pvalue_series < 0.05 OR
        # If any of the groups does not have equal variance:
        #   bartlett_pvalue < 0.05
        # Then we use Kruskal-Wallis test
        if (ks_pvalue_series < Constants.STRONG_THRESHOLD).any() or (
            bartlett_pvalue < Constants.STRONG_THRESHOLD
        ):
            test = stats.kruskal(
                *[info["values"] for info in group_info.values()]
            )
            message = (
                f"{numeric_values.name} grouped by {categorical_values.name} "
                f"are not gaussians with equal variances. "
                f"Computing Kruskal-Wallis p-value."
            )

        else:
            # If all the groups are gaussian and have equal variances
            # we use ANOVA test
            test = stats.f_oneway(
                *[info["values"] for info in group_info.values()]
            )
            message = (
                f"{numeric_values.name} grouped by {categorical_values.name} "
                f"are gaussians and have equal variances. Computing "
                f"ANOVA p-value."
            )

        statistic = test.statistic
        pvalue = test.pvalue

        mean_by_group = pd.Series(
            {key: info["mean"] for key, info in group_info.items()}
        )
        influence = cut_influence(mean_by_group)

        output = SignificanceOutput(
            pvalue=pvalue,
            influence=influence,
            statistic=statistic,
            message=message,
        )

        return output

    @staticmethod
    def _compute_group_info(
        numeric_values: pd.Series, categorical_values: pd.Series
    ) -> dict:
        """Compute information for each categorical group.

        Informations computed are: mean, std, normalized values and
        Kolmogorov-Smirnov test.

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
            group_values = numeric_values[
                categorical_values == categorical_value
            ]
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

    @staticmethod
    def _perform_tests(group_info: dict) -> Tuple[pd.Series, float]:
        """Perform Kolmogorov-Smirnov and Bartlett tests for the groups.

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


class AnovaSignificanceNumericTarget(
    SignificanceCategoricalNumeric, AnovaSignificance
):
    """Anova significance test with numeric target."""

    def _evaluate(
        self, categorical_values: pd.Series, numeric_values: pd.Series
    ) -> SignificanceOutput:
        return self.compute_significance(numeric_values, categorical_values)


class AnovaSignificanceCategoricalTarget(
    SignificanceNumericCategorical, AnovaSignificance
):
    """Anova significance test with categorical target."""

    def _evaluate(
        self, numeric_values: pd.Series, categorical_values: pd.Series
    ) -> SignificanceOutput:
        return self.compute_significance(numeric_values, categorical_values)


class Chi2Significance(SignificanceCategoricalCategorical):
    """Significance based on Chi2 test."""

    def __init__(self, hypotheses_threshold: Optional[int] = 5) -> None:
        """Initialize the Chi2 significance test.

        Parameters
        ----------
        hypotheses_threshold : Optional[int], optional
            Threshold for hypotheses verification, by default 5

        """
        self.hypotheses_threshold = hypotheses_threshold

    def _evaluate(
        self, categorical_values_1: pd.Series, categorical_values_2: pd.Series
    ) -> SignificanceOutput:
        """Chi2 significance test for categorical/categorical variables.

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
        contingency_table = pd.crosstab(
            categorical_values_1,
            categorical_values_2,
        )
        if (contingency_table < self.hypotheses_threshold).any().any():
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
