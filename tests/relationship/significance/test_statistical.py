import numpy as np
import pandas as pd
import pytest
from cc_tk.relationship.significance.base import SignificanceEnum, SignificanceOutput
from cc_tk.relationship.significance.statistical import (
    AnovaSignificance,
    AnovaSignificanceCategoricalTarget,
    Chi2Significance,
    PearsonSignificance,
)
from tests.relationship.significance.utils import SignificanceTestSuite


class TestPearsonSignificance(SignificanceTestSuite):
    test_class = PearsonSignificance
    values_1 = pd.Series([1, 2, 3, 4, 5])
    wrong_values_1 = pd.Series(["a", "b", "c", "d", "e"])
    wrong_values_1_from_index = pd.Series([1, 2, 3, 4, 5], index=[1, 2, 3, 4, 5])
    values_2 = pd.Series([1, 2, 3, 4, 5])
    wrong_values_2 = pd.Series(["a", "b", "c", "d", "e"])
    expected_pvalue = 0.0
    expected_influence = pd.Series(["++"])
    expected_statistic = 1.0


class TestAnovaSignificance:
    def test__compute_group_info(self):
        # Generate test data
        numeric_values = pd.Series([1, 2, 3, 4, 4])
        categorical_values = pd.Series(["A", "A", "A", "B", "B"])

        # Call the function
        result = AnovaSignificance._compute_group_info(
            numeric_values, categorical_values
        )

        # Assert the output
        assert isinstance(result, dict)
        assert len(result) == 2

        # Assert group A information
        group_a_info = result["A"]
        assert isinstance(group_a_info, dict)
        pd.testing.assert_series_equal(group_a_info["values"], numeric_values.iloc[:3])
        assert group_a_info["mean"] == 2
        assert "normalized_values" in group_a_info
        assert "std" in group_a_info
        assert "ks_test" in group_a_info

        # Assert group B information
        group_b_info = result["B"]
        assert isinstance(group_b_info, dict)
        pd.testing.assert_series_equal(group_b_info["values"], numeric_values.loc[3:])
        assert group_b_info["mean"] == 4
        assert "normalized_values" in group_b_info
        assert "std" in group_b_info
        assert "ks_test" in group_b_info


class TestAnovaSignificanceCategoricalTarget:
    def setup_method(self):
        np.random.seed(22)
        group_desc = {
            "group_0": {"size": 100, "mean": -5, "std": 2},
            "group_1": {"size": 80, "mean": 0, "std": 0.1},
            "group_2": {"size": 50, "mean": 5, "std": 10},
        }
        self.categorical_values = pd.concat(
            [
                pd.Series([group_name] * group["size"], name="category")
                for group_name, group in group_desc.items()
            ],
            ignore_index=True,
        )
        mean_array = np.concatenate(
            [[group["mean"]] * group["size"] for _, group in group_desc.items()]
        )
        std_array = np.concatenate(
            [[group["std"]] * group["size"] for _, group in group_desc.items()]
        )
        # Numeric values for the case where the groups are normally distributed
        # And have the same variance
        self.numeric_values_normal = pd.Series(
            np.random.randn(len(self.categorical_values)) + mean_array,
            name="number",
        )
        # Numeric values for the case where the groups are normally distributed
        # but have different variances
        self.numeric_values_normal_diff_var = pd.Series(
            np.random.randn(len(self.categorical_values)) * std_array + mean_array,
            name="number",
        )

    def test_significance_numeric_categorical_anova(self):
        # Call the function
        result = AnovaSignificanceCategoricalTarget()(
            self.numeric_values_normal, self.categorical_values
        )

        # Assert the output
        assert isinstance(result, SignificanceOutput)
        assert isinstance(result.significance, SignificanceEnum)
        assert isinstance(result.pvalue, float)
        assert len(result.influence) == 3
        assert result.influence.isin(["--", "-", " ", "", "+", "++"]).all()
        assert isinstance(result.statistic, float)
        assert "ANOVA" in result.message

    def test_significance_numeric_categorical_kw(self):
        # Call the function
        result = AnovaSignificanceCategoricalTarget()(
            self.numeric_values_normal_diff_var, self.categorical_values
        )

        # Assert the output
        assert isinstance(result, SignificanceOutput)
        assert isinstance(result.significance, SignificanceEnum)
        assert isinstance(result.pvalue, float)
        assert len(result.influence) == 3
        assert result.influence.isin(["--", "-", " ", "", "+", "++"]).all()
        assert isinstance(result.statistic, float)
        assert "Kruskal-Wallis" in result.message


class TestChi2Significance(SignificanceTestSuite):
    test_class = Chi2Significance
    # values_1 contains 2 classes with 10 individuals each
    values_1 = pd.Series(["a"] * 10 + ["b"] * 10, name="2_classes")
    wrong_values_1 = pd.Series([1] * 10 + [2] * 10)
    wrong_values_1_from_index = pd.Series(["a"] * 10 + ["b"] * 10, index=range(1, 21))
    # values_2 contains 4 classes with 5 individuals each
    values_2 = pd.Series(["a", "b", "c", "d"] * 5, name="4_classes")
    wrong_values_2 = pd.Series([1, 2, 3, 4] * 5)

    expected_pvalue = 0.85
    expected_influence = pd.Series(
        [" ", "-", " ", "-", "-", " ", "-", " "],
        index=pd.MultiIndex.from_product(
            [list("abcd"), list("ab")], names=["4_classes", "2_classes"]
        ),
    ).astype(
        pd.CategoricalDtype(categories=["--", "-", "", " ", "+", "++"], ordered=True)
    )
    expected_statistic = 0.8

    @pytest.mark.parametrize(
        ("significance", "expected_message"),
        [
            (Chi2Significance(), "Hypotheses not verified."),
            (Chi2Significance(hypotheses_threshold=1.0), "Hypotheses verified."),
        ],
    )
    def test_hypotheses_verification(self, significance, expected_message):
        output = significance(self.values_1, self.values_2)
        assert output.message == expected_message
