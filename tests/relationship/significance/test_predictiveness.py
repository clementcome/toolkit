from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from cc_tk.relationship.significance.base import SignificanceOutput
from cc_tk.relationship.significance.predictiveness import (
    PredictivenessCategoricalCategorical,
    PredictivenessCategoricalNumeric,
    PredictivenessNumericCategorical,
    PredictivenessNumericNumeric,
    PredictivenessSignificance,
    estimate_pvalue,
)
from pydantic import ValidationError
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from tests.relationship.significance.utils import SignificanceTestSuite


class TestEstimatePvalue:
    @pytest.mark.parametrize(
        "statistic_value, statistic_array, kind, expected",
        [
            (0.0, np.array([0, -1, 1]), "both", 1.0),
            (0.0, np.array([0, -1, 1]), "left", 0.5),
            (0.0, np.array([0, -1, 1]), "right", 0.5),
            (1.0, np.array([0, -1, 1]), "right", 1 - norm.cdf(1.0)),
            (-1.0, np.array([0, -1, 1]), "both", 2 * norm.cdf(-1.0)),
        ],
    )
    def test_estimate_pvalue(self, statistic_value, statistic_array, kind, expected):
        assert estimate_pvalue(statistic_value, statistic_array, kind) == expected

    def test_estimate_pvalue_wrong_kind(self):
        with pytest.raises(ValidationError, match="kind"):
            estimate_pvalue(0.0, np.array([0, -1, 1]), kind="wrong")

    def test_estimate_pvalue_wrong_statistic_array(self):
        with pytest.raises(ValidationError, match="ndarray"):
            estimate_pvalue(0.0, [0, -1, 1], kind="both")


class TestPredictivenessSignificance:
    @pytest.fixture
    def predictiveness_significance(self) -> PredictivenessSignificance:
        model = LinearRegression()
        scoring = "r2"
        cv = 5
        n_bootstrap = 100
        return PredictivenessSignificance(
            model=model,
            scoring=scoring,
            cv=cv,
            n_bootstrap=n_bootstrap,
        )

    @pytest.fixture
    def x_y(self):
        x = pd.Series(range(50))
        y = 2 * x
        return x, y

    def test_score_predictiveness(
        self,
        predictiveness_significance: PredictivenessSignificance,
        x_y: Tuple[pd.Series, pd.Series],
    ):
        x, y = x_y
        score = predictiveness_significance.score_predictiveness(x, y)
        assert isinstance(score, float)
        assert score == 1.0

    @pytest.mark.parametrize(
        ("n_sample", "expected_match"),
        [
            (
                5,
                (
                    "The predictiveness score is NaN. This may be due to too "
                    "few data points for cross-validation to work correctly."
                ),
            ),
            (
                4,
                (
                    "Cannot have number of splits n_splits=5 greater than the "
                    "number of samples: n_samples=4."
                ),
            ),
        ],
    )
    def test_score_predictiveness_not_enough_samples(
        self,
        predictiveness_significance: PredictivenessSignificance,
        n_sample: int,
        expected_match: str,
    ):
        x = pd.Series(range(n_sample))
        y = 2 * x
        with pytest.raises(
            ValueError,
            match=expected_match,
        ):
            predictiveness_significance.score_predictiveness(x, y)

    def test_evaluate(
        self,
        predictiveness_significance: PredictivenessSignificance,
        x_y: Tuple[pd.Series, pd.Series],
    ):
        feature_values, target_values = x_y
        output = predictiveness_significance._evaluate(feature_values, target_values)
        assert isinstance(output, SignificanceOutput)
        assert output.pvalue == pytest.approx(0.0, abs=1e-3)
        assert isinstance(output.influence, pd.Series)
        assert output.influence.empty
        assert isinstance(output.statistic, float)
        assert output.statistic == 1.0

    @pytest.mark.parametrize("n_sample, expected_length,", [(None, 50), (5, 5)])
    def test_sample_values(
        self,
        predictiveness_significance: PredictivenessSignificance,
        x_y: Tuple[pd.Series, pd.Series],
        n_sample: int,
        expected_length: int,
    ):
        feature_values, target_values = x_y
        predictiveness_significance.n_sample = n_sample
        (
            sampled_feature_values,
            sampled_target_values,
        ) = predictiveness_significance._sample_values(feature_values, target_values)
        assert isinstance(sampled_feature_values, pd.Series)
        assert isinstance(sampled_target_values, pd.Series)
        pd.testing.assert_index_equal(
            sampled_feature_values.index, sampled_target_values.index
        )
        assert len(sampled_feature_values) == expected_length

    def test_compute_bootstrap_scores(
        self,
        predictiveness_significance: PredictivenessSignificance,
        x_y: Tuple[pd.Series, pd.Series],
    ):
        feature_values, target_values = x_y
        bootstrap_scores = predictiveness_significance._compute_bootstrap_scores(
            feature_values, target_values
        )
        assert isinstance(bootstrap_scores, np.ndarray)
        assert bootstrap_scores.shape == (predictiveness_significance.n_bootstrap,)

    def test_wrapper_score_predictiveness(
        self,
        predictiveness_significance: PredictivenessSignificance,
        x_y: Tuple[pd.Series, pd.Series],
    ):
        feature_values, target_values = x_y
        score = predictiveness_significance._wrapper_score_predictiveness(
            (feature_values, target_values)
        )
        assert isinstance(score, float)
        assert score == 1.0


class TestPredictivenessNumericNumeric(SignificanceTestSuite):
    test_class = PredictivenessNumericNumeric
    values_1 = pd.Series(range(50))
    wrong_values_1 = pd.Series(["a", "b", "c", "d", "e"] * 10)
    values_2 = pd.Series([0] * 25 + [1] * 25)
    wrong_values_2 = pd.Series(["f", "g", "h", "i", "j"] * 10)
    wrong_values_1_from_index = pd.Series([1, 2, 3])
    expected_influence = pd.Series()
    expected_statistic = 1.0
    expected_pvalue = 0.0


class TestPredictivenessCategoricalNumeric(SignificanceTestSuite):
    test_class = PredictivenessCategoricalNumeric
    values_1 = pd.Series(["A"] * 25 + ["B"] * 25)
    wrong_values_1 = pd.Series([1, 2, 3, 4, 5] * 10)
    values_2 = pd.Series([0] * 25 + [1] * 25)
    wrong_values_2 = pd.Series(["f", "g", "h", "i", "j"] * 10)
    wrong_values_1_from_index = pd.Series(["A", "B", "C"])
    expected_influence = pd.Series()
    expected_statistic = 1.0
    expected_pvalue = 0.0


class TestPredictivenessNumericCategorical(SignificanceTestSuite):
    test_class = PredictivenessNumericCategorical
    values_1 = pd.Series(np.arange(60) % 30)
    wrong_values_1 = pd.Series(["a", "b", "c", "d", "e", "f"] * 10)
    values_2 = pd.Series((["A"] * 15 + ["B"] * 15) * 2)
    wrong_values_2 = pd.Series([1, 2, 3, 4, 5, 6] * 10)
    wrong_values_1_from_index = pd.Series([1, 2, 3])
    expected_influence = pd.Series()
    expected_statistic = 1.0
    expected_pvalue = 0.0


class TestPredictivenessCategoricalCategorical(SignificanceTestSuite):
    test_class = PredictivenessCategoricalCategorical
    values_1 = pd.Series(["A"] * 25 + ["B"] * 25)
    wrong_values_1 = pd.Series([1, 2, 3, 4, 5] * 10)
    values_2 = pd.Series(["A"] * 25 + ["B"] * 25)
    wrong_values_2 = pd.Series([1, 2, 3, 4, 5] * 10)
    wrong_values_1_from_index = pd.Series(["A", "B", "C"])
    expected_influence = pd.Series()
    expected_statistic = 1.0
    expected_pvalue = 0.0
