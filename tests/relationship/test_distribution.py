import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from cc_tk.relationship.distribution import (
    categorical_distribution,
    numeric_distribution,
    summary_distribution_by_target,
)


class TestNumericDistribution:
    @pytest.fixture
    def valid_dataframe(self):
        return pd.DataFrame({"col1": [1, 2, 3], "col2": [4.5, 6.7, 8.9]})

    @pytest.fixture
    def invalid_dataframe(self):
        return pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

    def test_valid_dataframe(self, valid_dataframe):
        result = numeric_distribution(valid_dataframe)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == valid_dataframe.shape[1]
        assert result.columns.tolist() == [
            "Variable",
            "count",
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
        ]

    def test_invalid_dataframe(self, invalid_dataframe):
        with pytest.raises(pa.errors.SchemaError):
            numeric_distribution(invalid_dataframe)

    def test_empty_numeric_dataframe(self):
        actual = numeric_distribution(pd.DataFrame())
        assert isinstance(actual, pd.DataFrame)
        assert actual.empty


class TestCategoricalDistribution:
    @pytest.fixture
    def valid_categorical_dataframe(self):
        return pd.DataFrame({"col1": ["a", "b", "a"], "col2": ["b", "b", "c"]})

    @pytest.fixture
    def invalid_categorical_dataframe(self):
        return pd.DataFrame({"col1": [1, 2, 3], "col2": [4.5, 6.7, 8.9]})

    def test_valid_categorical_dataframe(self, valid_categorical_dataframe):
        result = categorical_distribution(valid_categorical_dataframe)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == valid_categorical_dataframe.nunique().sum()
        assert set(result.columns) == {
            "Variable",
            "Value",
            "count",
            "proportion",
        }

    def test_invalid_categorical_dataframe(self, invalid_categorical_dataframe):
        with pytest.raises(pa.errors.SchemaError):
            categorical_distribution(invalid_categorical_dataframe)

    def test_empty_categorical_dataframe(self):
        actual = categorical_distribution(pd.DataFrame())
        assert isinstance(actual, pd.DataFrame)
        assert actual.empty


class TestSummaryDistributionByTarget:
    def test_categorical_and_numeric_features(
        self, features: pd.DataFrame, target_categorical: pd.Series
    ):
        summary_numeric, summary_categorical = summary_distribution_by_target(
            features, target_categorical
        )
        assert isinstance(summary_numeric, pd.DataFrame)
        assert isinstance(summary_categorical, pd.DataFrame)
        assert summary_numeric.shape[0] == (
            sum(features.columns.str.contains("num")) * target_categorical.nunique()
        )
        pd.testing.assert_index_equal(
            summary_numeric.index,
            pd.MultiIndex.from_tuples(
                [
                    ["num1", 0],
                    ["num1", 1],
                    ["num2", 0],
                    ["num2", 1],
                ],
                names=["Variable", "Target"],
            ),
        )
        pd.testing.assert_index_equal(
            summary_numeric.columns,
            pd.Index(["count", "mean", "std", "min", "25%", "50%", "75%", "max"]),
        )
        assert summary_categorical.shape[0] == (
            sum(features.loc[:, features.columns.str.contains("cat")].nunique())
            * target_categorical.nunique()
        )
        pd.testing.assert_index_equal(
            summary_categorical.columns, pd.Index(["count", "proportion"])
        )
        pd.testing.assert_index_equal(
            summary_categorical.index,
            pd.MultiIndex.from_tuples(
                [
                    ["cat1", 0, "a"],
                    ["cat1", 0, "b"],
                    ["cat1", 0, "c"],
                    ["cat1", 1, "a"],
                    ["cat1", 1, "b"],
                    ["cat1", 1, "c"],
                    ["cat2", 0, "a"],
                    ["cat2", 0, "b"],
                    ["cat2", 0, "c"],
                    ["cat2", 1, "a"],
                    ["cat2", 1, "b"],
                    ["cat2", 1, "c"],
                ],
                names=["Variable", "Target", "Value"],
            ),
        )

    def test_invalid_target(self, features: pd.DataFrame, target_categorical: pd.Series):
        with pytest.raises(TypeError):
            summary_distribution_by_target(features, target_categorical.astype(float))

    def test_only_categorical_features(
        self, features: pd.DataFrame, target_categorical: pd.Series
    ):
        summary_numeric, summary_categorical = summary_distribution_by_target(
            features.select_dtypes(exclude=[np.number]), target_categorical
        )
        assert isinstance(summary_numeric, pd.DataFrame)
        assert isinstance(summary_categorical, pd.DataFrame)
        assert summary_numeric.empty

    def test_only_numeric_features(
        self, features: pd.DataFrame, target_categorical: pd.Series
    ):
        summary_numeric, summary_categorical = summary_distribution_by_target(
            features.select_dtypes(include=[np.number]), target_categorical
        )
        assert isinstance(summary_numeric, pd.DataFrame)
        assert isinstance(summary_categorical, pd.DataFrame)
        assert summary_categorical.empty

    def test_no_features(self, target_categorical: pd.Series):
        summary_numeric, summary_categorical = summary_distribution_by_target(
            pd.DataFrame(), target_categorical
        )
        assert isinstance(summary_numeric, pd.DataFrame)
        assert isinstance(summary_categorical, pd.DataFrame)
        assert summary_numeric.empty
        assert summary_categorical.empty
