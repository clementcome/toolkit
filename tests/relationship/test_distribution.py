import pandas as pd
import pandera as pa
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

    def test_invalid_categorical_dataframe(
        self, invalid_categorical_dataframe
    ):
        with pytest.raises(pa.errors.SchemaError):
            categorical_distribution(invalid_categorical_dataframe)


class TestSummaryDistributionByTarget:
    @pytest.fixture
    def features(self):
        # Dataframe with 2 numeric features and 2 categorical features with
        # 10 rows
        return pd.DataFrame(
            {
                "num1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "num2": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                "cat1": ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"],
                "cat2": ["b", "b", "c", "c", "a", "a", "b", "b", "c", "c"],
            }
        )

    @pytest.fixture
    def target(self):
        # Series with 10 rows
        return pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

    def test_valid_categorical_dataframe(
        self, features: pd.DataFrame, target: pd.Series
    ):
        summary_numeric, summary_categorical = summary_distribution_by_target(
            features, target
        )
        assert isinstance(summary_numeric, pd.DataFrame)
        assert isinstance(summary_categorical, pd.DataFrame)
        assert (
            summary_numeric.shape[0]
            == sum(features.columns.str.contains("num")) * target.nunique()
        )

    def test_invalid_categorical_dataframe(
        self, invalid_categorical_dataframe
    ):
        with pytest.raises(pa.errors.SchemaError):
            summary_distribution_by_target(invalid_categorical_dataframe)
