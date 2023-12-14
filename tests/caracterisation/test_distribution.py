import pandas as pd
import pandera as pa
import pytest

from cc_tk.caracterisation.distribution import (
    categorical_distribution,
    numeric_distribution,
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
