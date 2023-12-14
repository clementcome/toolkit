import pandas as pd
import pandera as pa
import pytest

from cc_tk.caracterisation.schema import (
    OnlyCategoricalSchema,
    OnlyNumericSchema,
)


class TestOnlyNumericSchema:
    @pytest.fixture
    def valid_dataframe(self):
        return pd.DataFrame({"col1": [1, 2, 3], "col2": [4.5, 6.7, 8.9]})

    @pytest.fixture
    def invalid_dataframe(self):
        return pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

    def test_valid_dataframe(self, valid_dataframe):
        OnlyNumericSchema.validate(valid_dataframe)

    def test_invalid_dataframe(self, invalid_dataframe):
        with pytest.raises(pa.errors.SchemaError):
            OnlyNumericSchema.validate(invalid_dataframe)


class TestOnlyCategoricalSchema:
    @pytest.fixture
    def valid_dataframe(self):
        return pd.DataFrame({"col1": ["a", "b", "c"], "col2": ["x", "y", "z"]})

    @pytest.fixture
    def invalid_dataframe(self):
        return pd.DataFrame({"col1": [1, 2, 3], "col2": ["x", "y", "z"]})

    def test_valid_dataframe(self, valid_dataframe):
        OnlyCategoricalSchema.validate(valid_dataframe)

    def test_valid_dataframe_with_category(self, valid_dataframe):
        valid_dataframe["col1"] = valid_dataframe["col1"].astype("category")
        OnlyCategoricalSchema.validate(valid_dataframe)

    def test_invalid_dataframe(self, invalid_dataframe):
        with pytest.raises(pa.errors.SchemaError):
            OnlyCategoricalSchema.validate(invalid_dataframe)
