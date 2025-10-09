import inspect

import pandas as pd
import pandera.pandas as pa
import pytest
from pydantic import ValidationError

from cc_tk.relationship.schema import (
    OnlyCategoricalSchema,
    OnlyNumericSchema,
    SeriesType,
    check_input_index,
    check_input_types,
    check_series_in_signature,
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


class TestCheckSeriesInSignature:
    def test_check_series_in_signature_valid(self):
        def func(arg1: pd.Series, arg2: pd.Series):
            pass

        signature = check_series_in_signature(func, "arg1", "arg2")
        assert isinstance(signature, inspect.Signature)

    def test_check_series_in_signature_invalid_arg_name(self):
        def func(arg1: pd.Series, arg2: pd.Series):
            pass

        with pytest.raises(ValueError):
            check_series_in_signature(func, "arg1", "arg3")

    def test_check_series_in_signature_invalid_arg_type(self):
        def func(arg1: pd.Series, arg2: int):
            pass

        with pytest.raises(TypeError):
            check_series_in_signature(func, "arg1", "arg2")


class TestCheckInputTypesDecoration:
    def test_check_input_types_decoration_valid(self):
        @check_input_types(
            ("arg1", SeriesType.NUMERIC), ("arg2", SeriesType.CATEGORICAL)
        )
        def dummy_function(arg1: pd.Series, arg2: pd.Series, arg3: int):
            pass

    def test_check_input_types_decoration_invalid_arg(self):
        with pytest.raises(ValueError):

            @check_input_types(
                ("arg1", SeriesType.NUMERIC), ("arg4", SeriesType.CATEGORICAL)
            )
            def dummy_function(arg1: pd.Series, arg2: pd.Series, arg3: int):
                pass

    def test_check_input_types_decoration_invalid_type(self):
        with pytest.raises(TypeError):

            @check_input_types(
                ("arg1", SeriesType.NUMERIC), ("arg3", SeriesType.CATEGORICAL)
            )
            def dummy_function(arg1: pd.Series, arg2: pd.Series, arg3: int):
                pass

    def test_check_input_types_decoration_invalid_series_type(self):
        with pytest.raises(ValidationError):

            @check_input_types(("arg1", SeriesType.NUMERIC), ("arg2", "numerical"))
            def dummy_function(arg1: pd.Series, arg2: pd.Series, arg3: int):
                pass


class TestCheckInputTypes:
    @staticmethod
    @check_input_types(("arg1", SeriesType.NUMERIC), ("arg2", SeriesType.CATEGORICAL))
    def dummy_function(arg1: pd.Series, arg2: pd.Series):
        pass

    @pytest.fixture
    def numeric_series(self):
        return pd.Series([1, 2, 3])

    @pytest.fixture
    def categorical_series(self):
        return pd.Series(["a", "b", "c"])

    def test_check_input_types_valid(self, numeric_series, categorical_series):
        self.dummy_function(numeric_series, categorical_series)

    def test_check_input_types_invalid_numeric(self, categorical_series):
        invalid_numeric_series = pd.Series([1, 2, 3, "a"])
        with pytest.raises(TypeError):
            self.dummy_function(invalid_numeric_series, categorical_series)

    def test_check_input_types_invalid_categorical(self, numeric_series):
        invalid_categorical_series = numeric_series.copy()
        with pytest.raises(TypeError):
            self.dummy_function(numeric_series, invalid_categorical_series)


class TestCheckInputIndex:
    @staticmethod
    @check_input_index("arg1", "arg2")
    def dummy_function(arg1: pd.Series, arg2: pd.Series):
        pass

    @pytest.fixture
    def series_with_same_index(self):
        index = pd.Index([1, 2, 3])
        return pd.Series([1, 2, 3], index=index)

    @pytest.fixture
    def series_with_different_index(self):
        index1 = pd.Index([1, 2, 3])
        index2 = pd.Index([4, 5, 6])
        return pd.Series([1, 2, 3], index=index1), pd.Series([4, 5, 6], index=index2)

    def test_check_input_index_valid(self, series_with_same_index):
        self.dummy_function(series_with_same_index, series_with_same_index)

    def test_check_input_index_invalid(self, series_with_different_index):
        with pytest.raises(ValueError):
            self.dummy_function(*series_with_different_index)
