import pandas as pd
from cc_tk.relationship.significance.base import (
    SignificanceCategoricalCategorical,
    SignificanceCategoricalNumeric,
    SignificanceEnum,
    SignificanceNumericCategorical,
    SignificanceNumericNumeric,
    SignificanceOutput,
)
from tests.relationship.significance.utils import SignificanceTestSuite


class TestSignificanceOutput:
    def test_significance_property_weak(self):
        output = SignificanceOutput(pvalue=0.1, influence=pd.Series(), statistic=1.0)
        assert output.significance == SignificanceEnum.WEAK_VALUE

    def test_significance_property_medium(self):
        output = SignificanceOutput(pvalue=0.06, influence=pd.Series(), statistic=1.0)
        assert output.significance == SignificanceEnum.MEDIUM_VALUE

    def test_significance_property_strong(self):
        output = SignificanceOutput(pvalue=0.01, influence=pd.Series(), statistic=1.0)
        assert output.significance == SignificanceEnum.STRONG_VALUE

    def test_to_dataframe(self):
        output = SignificanceOutput(
            pvalue=0.05, influence=pd.Series(["+", ""]), statistic=1.0
        )
        df = output.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        pd.testing.assert_frame_equal(
            df,
            pd.DataFrame(
                {
                    "influence": ["+", ""],
                    "pvalue": 0.05,
                    "statistic": 1.0,
                    "message": "",
                    "significance": SignificanceEnum.MEDIUM_VALUE,
                }
            ),
        )


class TestSignificanceNumericNumeric(SignificanceTestSuite):
    test_class = SignificanceNumericNumeric
    values_1 = pd.Series([1, 2, 3])
    wrong_values_1 = pd.Series(["a", "b", "c"])
    values_2 = pd.Series([4, 5, 6])
    wrong_values_2 = pd.Series(["d", "e", "f"])
    wrong_values_1_from_index = pd.Series([1, 2, 3], index=[1, 2, 3])


class TestSignificanceCategoricalNumeric(SignificanceTestSuite):
    test_class = SignificanceCategoricalNumeric
    values_1 = pd.Series(["A", "B", "C"])
    wrong_values_1 = pd.Series([1, 2, 3])
    values_2 = pd.Series([4, 5, 6])
    wrong_values_2 = pd.Series(["d", "e", "f"])
    wrong_values_1_from_index = pd.Series(["A", "B", "C"], index=[1, 2, 3])


class TestSignificanceNumericCategorical(SignificanceTestSuite):
    test_class = SignificanceNumericCategorical
    values_1 = pd.Series([1, 2, 3])
    wrong_values_1 = pd.Series(["a", "b", "c"])
    values_2 = pd.Series(["A", "B", "C"])
    wrong_values_2 = pd.Series([4, 5, 6])
    wrong_values_1_from_index = pd.Series([1, 2, 3], index=[1, 2, 3])


class TestSignificanceCategoricalCategorical(SignificanceTestSuite):
    test_class = SignificanceCategoricalCategorical
    values_1 = pd.Series(["A", "B", "C"])
    wrong_values_1 = pd.Series([1, 2, 3])
    values_2 = pd.Series(["D", "E", "F"])
    wrong_values_2 = pd.Series([4, 5, 6])
    wrong_values_1_from_index = pd.Series(["A", "B", "C"], index=[1, 2, 3])
