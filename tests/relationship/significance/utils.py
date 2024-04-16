import pandas as pd
import pytest
from cc_tk.relationship.significance.base import SignificanceOutput


class SignificanceTestSuite:
    """Test suite for significance tests.

    Attributes
    ----------
    test_class : class
        Significance test class to test
    values_1 : pd.Series
        First values to test
    wrong_values_1 : pd.Series
        Wrong first values to test
    wrong_values_1_from_index : pd.Series
        Wrong first values (because of the index) to test
    values_2 : pd.Series
        Second values to test
    wrong_values_2 : pd.Series
        Wrong second values to test
    expected_pvalue : float
        Expected pvalue
    expected_influence : pd.Series
        Expected influence
    expected_statistic : float
        Expected statistic

    Methods
    -------
    setup_method()
        Setup method for the test
    test_evaluate_method()
        Test the _evaluate method
    test_call_correct()
        Test the call method with correct values
    test_call_error_wrong_values_1()
        Test the call method with wrong values 1
    test_call_error_wrong_values_2()
        Test the call method with wrong values 2
    test_call_error_wrong_values_1_from_index()
        Test the call method with wrong values 1 from index

    """

    # Attributes to set in the subclass
    test_class = None
    values_1 = None
    wrong_values_1 = None
    wrong_values_1_from_index = None
    values_2 = None
    wrong_values_2 = None
    expected_pvalue = 0.05
    expected_influence = pd.Series(["+", ""])
    expected_statistic = 1.0

    def setup_method(self):
        if (
            isinstance(self.test_class, type)
            and hasattr(self.test_class._evaluate, "__isabstractmethod__")
            and self.test_class._evaluate.__isabstractmethod__
        ):

            class MockSignificance(self.test_class):
                def _evaluate(self_mock, values_1, values_2):
                    return SignificanceOutput(
                        pvalue=self.expected_pvalue,
                        influence=self.expected_influence,
                        statistic=self.expected_statistic,
                    )

            self.significance = MockSignificance()
        else:
            self.significance = self.test_class()

    def test_evaluate_method(self):
        output = self.significance._evaluate(self.values_1, self.values_2)
        assert isinstance(output, SignificanceOutput)
        assert output.pvalue == pytest.approx(self.expected_pvalue, abs=1e-2)
        assert output.statistic == pytest.approx(self.expected_statistic, rel=1e-2)
        pd.testing.assert_series_equal(output.influence, self.expected_influence)

    def test_call_correct(self):
        output = self.significance(self.values_1, self.values_2)
        assert isinstance(output, SignificanceOutput)
        assert output.pvalue == pytest.approx(self.expected_pvalue, abs=1e-2)
        assert output.statistic == pytest.approx(self.expected_statistic, rel=1e-2)
        pd.testing.assert_series_equal(output.influence, self.expected_influence)

    def test_call_error_wrong_values_1(self):
        with pytest.raises(TypeError):
            self.significance(self.wrong_values_1, self.values_2)

    def test_call_error_wrong_values_2(self):
        with pytest.raises(TypeError):
            self.significance(self.values_1, self.wrong_values_2)

    def test_call_error_wrong_values_1_from_index(self):
        with pytest.raises(ValueError):
            self.significance(self.wrong_values_1_from_index, self.values_2)
