# FILEPATH: /home/ccome/perso/toolkit/tests/relationship/utils_test.py

import pandas as pd
import pytest
from cc_tk.relationship.utils import cut_influence, influence_from_correlation


def test_cut_influence():
    # Create a pandas Series for testing
    influence_values = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # Call the function with the test data
    result = cut_influence(influence_values)

    expected = pd.Series(["--", "--", "-", "", "", " ", "+", "+", "++", "++"])

    pd.testing.assert_series_equal(result.astype(str), expected)


@pytest.mark.parametrize(
    "correlation, expected_influence",
    [
        (-0.7, "--"),
        (-0.4, "-"),
        (0.2, ""),
        (0.4, "+"),
        (0.7, "++"),
    ],
)
def test_influence_from_correlation(correlation, expected_influence):
    # Test with a strong negative correlation
    assert influence_from_correlation(correlation) == expected_influence
