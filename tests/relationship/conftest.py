import pytest
import pandas as pd


@pytest.fixture
def features():
    # Dataframe with 2 numeric features and 2 categorical features with
    # 10 rows
    return pd.DataFrame(
        {
            "num1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "num2": [10, 9, 8, 7, 6, 5, 4, 3, 1, 2],
            "cat1": ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"],
            "cat2": ["b", "b", "c", "c", "a", "a", "b", "b", "c", "c"],
        }
    )


@pytest.fixture
def target_categorical():
    # Series with 10 rows
    return pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]).astype(object)


@pytest.fixture
def target_numeric():
    # Series with 10 rows
    return pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
