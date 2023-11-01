from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator

from cc_tk.feature.correlation import CorrelationToTarget


class SuiteFeatureSelector(ABC):
    @property
    @abstractmethod
    def estimator(self) -> BaseEstimator:
        pass

    @pytest.fixture
    def X_array(self):
        np.random.seed(42)
        return np.random.rand(100, 10)

    @pytest.fixture
    def X_df(self, X_array):
        return pd.DataFrame(X_array, columns=[f"col_{i}" for i in range(10)])

    @pytest.fixture
    def y_array(self):
        np.random.seed(42)
        return np.random.rand(100)

    @pytest.fixture
    def y_series(self, y_array):
        return pd.Series(y_array, name="target")


class TestCorrelationToTarget(SuiteFeatureSelector):
    @property
    def estimator(self):
        if not hasattr(self, "_estimator"):
            self._estimator = CorrelationToTarget()
        return self._estimator

    def test_plot_correlation_error(self):
        with pytest.raises(NotFittedError):
            self.estimator.plot_correlation()

    def test_plot_correlation(self, mocker: MockerFixture, X_df, y_series):
        mock_plot = mocker.patch.object(pd.DataFrame, "plot")
        self.estimator.fit(X_df, y_series)
        self.estimator.plot_correlation()
        mock_plot.barh.assert_called_once()
