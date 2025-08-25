import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator

from cc_tk.feature.correlation import (
    ClusteringCorrelation,
    CorrelationToTarget,
)


class SuiteFeatureSelector:
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
    def setup_method(self):
        self.estimator = CorrelationToTarget()

    def test_plot_correlation_error(self):
        with pytest.raises(NotFittedError):
            self.estimator.plot_correlation()

    def test_plot_correlation(self, mocker: MockerFixture, X_df, y_series):
        mock_plot = mocker.patch.object(pd.DataFrame, "plot")
        self.estimator.fit(X_df, y_series)
        self.estimator.plot_correlation()
        mock_plot.barh.assert_called_once()


class SuiteClusteringCorrelation(SuiteFeatureSelector):
    def test_fit(self, X_df, y_series):
        self.estimator.fit(X_df, y_series)
        assert self.estimator.clusters_col_ is not None

    def test_fit_with_constant(self, X_df, y_series):
        X_df.iloc[:, 0] = 1
        self.estimator.fit(X_df, y_series)
        assert self.estimator.clusters_col_ is not None

    def test_transform(self, X_df, y_series):
        self.estimator.fit(X_df, y_series)
        self.estimator.transform(X_df)


class TestClusteringCorrelationFirst(SuiteClusteringCorrelation):
    def setup_method(self):
        self.estimator = ClusteringCorrelation(summary_method="first")


class TestClusteringCorrelationPCA(SuiteClusteringCorrelation):
    def setup_method(self):
        self.estimator = ClusteringCorrelation(summary_method="pca", threshold=0.9)
