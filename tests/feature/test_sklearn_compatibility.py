import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from cc_tk.feature.correlation import (
    ClusteringCorrelation,
    CorrelationToTarget,
    PairwiseCorrelationDrop,
)


@parametrize_with_checks([CorrelationToTarget()])
def test_CorrelationToTarget(estimator, check):
    check(estimator)


@parametrize_with_checks(
    [ClusteringCorrelation(summary_method="first")],
    expected_failed_checks=lambda x: {
        "check_n_features_in_after_fitting": "ensuring at least 2 features"
    },
)
def test_ClusteringCorrelationFirst(estimator, check):
    check(estimator)


@parametrize_with_checks(
    [ClusteringCorrelation(summary_method="pca")],
    expected_failed_checks=lambda x: {
        "check_n_features_in_after_fitting": "ensuring at least 2 features"
    },
)
def test_ClusteringCorrelationPCA(estimator, check):
    check(estimator)


@parametrize_with_checks(
    [PairwiseCorrelationDrop()],
    expected_failed_checks=lambda x: {
        "check_n_features_in_after_fitting": "ensuring at least 2 features"
    },
)
def test_PairwiseCorrelationDrop(estimator, check):
    check(estimator)
