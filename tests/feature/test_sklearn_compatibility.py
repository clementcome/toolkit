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


@parametrize_with_checks([ClusteringCorrelation(summary_method="first")])
def test_ClusteringCorrelationFirst(estimator, check):
    check(estimator)


@parametrize_with_checks([ClusteringCorrelation(summary_method="pca")])
def test_ClusteringCorrelationPCA(estimator, check):
    check(estimator)


@parametrize_with_checks([PairwiseCorrelationDrop()])
def test_PairwiseCorrelationDrop(estimator, check):
    check(estimator)
