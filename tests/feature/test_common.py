import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from cc_tk.feature.correlation import CorrelationToTarget


@parametrize_with_checks([CorrelationToTarget()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
