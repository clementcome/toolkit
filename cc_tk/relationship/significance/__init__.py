"""Significance functions for different input types and significance types."""

from typing import Literal, overload

from pydantic import validate_call

from cc_tk.relationship.significance.base import (
    SignificanceCategoricalCategorical,
    SignificanceCategoricalNumeric,
    SignificanceNumericCategorical,
    SignificanceNumericNumeric,
    SignificanceType,
    VariableType,
)
from cc_tk.relationship.significance.predictiveness import (
    PredictivenessCategoricalCategorical,
    PredictivenessCategoricalNumeric,
    PredictivenessNumericCategorical,
    PredictivenessNumericNumeric,
)
from cc_tk.relationship.significance.statistical import (
    AnovaSignificanceCategoricalTarget,
    AnovaSignificanceNumericTarget,
    Chi2Significance,
    PearsonSignificance,
)

__all__ = ["get_significance"]


SIGNIFICANCE_DICT = {
    SignificanceType.STATISTICAL: {
        VariableType.NUMERIC: {
            VariableType.NUMERIC: PearsonSignificance,
            VariableType.CATEGORICAL: AnovaSignificanceCategoricalTarget,
        },
        VariableType.CATEGORICAL: {
            VariableType.NUMERIC: AnovaSignificanceNumericTarget,
            VariableType.CATEGORICAL: Chi2Significance,
        },
    },
    SignificanceType.PREDICTIVENESS: {
        VariableType.NUMERIC: {
            VariableType.NUMERIC: PredictivenessNumericNumeric,
            VariableType.CATEGORICAL: PredictivenessNumericCategorical,
        },
        VariableType.CATEGORICAL: {
            VariableType.NUMERIC: PredictivenessCategoricalNumeric,
            VariableType.CATEGORICAL: PredictivenessCategoricalCategorical,
        },
    },
}


@overload
def get_significance(
    feature_type: Literal[VariableType.NUMERIC],
    target_type: Literal[VariableType.NUMERIC],
    significance_type: SignificanceType,
) -> SignificanceNumericNumeric: ...  # pragma: no cover


@overload
def get_significance(
    feature_type: Literal[VariableType.NUMERIC],
    target_type: Literal[VariableType.CATEGORICAL],
    significance_type: SignificanceType,
) -> SignificanceNumericCategorical: ...  # pragma: no cover


@overload
def get_significance(
    feature_type: Literal[VariableType.CATEGORICAL],
    target_type: Literal[VariableType.NUMERIC],
    significance_type: SignificanceType,
) -> SignificanceCategoricalNumeric: ...  # pragma: no cover


@overload
def get_significance(
    feature_type: Literal[VariableType.CATEGORICAL],
    target_type: Literal[VariableType.CATEGORICAL],
    significance_type: SignificanceType,
) -> SignificanceCategoricalCategorical: ...  # pragma: no cover


@validate_call
def get_significance(
    feature_type: VariableType,
    target_type: VariableType,
    significance_type: SignificanceType,
    **kwargs,
):
    """Get significance function based on variable types and significance type.

    Parameters
    ----------
    feature_type : VariableType
        Input type of the feature.
    target_type : VariableType
        Input type of the target.
    significance_type : SignificanceType
        Significance type.
    **kwargs
        Additional keyword arguments passed to the significance initialization.

    Returns
    -------
    Significance
        Significance instance.

    Raises
    ------
    ValueError
        If the feature type and target type are not supported or incompatible
        with the significance type.

    """
    try:
        return SIGNIFICANCE_DICT[significance_type][feature_type][target_type](
            **kwargs
        )
    except KeyError:
        raise ValueError(
            f"Unsupported significance type '{significance_type}' for feature "
            f"type '{feature_type}' and target type '{target_type}'"
        ) from None
