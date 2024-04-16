from unittest.mock import Mock

import pytest
from cc_tk.relationship.significance import (
    SignificanceCategoricalCategorical,
    SignificanceCategoricalNumeric,
    SignificanceNumericCategorical,
    SignificanceNumericNumeric,
    SignificanceType,
    VariableType,
    get_significance,
)
from pydantic import ValidationError
from pytest_mock import MockerFixture


class TestGetSignificance:
    @pytest.mark.parametrize(
        ("feature_type", "target_type", "significance_type", "kwargs", "expected_type"),
        [
            (
                VariableType.NUMERIC,
                VariableType.NUMERIC,
                SignificanceType.STATISTICAL,
                {},
                SignificanceNumericNumeric,
            ),
            (
                VariableType.NUMERIC,
                VariableType.CATEGORICAL,
                SignificanceType.STATISTICAL,
                {},
                SignificanceNumericCategorical,
            ),
            (
                VariableType.CATEGORICAL,
                VariableType.NUMERIC,
                SignificanceType.PREDICTIVENESS,
                {"n_bootstrap": 10},
                SignificanceCategoricalNumeric,
            ),
            (
                VariableType.CATEGORICAL,
                VariableType.CATEGORICAL,
                SignificanceType.PREDICTIVENESS,
                {"cv": 4},
                SignificanceCategoricalCategorical,
            ),
        ],
    )
    def test_get_significance(
        self, feature_type, target_type, significance_type, kwargs, expected_type
    ):
        significance = get_significance(
            feature_type=feature_type,
            target_type=target_type,
            significance_type=significance_type,
            **kwargs,
        )
        assert isinstance(significance, expected_type)
        if kwargs:
            for key, value in kwargs.items():
                assert getattr(significance, key) == value

    @pytest.mark.parametrize(
        "parameter",
        ["feature_type", "target_type", "significance_type"],
    )
    def test_get_significance_pydantic_validation(self, parameter):
        original_parameters = {
            "feature_type": VariableType.NUMERIC,
            "target_type": VariableType.NUMERIC,
            "significance_type": SignificanceType.STATISTICAL,
        }
        original_parameters[parameter] = "unsupported_type"

        with pytest.raises(ValidationError):
            get_significance(**original_parameters)

    def test_get_significance_missing_significance(self, mocker: MockerFixture):
        mock_significance = Mock()
        mocker.patch.dict(
            "cc_tk.relationship.significance.SIGNIFICANCE_DICT",
            {
                SignificanceType.STATISTICAL: {
                    VariableType.NUMERIC: {VariableType.NUMERIC: mock_significance}
                }
            },
            clear=True,
        )
        with pytest.raises(ValueError):
            get_significance(
                feature_type=VariableType.NUMERIC,
                target_type=VariableType.NUMERIC,
                significance_type=SignificanceType.PREDICTIVENESS,
            )
