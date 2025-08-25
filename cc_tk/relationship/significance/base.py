"""Evaluate the significance of the relationship between 2 variables.

Usually this consists in evaluating the relationship between a feature and
the target variable.
"""

from abc import ABC, abstractmethod
from enum import Enum, unique

import pandas as pd
from pydantic import BaseModel, ConfigDict, validate_call

from cc_tk.relationship.schema import (
    SeriesType,
    check_input_index,
    check_input_types,
)


class Constants:
    """Constants for the relationship functions."""

    WEAK_THRESHOLD = 0.1
    STRONG_THRESHOLD = 0.05


class VariableType(str, Enum):
    """Defines the type of a variable."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


class SignificanceType(str, Enum):
    """Defines the type of significance to compute."""

    STATISTICAL = "statistical"
    PREDICTIVENESS = "predictiveness"


@unique
class SignificanceEnum(str, Enum):
    """Defines the significance levels."""

    WEAK_VALUE = "weak"
    MEDIUM_VALUE = "medium"
    STRONG_VALUE = "strong"


class SignificanceOutput(BaseModel):
    """Output of the significance functions."""

    pvalue: float
    influence: pd.Series
    statistic: float
    message: str = ""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def significance(self) -> SignificanceEnum:
        """Computing significativity based on pvalue."""
        significance = SignificanceEnum.WEAK_VALUE
        if self.pvalue < Constants.WEAK_THRESHOLD:
            significance = SignificanceEnum.MEDIUM_VALUE
        if self.pvalue < Constants.STRONG_THRESHOLD:
            significance = SignificanceEnum.STRONG_VALUE

        return significance

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the output to a dataframe."""
        return pd.DataFrame(
            {
                "influence": self.influence,
                "pvalue": self.pvalue,
                "statistic": self.statistic,
                "message": self.message,
                "significance": self.significance.value,
            }
        )


class SignificanceNumericNumeric(ABC):
    """Base class for the numeric/numeric significance."""

    @check_input_types(
        ("numeric_values_1", SeriesType.NUMERIC),
        ("numeric_values_2", SeriesType.NUMERIC),
    )
    @check_input_index("numeric_values_1", "numeric_values_2")
    @validate_call(config={"arbitrary_types_allowed": True})
    def __call__(
        self, numeric_values_1: pd.Series, numeric_values_2: pd.Series
    ) -> SignificanceOutput:
        """Numeric/numeric significance with input checks.

        Parameters
        ----------
        numeric_values_1 : pd.Series
            First numeric values
        numeric_values_2 : pd.Series
            Second numeric values

        Returns
        -------
        SignificanceOutput
            Output of the significance function

        """
        return self._evaluate(numeric_values_1, numeric_values_2)

    @abstractmethod
    def _evaluate(
        self, numeric_values_1: pd.Series, numeric_values_2: pd.Series
    ) -> SignificanceOutput:
        """Actual significance computation."""
        pass


class SignificanceNumericCategorical(ABC):
    """Base class for the numeric/categorical significance."""

    @check_input_types(
        ("numeric_values", SeriesType.NUMERIC),
        ("categorical_values", SeriesType.CATEGORICAL),
    )
    @check_input_index("numeric_values", "categorical_values")
    @validate_call(config={"arbitrary_types_allowed": True})
    def __call__(
        self, numeric_values: pd.Series, categorical_values: pd.Series
    ) -> SignificanceOutput:
        """Numeric/categorical significance with input checks.

        Parameters
        ----------
        numeric_values : pd.Series
            Numeric values
        categorical_values : pd.Series
            Categorical values

        Returns
        -------
        SignificanceOutput
            Output of the significance function

        """
        return self._evaluate(numeric_values, categorical_values)

    @abstractmethod
    def _evaluate(
        self, numeric_values: pd.Series, categorical_values: pd.Series
    ) -> SignificanceOutput:
        """Actual significance computation."""
        pass


class SignificanceCategoricalNumeric(ABC):
    """Base class for the categorical/numeric significance."""

    @check_input_types(
        ("categorical_values", SeriesType.CATEGORICAL),
        ("numeric_values", SeriesType.NUMERIC),
    )
    @check_input_index("categorical_values", "numeric_values")
    @validate_call(config={"arbitrary_types_allowed": True})
    def __call__(
        self, categorical_values: pd.Series, numeric_values: pd.Series
    ) -> SignificanceOutput:
        """Categorical/numeric significance with input checks.

        Parameters
        ----------
        categorical_values : pd.Series
            Categorical values
        numeric_values : pd.Series
            Numeric values

        Returns
        -------
        SignificanceOutput
            Output of the significance function

        """
        return self._evaluate(categorical_values, numeric_values)

    @abstractmethod
    def _evaluate(
        self, categorical_values: pd.Series, numeric_values: pd.Series
    ) -> SignificanceOutput:
        """Actual significance computation."""
        pass


class SignificanceCategoricalCategorical(ABC):
    """Base class for the categorical/categorical significance."""

    @check_input_types(
        ("categorical_values_1", SeriesType.CATEGORICAL),
        ("categorical_values_2", SeriesType.CATEGORICAL),
    )
    @check_input_index("categorical_values_1", "categorical_values_2")
    @validate_call(config={"arbitrary_types_allowed": True})
    def __call__(
        self, categorical_values_1: pd.Series, categorical_values_2: pd.Series
    ) -> SignificanceOutput:
        """Categorical/categorical significance with input checks.

        Parameters
        ----------
        categorical_values_1 : pd.Series
            First categorical values
        categorical_values_2 : pd.Series
            Second categorical values

        Returns
        -------
        SignificanceOutput
            Output of the significance function

        """
        return self._evaluate(categorical_values_1, categorical_values_2)

    @abstractmethod
    def _evaluate(
        self, categorical_values_1: pd.Series, categorical_values_2: pd.Series
    ) -> SignificanceOutput:
        """Actual significance computation."""
        pass
