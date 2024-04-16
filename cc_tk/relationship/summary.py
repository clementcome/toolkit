"""Builds relationship summary."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel
from pydantic.config import ConfigDict

from cc_tk.relationship import distribution
from cc_tk.relationship.significance import (
    SignificanceType,
    VariableType,
    get_significance,
)


class SummaryOutput(BaseModel):
    """Output of the relationship summary.

    Parameters
    ----------
    numeric_distribution : pd.DataFrame
        The numeric distribution.
    categorical_distribution : pd.DataFrame
        The categorical distribution.
    numeric_significance : pd.DataFrame
        The numeric significance.
    categorical_significance : pd.DataFrame
        The categorical significance.

    """

    numeric_distribution: pd.DataFrame
    categorical_distribution: pd.DataFrame
    numeric_significance: pd.DataFrame
    categorical_significance: pd.DataFrame

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_excel(self, path: str) -> None:
        """Write the summary to an Excel file.

        Parameters
        ----------
        path : str
            Path to the Excel file.

        """
        with pd.ExcelWriter(path) as writer:
            for name, df in self.model_dump().items():
                df.to_excel(writer, sheet_name=name, merge_cells=False)


class RelationshipSummary:
    """Builds the relationship summary.

    Parameters
    ----------
    features : pd.DataFrame
        The input features.
    target : pd.Series
        The target variable.
    significance_type : Optional[SignificanceType]
        The type of significance to compute, by default "statistical".

    Attributes
    ----------
    features : pd.DataFrame
        The input features.
    target : pd.Series
        The target variable.
    numeric_features : pd.DataFrame
        The numeric features extracted from the input features.
    categorical_features : pd.DataFrame
        The categorical features extracted from the input features.
    summary_output : Optional[SummaryOutput]
        The summary output of the relationship summary.

    Methods
    -------
    build_summary() -> SummaryOutput:
        Build the relationship summary.
    to_excel(path: str) -> None:
        Write the summary to an Excel file.

    Private Methods
    ---------------
    _build_numeric_distribution() -> pd.DataFrame:
        Build the numeric distribution.
    _build_categorical_distribution() -> pd.DataFrame:
        Build the categorical distribution.
    _build_numeric_significance() -> pd.DataFrame:
        Build the numeric significance.
    _build_categorical_significance() -> pd.DataFrame:
        Build the categorical significance.
    _build_distribution_by_target() -> Tuple[pd.DataFrame, pd.DataFrame]:
        Build the distribution by target.

    """

    def __init__(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        significance_type: SignificanceType = SignificanceType.STATISTICAL,
    ):
        """Initialize the relationship summary.

        Parameters
        ----------
        features : pd.DataFrame
            The input features.
        target : pd.Series
            The target variable.
        significance_type : SignificanceType, optional
            The type of significance to compute, by default "statistical".

        """
        self.features = features
        self.target = target
        self.target_type = (
            VariableType.NUMERIC
            if pd.api.types.is_numeric_dtype(self.target)
            else VariableType.CATEGORICAL
        )
        self.significance_type = significance_type
        self.numeric_features = self.features.select_dtypes(
            include=[np.number]
        )
        self.categorical_features = self.features.select_dtypes(
            exclude=[np.number]
        )
        self.summary_output: Optional[SummaryOutput] = None
        self.numeric_significance_function = get_significance(
            VariableType.NUMERIC, self.target_type, self.significance_type
        )
        self.categorical_significance_function = get_significance(
            VariableType.CATEGORICAL, self.target_type, self.significance_type
        )

    def build_summary(self) -> SummaryOutput:
        """Build the relationship summary.

        Returns
        -------
        SummaryOutput
            Relationship summary.

        """
        (
            numeric_distribution_by_target_class,
            categorical_distribution_by_target_class,
        ) = self._build_distribution_by_target()
        self.summary_output = SummaryOutput(
            numeric_distribution=self._build_numeric_distribution(),
            categorical_distribution=self._build_categorical_distribution(),
            numeric_significance=pd.concat(
                [
                    self._build_numeric_significance(),
                    numeric_distribution_by_target_class,
                ],
                axis=1,
            ),
            categorical_significance=pd.concat(
                [
                    self._build_categorical_significance(),
                    categorical_distribution_by_target_class,
                ],
                axis=1,
            ),
        )
        return self.summary_output

    def to_excel(self, path: str) -> None:
        """Write the summary to an Excel file.

        Parameters
        ----------
        path : str
            Path to the Excel file.

        """
        if self.summary_output is None:
            self.build_summary()
        self.summary_output.to_excel(path)

    def _build_numeric_distribution(self) -> pd.DataFrame:
        return distribution.numeric_distribution(self.numeric_features)

    def _build_categorical_distribution(self) -> pd.DataFrame:
        return distribution.categorical_distribution(self.categorical_features)

    def _build_numeric_significance(self) -> pd.DataFrame:
        if self.numeric_features.empty:
            return pd.DataFrame()
        if pd.api.types.is_numeric_dtype(self.target):
            significance_df = pd.concat(
                {
                    feature_name: self.numeric_significance_function(
                        self.features[feature_name], self.target
                    ).to_dataframe()
                    for feature_name in self.numeric_features
                }
            ).reset_index(level=1, drop=True)
            significance_df.index.name = "Variable"
            return significance_df
        significance_df = pd.concat(
            {
                feature_name: self.numeric_significance_function(
                    self.features[feature_name], self.target
                ).to_dataframe()
                for feature_name in self.numeric_features
            }
        ).sort_index()
        significance_df.index.names = ["Variable", "Target"]
        return significance_df

    def _build_categorical_significance(self) -> pd.DataFrame:
        if self.categorical_features.empty:
            return pd.DataFrame()
        if pd.api.types.is_numeric_dtype(self.target):
            significance_df = pd.concat(
                {
                    feature_name: self.categorical_significance_function(
                        self.features[feature_name], self.target
                    ).to_dataframe()
                    for feature_name in self.categorical_features
                }
            ).sort_index()
            significance_df.index.names = ["Variable", "Value"]
            return significance_df
        significance_df = pd.concat(
            {
                feature_name: self.categorical_significance_function(
                    self.features[feature_name], self.target
                ).to_dataframe()
                for feature_name in self.categorical_features
            }
        ).sort_index()
        significance_df.index.names = ["Variable", "Target", "Value"]
        return significance_df

    def _build_distribution_by_target(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not pd.api.types.is_numeric_dtype(self.target):
            (
                numeric_distribution_by_target_class,
                categorical_distribution_by_target_class,
            ) = distribution.summary_distribution_by_target(
                self.features, self.target
            )
            return (
                numeric_distribution_by_target_class,
                categorical_distribution_by_target_class,
            )
        return pd.DataFrame(), pd.DataFrame()
