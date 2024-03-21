"""Builds relationship summary."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel
from pydantic.config import ConfigDict

from cc_tk.relationship import distribution, significance


class SummaryOutput(BaseModel):
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
                df.to_excel(writer, sheet_name=name)


class RelationshipSummary:
    """Class for building and summarizing the relationship between features and
    target variable.

    Parameters
    ----------
    features : pd.DataFrame
        The input features.
    target : pd.Series
        The target variable.

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

    def __init__(self, features: pd.DataFrame, target: pd.Series):
        self.features = features
        self.target = target
        self.numeric_features = self.features.select_dtypes(include=[np.number])
        self.categorical_features = self.features.select_dtypes(exclude=[np.number])
        self.summary_output: Optional[SummaryOutput] = None

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
                    feature_name: significance.significance_numeric_numeric(
                        self.features[feature_name], self.target
                    ).to_dataframe()
                    for feature_name in self.numeric_features
                }
            ).reset_index(level=1, drop=True)
            significance_df.index.name = "Variable"
            return significance_df
        significance_df = pd.concat(
            {
                feature_name: significance.significance_numeric_categorical(
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
                    feature_name: significance.significance_numeric_categorical(
                        self.target, self.features[feature_name]
                    ).to_dataframe()
                    for feature_name in self.categorical_features
                }
            ).sort_index()
            significance_df.index.names = ["Variable", "Value"]
            return significance_df
        significance_df = pd.concat(
            {
                feature_name: significance.significance_categorical_categorical(
                    self.features[feature_name], self.target
                ).to_dataframe()
                for feature_name in self.categorical_features
            }
        ).sort_index()
        significance_df.index.names = ["Variable", "Target", "Value"]
        return significance_df

    def _build_distribution_by_target(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not pd.api.types.is_numeric_dtype(self.target):
            (
                numeric_distribution_by_target_class,
                categorical_distribution_by_target_class,
            ) = distribution.summary_distribution_by_target(self.features, self.target)
            return (
                numeric_distribution_by_target_class,
                categorical_distribution_by_target_class,
            )
        return pd.DataFrame(), pd.DataFrame()
