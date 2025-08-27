"""Significance tests based predictiveness scores."""

from multiprocessing import Pool
from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validate_call
from pydantic.config import ConfigDict
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from cc_tk.relationship.significance.base import (
    SignificanceCategoricalCategorical,
    SignificanceCategoricalNumeric,
    SignificanceNumericCategorical,
    SignificanceNumericNumeric,
    SignificanceOutput,
)


@validate_call(config={"arbitrary_types_allowed": True})
def estimate_pvalue(
    statistic_value: float,
    statistic_array: np.ndarray,
    kind: Literal["both", "left", "right"] = "both",
) -> float:
    """Estimate the pvalue of a statistic using empirical bootstrap approach.

    Parameters
    ----------
    statistic_value : float
        Value of the statistic
    statistic_array : np.ndarray
        Array of bootstrap statistics under null hypothesis
    kind : Literal["both", "left", "right"], optional
        Tail-kind of the test:
            - "both" : two-sided test
            - "left" : one-sided test, left tail
            - "right" : one-sided test, right tail
        , by default "both".

    Returns
    -------
    float
        Empirical pvalue estimation

    Notes
    -----
    This function calculates the empirical p-value by counting the proportion
    of bootstrap statistics that are more extreme than the observed statistic.
    A continuity correction is applied by adding 1 to both numerator and
    denominator to avoid p-values of exactly 0.

    """
    n_bootstrap = len(statistic_array)
    
    if kind == "left":
        # Left tail: count how many bootstrap stats are <= observed statistic
        n_extreme = np.sum(statistic_array <= statistic_value)
    elif kind == "right":
        # Right tail: count how many bootstrap stats are >= observed statistic
        n_extreme = np.sum(statistic_array >= statistic_value)
    else:  # kind == "both"
        # Two-sided: count how many bootstrap stats are more extreme in either direction
        # More extreme means further from the median than the observed statistic
        median_bootstrap = np.median(statistic_array)
        distance_observed = abs(statistic_value - median_bootstrap)
        distances_bootstrap = np.abs(statistic_array - median_bootstrap)
        n_extreme = np.sum(distances_bootstrap >= distance_observed)
    
    # Apply continuity correction to avoid p-value of exactly 0
    pvalue_estimation = (n_extreme + 1) / (n_bootstrap + 1)
    
    return pvalue_estimation


class PredictivenessSignificance(BaseModel):
    """Significance test based on predictiveness scores.

    Parameters
    ----------
    model : Union[RegressorMixin, ClassifierMixin]
        Model to use for the predictiveness score, depends on the type of the
        target. Advised to use a model that can handle non-linear relationships
        such as a tree-based model.
    scoring : Union[str, callable]
        Scoring function to use for the predictiveness score. It should be an
        increasing function, the higher the better.
    cv : int, optional
        Number of folds for the cross-validation, by default 5.
    n_bootstrap : int, optional
        Number of bootstrap samples to compute the pvalue, by default 100.
    n_sample : Optional[int], optional
        Number of samples to use for the predictiveness score, by default None
        will use all the samples.

    """

    model: Union[RegressorMixin, ClassifierMixin]
    scoring: str
    cv: int = 5
    n_bootstrap: int = 100
    n_sample: Optional[int] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def score_predictiveness(
        self,
        x: pd.Series,
        y: pd.Series,
    ) -> float:
        """Compute predictiveness score.

        Parameters
        ----------
        x : pd.Series
            Feature values
        y : pd.Series
            Target values

        Returns
        -------
        float
            Predictiveness score

        """
        score = cross_val_score(
            self.model,
            x.values.reshape(-1, 1),
            y.values,
            scoring=self.scoring,
            cv=self.cv,
        ).mean()
        if np.isnan(score):
            raise ValueError(
                "The predictiveness score is NaN. This may be due to too few "
                "data points for cross-validation to work correctly."
            )
        return score

    def _evaluate(
        self, feature_values: pd.Series, target_values: pd.Series
    ) -> SignificanceOutput:
        """Compute the predictiveness significance.

        Parameters
        ----------
        feature_values : pd.Series
            Feature values
        target_values : pd.Series
            Target values

        Returns
        -------
        float
            Pvalue estimation

        """
        sampled_feature_values, sampled_target_values = self._sample_values(
            feature_values, target_values
        )
        predictiveness_score = self.score_predictiveness(
            sampled_feature_values, sampled_target_values
        )

        bootstrap_scores = self._compute_bootstrap_scores(
            sampled_feature_values, sampled_target_values
        )
        pvalue = estimate_pvalue(
            predictiveness_score, bootstrap_scores, kind="right"
        )
        output = SignificanceOutput(
            pvalue=pvalue,
            influence=pd.Series(),
            statistic=predictiveness_score,
        )
        return output

    def _sample_values(
        self, feature_values: pd.Series, target_values: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """Sample the feature and target values.

        Parameters
        ----------
        feature_values : pd.Series
            Feature values
        target_values : pd.Series
            Target values

        Returns
        -------
        Tuple[pd.Series, pd.Series]
            Sampled feature and target values

        Notes
        -----
        If n_sample is None, the function will return the feature and target

        """
        if self.n_sample is None:
            return feature_values, target_values
        sampled_index = np.random.choice(
            target_values.index, self.n_sample, replace=False
        )
        sampled_feature_values = feature_values.loc[sampled_index]
        sampled_target_values = target_values.loc[sampled_index]
        return sampled_feature_values, sampled_target_values

    def _compute_bootstrap_scores(
        self, feature_values: pd.Series, target_values: pd.Series
    ) -> np.ndarray:
        """Compute the bootstrap scores.

        Parameters
        ----------
        feature_values : pd.Series
            Feature values
        target_values : pd.Series
            Target values

        Returns
        -------
        np.ndarray
            Bootstrap scores

        Notes
        -----
        The function uses a pool to parallelize the computation of the
        bootstrap scores.

        """
        with Pool() as pool:
            bootstrap_scores = np.array(
                pool.map(
                    self._wrapper_score_predictiveness,
                    [
                        (
                            feature_values,
                            target_values.sample(frac=1, replace=False),
                        )
                        for _ in range(self.n_bootstrap)
                    ],
                )
            )

        return bootstrap_scores

    def _wrapper_score_predictiveness(self, args):
        return self.score_predictiveness(*args)


class PredictivenessNumericNumeric(
    PredictivenessSignificance, SignificanceNumericNumeric
):
    """Predictiveness numeric/numeric significance test."""

    __doc__ = __doc__ + "\n" + PredictivenessSignificance.__doc__
    model: RegressorMixin = Field(
        default_factory=lambda: DecisionTreeRegressor(max_depth=5)
    )
    scoring: str = "r2"


class PredictivenessCategoricalNumeric(
    PredictivenessSignificance, SignificanceCategoricalNumeric
):
    """Predictiveness categorical/numeric significance test."""

    __doc__ = __doc__ + "\n" + PredictivenessSignificance.__doc__
    model: RegressorMixin = Field(
        default_factory=lambda: make_pipeline(
            OneHotEncoder(), DecisionTreeRegressor(max_depth=5)
        )
    )
    scoring: str = "r2"


class PredictivenessNumericCategorical(
    PredictivenessSignificance, SignificanceNumericCategorical
):
    """Predictiveness numeric/categorical significance test."""

    __doc__ = __doc__ + "\n" + PredictivenessSignificance.__doc__
    model: ClassifierMixin = Field(
        default_factory=lambda: DecisionTreeClassifier(max_depth=5)
    )
    scoring: str = "roc_auc"


class PredictivenessCategoricalCategorical(
    PredictivenessSignificance, SignificanceCategoricalCategorical
):
    """Predictiveness categorical/categorical significance test."""

    __doc__ = __doc__ + "\n" + PredictivenessSignificance.__doc__
    model: ClassifierMixin = Field(
        default_factory=lambda: make_pipeline(
            OneHotEncoder(), DecisionTreeClassifier(max_depth=5)
        )
    )
    scoring: str = "roc_auc"
