import logging
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

logger = logging.getLogger(__name__)


class SelectionException(Exception):
    pass


class CorrelationToTarget(BaseEstimator, TransformerMixin):
    """
    Select columns with correlation to target above a threshold.

    Parameters
    ----------
    threshold : float, optional
        The threshold for the correlation to the target.
        Default is 0.1.
    """

    def __init__(self, threshold: float = 0.1) -> None:
        super().__init__()
        self.threshold = threshold

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ) -> "CorrelationToTarget":
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : Union[np.ndarray, pd.DataFrame]
            The features.
        y : Union[np.ndarray, pd.Series]
            The target.
        """
        X, y = check_X_y(X, y, y_numeric=True)
        self.n_features_in_ = X.shape[1]
        self._corr = np.corrcoef(X.T, y)[-1, :-1]
        # self.corr = X.corrwith(y)
        self.mask_selection_ = abs(self._corr) > self.threshold
        if self.mask_selection_.sum() == 0:
            logger.warning(
                f"Threshold {self.threshold} is too high, no columns should "
                "have been selected. Selecting columns with highest "
                "correlation."
            )
            self.mask_selection_ = abs(self._corr) == abs(self._corr).max()
        if isinstance(X, pd.DataFrame):
            self._selected_columns = self._corr[self.mask_selection_].index
        else:
            self._selected_columns = np.arange(X.shape[1])[
                self.mask_selection_
            ]

        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        check_is_fitted(self, ["mask_selection_", "n_features_in_"])
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "Shape of input is different from what was seen" "in `fit`"
            )
        return X[:, self.mask_selection_]

    def plot_correlation(self):
        check_is_fitted(self, ["mask_selection_", "n_features_in_"])
        plot_df = pd.Series(self._corr, name="Correlation").to_frame()
        plot_df["Selected"] = self.mask_selection_
        plot_df = plot_df.sort_values("Correlation")
        plot_df.plot.barh(y="Correlation")
