import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from cc_tk.util.types import ArrayLike1D, ArrayLike2D

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
        X: ArrayLike2D,
        y: ArrayLike1D,
    ) -> "CorrelationToTarget":
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X : ArrayLike2D
            The features.
        y : ArrayLike1D
            The target.
        """
        X_, y = check_X_y(X, y, y_numeric=True)
        self.n_features_in_ = X_.shape[1]
        self._corr = np.corrcoef(X_.T, y)[-1, :-1]
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
            self._columns = X.columns
        else:
            self._columns = np.arange(X_.shape[1])
        self._selected_columns = self._columns[self.mask_selection_]

        return self

    def transform(self, X: ArrayLike2D, y: ArrayLike1D = None) -> ArrayLike2D:
        """Retrieve only the selected columns.

        Parameters
        ----------
        X : ArrayLike2D
            The features.
        y : ArrayLike1D, optional
            The target, by default None

        Returns
        -------
        ArrayLike2D
            The selected features.

        Raises
        ------
        ValueError
            If the number of columns in X is different from the number of
            columns in the training data.
        """
        check_is_fitted(self, ["mask_selection_", "n_features_in_"])
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "Shape of input is different from what was seen" "in `fit`"
            )
        return X[:, self.mask_selection_]

    def plot_correlation(self):
        """Plot the correlation of each feature to the target.

        The selected features are highlighted in green, the others in red.
        The threshold values are indicated with dashed lines.
        """
        check_is_fitted(self, ["mask_selection_", "n_features_in_"])
        plot_df = pd.DataFrame(
            {
                "Correlation": self._corr,
                "Columns": self._columns,
                "Selected": self.mask_selection_,
            }
        )
        plot_df = plot_df.sort_values("Correlation")
        ax = plot_df.plot.barh(
            x="Columns",
            y="Correlation",
            color=plot_df["Selected"].map(
                {True: "tab:green", False: "tab:red"}
            ),
        )
        ax.vlines(
            [-self.threshold, self.threshold],
            ymin=-1,
            ymax=len(plot_df),
            colors="k",
            linestyles="dashed",
        )
        ax.set_xlabel("Correlation to target")
        ax.legend().remove()
