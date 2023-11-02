"""Scikit-learn like estimators to deal with correlation in variables."""
import logging
from collections import defaultdict
from typing import Any, Dict, List, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from cc_tk.util.types import ArrayLike1D, ArrayLike2D

logger = logging.getLogger(__name__)


# pylint: disable=W0201
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
        features: ArrayLike2D,
        y: ArrayLike1D,
    ) -> "CorrelationToTarget":
        """
        Fit the transformer to the data.

        Parameters
        ----------
        features : ArrayLike2D
            The features.
        y : ArrayLike1D
            The target.
        """
        features_, y = check_X_y(features, y, y_numeric=True)
        self.n_features_in_ = features_.shape[1]
        self._corr = np.corrcoef(features_.T, y)[-1, :-1]
        self.mask_selection_ = abs(self._corr) > self.threshold
        if self.mask_selection_.sum() == 0:
            logger.warning(
                "Threshold %s is too high, no columns should "
                "have been selected. Selecting columns with highest "
                "correlation.",
                self.threshold,
            )
            self.mask_selection_ = abs(self._corr) == abs(self._corr).max()
        if isinstance(features, pd.DataFrame):
            self._columns = features.columns
        else:
            self._columns = np.arange(features_.shape[1])
        self._selected_columns = self._columns[self.mask_selection_]

        return self

    # pylint: disable=W0613
    def transform(
        self, features: ArrayLike2D, y: ArrayLike1D = None
    ) -> ArrayLike2D:
        """Retrieve only the selected columns.

        Parameters
        ----------
        features : ArrayLike2D
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
            If the number of columns in features is different from the number of
            columns in the training data.
        """
        check_is_fitted(self, ["mask_selection_", "n_features_in_"])
        features = check_array(features)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                "Shape of input is different from what was seen in `fit`"
            )
        return features[:, self.mask_selection_]

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


# pylint: disable=W0201
class ClusteringCorrelation(BaseEstimator, TransformerMixin):
    """Scikit-learn like estimator to deal with group of correlated features."""

    def __init__(
        self,
        threshold: float = 0.1,
        summary_method: Literal["first", "pca"] = "first",
        n_variables_by_cluster: int = 1,
    ) -> None:
        """
        Initialize the Feature selector based on Clustering of correlations
        https://kobia.fr/automatiser-la-reduction-des-correlations-par-clustering/

        Parameters
        ----------
        threshold : float, optional
            Correlation threshold to consider that a group of variables
            are all correlated together, by default 0.1
            0.1 means that all variables in the same cluster have a correlation
            of less than 0.1
        summary_method : str, optional
            Method to summarize each cluster of variables, implemented methods are:
            - "first" = keep only first variable
            - "pca" = performs principal component analysis to keep only the first
                component
            , by default "first"
        n_variables_by_cluster : int, optional
            Number of variables to extract by cluster, by default 1
        """
        self.threshold = threshold
        assert summary_method in ["first", "pca"]
        self.summary_method = summary_method
        self.n_variables_by_cluster = n_variables_by_cluster

    def fit(self, features: pd.DataFrame, y: pd.Series = None):
        """
        Fit the feature selection to features

        Parameters
        ----------
        features : pd.DataFrame
            Features to fit the feature selection to
        y : pd.Series, optional
            Target, by default None
        """
        features_, y = check_X_y(features, y, ensure_min_features=2)
        self.n_features_in_ = features_.shape[1]
        if isinstance(features, pd.DataFrame):
            self._columns = features.columns
        else:
            self._columns = np.arange(features_.shape[1])
        # Computing correlation
        self._corr = np.corrcoef(features_.T)
        self._corr = np.nan_to_num(self._corr)
        # Symmetrizing correlation matrix
        self._corr = (self._corr + self._corr.T) / 2
        # Filling diagonal with 1
        np.fill_diagonal(self._corr, 1.0)

        # Computing distance matrix
        dist = squareform(1 - abs(self._corr)).round(6)

        # Clustering with complete linkage
        self._corr_linkage = hierarchy.complete(dist)

        self.clusters_col_ = self.get_clusters(self._corr_linkage)

        if self.summary_method == "first":
            self._selected_columns_ = [
                cluster[i]
                for cluster in self.clusters_col_
                for i in range(self.n_variables_by_cluster)
                if i < len(cluster)
            ]
            self.mask_selection_ = np.isin(
                self._columns, self._selected_columns_
            )

        elif self.summary_method == "pca":
            self.pca_by_cluster_ = [
                PCA(
                    n_components=min(len(cluster), self.n_variables_by_cluster)
                ).fit(features_[:, np.isin(self._columns, cluster)])
                for cluster in self.clusters_col_
            ]
            self._output_columns = [
                [
                    f"{'-'.join(map(str, cluster))} {i}"
                    for i in range(pca.n_components_)
                ]
                for pca, cluster in zip(
                    self.pca_by_cluster_, self.clusters_col_
                )
            ]

        return self

    # pylint: disable=W0613
    def transform(
        self, features: pd.DataFrame, y: pd.Series = None
    ) -> pd.DataFrame:
        """
        Apply feature selection to DataFrame features and return the
        transformed variables

        Parameters
        ----------
        features : pd.DataFrame
            Features
        y : pd.Series, optional
            Target, by default None

        Returns
        -------
        pd.DataFrame
            Transformed features with feature selection
        """
        features_ = check_array(features)
        check_is_fitted(self, ["clusters_col_", "n_features_in_"])
        if features_.shape[1] != self.n_features_in_:
            raise ValueError(
                "Shape of input is different from what was seen in `fit`"
            )
        if self.summary_method == "first":
            check_is_fitted(self, ["mask_selection_"])
            return features_[:, self.mask_selection_]
        if self.summary_method == "pca":
            check_is_fitted(self, ["pca_by_cluster_"])
            features_by_cluster = []
            for pca, cluster, pca_output_columns in zip(
                self.pca_by_cluster_, self.clusters_col_, self._output_columns
            ):
                assert all(
                    map(
                        lambda value: str(value) in pca_output_columns[0],
                        cluster,
                    )
                )
                pca_output = pca.transform(
                    features_[:, np.isin(self._columns, cluster)]
                )
                if isinstance(pca_output, pd.DataFrame):
                    pca_output.columns = pca_output_columns
                else:
                    pca_output = pd.DataFrame(
                        pca_output,
                        columns=pca_output_columns,
                    )
                features_by_cluster.append(pca_output)
            features_transform = pd.concat(features_by_cluster, axis=1)
            if isinstance(features, pd.DataFrame):
                features_transform.index = features.index
                return features_transform
            return features_transform.values
        return features

    def plot_dendro(self, ax: plt.Axes = None) -> Dict[str, Any]:
        """
        Plot dendrogram of the correlation matrix.

        Parameters
        ----------
        ax : plt.Axes, optional
            Axis to plot the dendrogram on, by default None

        Returns
        -------
        Dict[str, Any]
            Dendrogram object
        """
        self.dendro = hierarchy.dendrogram(
            self._corr_linkage,
            orientation="right",
            labels=self._columns,
            color_threshold=self.threshold,
            ax=ax,
        )
        return self.dendro

    def plot_correlation_matrix(
        self, fig=None, ax: plt.Axes = None
    ) -> plt.Axes:
        """
        Plot correlation matrix of the features.

        Parameters
        ----------
        fig : plt.Figure, optional
            Figure to plot the correlation matrix on, by default None
        ax : plt.Axes, optional
            Axis to plot the correlation matrix on, by default None

        Returns
        -------
        plt.Axes
            Axis with the correlation matrix"""
        if ax is None:
            fig = plt.gcf()
            ax = plt.gca()
        plot = ax.pcolor(
            abs(self._corr[self.dendro["leaves"], :][:, self.dendro["leaves"]])
        )
        dendro_idx = np.arange(0, len(self.dendro["ivl"]))
        ax.set_xticks(dendro_idx)
        ax.set_yticks(dendro_idx)
        ax.set_xticklabels(self.dendro["ivl"], rotation="vertical")
        ax.set_yticklabels(self.dendro["ivl"])

        fig.colorbar(plot, format=ticker.PercentFormatter(xmax=1))
        return ax

    def get_clusters(self, linkage: np.ndarray) -> List[List[str]]:
        """
        Retrieves the cluster of variables given a specific threshold

        Parameters
        ----------
        linkage : np.ndarray
            Linkage matrix from scipy.cluster.hierarchy

        Returns
        -------
        List[List[str]]
            List of lists of variable names according to each cluster
        """
        # Récupération des clusters à partir de la hiérarchie
        cluster_ids = hierarchy.fcluster(
            linkage, self.threshold, criterion="distance"
        )
        # Assignation des index de chaque variable dans un dictionnaire
        cluster_id_to_feature_ids = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)
        # Récupération de la liste des clusters (indices des variables)
        clusters = [
            list(v) for v in cluster_id_to_feature_ids.values() if len(v) > 0
        ]
        # Récupération de la liste des clusters (noms des variables)
        clusters_col = [list(self._columns[v]) for v in clusters]

        return clusters_col
