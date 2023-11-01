"""Classification plots."""
from typing import Literal, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import ConfigDict, validate_call


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def plot_confusion(
    confusion_matrix: pd.DataFrame,
    fmt: str,
    near_diag: int = 1,
    vrange: Union[Literal["global", "local"], Tuple[float]] = "global",
) -> None:
    """
    Plot a confusion matrix with green, blue and red color scales.

    Parameters:
    -----------
    confusion_matrix : pd.DataFrame
        The confusion matrix to plot.
    fmt : str
        The format string for the annotations in the heatmap.
    near_diag : int, optional
        The number of cells near the diagonal to highlight with a different
        color.
        Default is 1, which corresponds to the cells immediately adjacent to
        the diagonal.
        A value of 0 will include the diagonal cells, and a value of -1 will
        disable the near-diagonal cells.
    vrange : Union[Literal['global', 'local'], Tuple[float, float]], optional
        The range of values to use for the color scale.
        If 'global', the range is set to (0, maximum value of the entire
        confusion matrix).
        If 'local', the range is set to (minimum value of each heatmap,
        maximum value of each heatmap).
        If a tuple, the range is set to the given values.
        Default is 'global'.
    """
    n = confusion_matrix.shape[0]

    # Create a mask to separate diagonal, near-diagonal and off-diagonal cells
    if near_diag == 0:
        mask_neardiag = np.zeros((n, n), dtype=bool)
    else:
        mask_neardiag = ~(
            np.eye(n, k=near_diag, dtype=bool)
            + np.eye(n, k=-near_diag, dtype=bool)
        )
    mask_diag = ~np.eye(n, dtype=bool)
    mask_offdiag = ~(mask_diag & mask_neardiag)

    # Create a diverging color palette for the diagonal cells (green),
    # the near-diagonal cells (blue) and the off-diagonal cells (red)
    cmap_diag = sns.color_palette("Greens")
    cmap_neardiag = sns.color_palette("Blues")
    cmap_offdiag = sns.color_palette("Reds")

    # Set vmin and vmax based on the input parameters
    if vrange == "global":
        vmin, vmax = 0, confusion_matrix.max().max()
    elif vrange == "local":
        vmin, vmax = None, None
    else:
        vmin, vmax = vrange

    # Plot the heatmaps
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=fmt,
        cmap=cmap_diag,
        mask=mask_diag,
        cbar=False,
        vmin=vmin,
        vmax=vmax,
    )
    if near_diag != 0:
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt=fmt,
            cmap=cmap_neardiag,
            mask=mask_neardiag,
            cbar=False,
            vmin=vmin,
            vmax=vmax,
        )
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=fmt,
        cmap=cmap_offdiag,
        mask=mask_offdiag,
        cbar=False,
        vmin=vmin,
        vmax=vmax,
    )
