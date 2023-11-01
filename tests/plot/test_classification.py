import pandas as pd
import pytest
import seaborn as sns
from pytest_mock import MockerFixture

from cc_tk.plot.classification import plot_confusion


def test_plot_confusion(mocker: MockerFixture):
    # Create a dummy confusion matrix
    data = {
        "y_Actual": ["A", "B", "A", "B", "A", "B", "A", "B", "A"],
        "y_Predicted": ["A", "B", "B", "B", "A", "B", "A", "A", "A"],
    }

    df = pd.DataFrame(data, columns=["y_Actual", "y_Predicted"])
    confusion_matrix = pd.crosstab(
        df["y_Actual"],
        df["y_Predicted"],
        rownames=["Actual"],
        colnames=["Predicted"],
    )

    # Mock the seaborn heatmap function
    mock_heatmap = mocker.patch.object(sns, "heatmap", return_value=None)
    # Call the function with the dummy confusion matrix
    plot_confusion(confusion_matrix, fmt="d")

    # Assert that the heatmap function was called
    assert mock_heatmap.call_count == 3
