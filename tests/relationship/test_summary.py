import pytest
from pytest_mock import MockerFixture
from cc_tk.relationship.summary import SummaryOutput, RelationshipSummary
import pandas as pd
import numpy as np
from cc_tk.relationship import distribution, significance


class TestSummaryOutput:
    def test_to_excel_writes_correct_sheets(self, mocker: MockerFixture, tmp_path):
        # Given
        # Create a SummaryOutput instance
        summary = SummaryOutput(
            numeric_distribution=pd.DataFrame(
                {"A": [1, 2, 3]}, index=pd.RangeIndex(3, name="index")
            ),
            categorical_distribution=pd.DataFrame(
                {"B": ["a", "b", "c"]}, index=pd.RangeIndex(3, name="index")
            ),
            numeric_significance=pd.DataFrame(
                {"C": [4, 5, 6]}, index=pd.RangeIndex(3, name="index")
            ),
            categorical_significance=pd.DataFrame(
                {"D": ["d", "e", "f"]}, index=pd.RangeIndex(3, name="index")
            ),
        )

        # Write to a temporary file using tmp_path
        file_path = tmp_path / "test_summary.xlsx"

        # When
        summary.to_excel(file_path)

        # Then
        # Assert that the workbook was saved successfully
        assert (tmp_path / "test_summary.xlsx").exists()

        # Read the written file
        written_data = pd.read_excel(file_path, sheet_name=None)

        # Assert that the correct sheets have been written
        assert set(written_data.keys()) == {
            "numeric_distribution",
            "categorical_distribution",
            "numeric_significance",
            "categorical_significance",
        }

        # Assert that the mock writer's write method was called with the correct arguments
        for sheet_name, df in summary.model_dump().items():
            pd.testing.assert_frame_equal(written_data[sheet_name], df.reset_index())


class TestRelationshipSummary:
    def test_build_summary_numeric_target(
        self, features: pd.DataFrame, target_numeric: pd.Series
    ):
        # Given
        relationship_summary = RelationshipSummary(features, target_numeric)

        # When
        summary = relationship_summary.build_summary()

        # Then
        assert isinstance(summary, SummaryOutput)
        assert isinstance(summary.numeric_distribution, pd.DataFrame)
        assert not summary.numeric_distribution.empty
        assert isinstance(summary.categorical_distribution, pd.DataFrame)
        assert not summary.categorical_distribution.empty
        assert isinstance(summary.numeric_significance, pd.DataFrame)
        assert not summary.numeric_significance.empty
        assert isinstance(summary.categorical_significance, pd.DataFrame)
        assert not summary.categorical_significance.empty

    def test_build_summary_categorical_target(
        self, features: pd.DataFrame, target_categorical: pd.Series
    ):
        # Given
        relationship_summary = RelationshipSummary(features, target_categorical)

        # When
        summary = relationship_summary.build_summary()

        # Then
        assert isinstance(summary, SummaryOutput)
        assert isinstance(summary.numeric_distribution, pd.DataFrame)
        assert not summary.numeric_distribution.empty
        assert isinstance(summary.categorical_distribution, pd.DataFrame)
        assert not summary.categorical_distribution.empty
        assert isinstance(summary.numeric_significance, pd.DataFrame)
        assert not summary.numeric_significance.empty
        assert "mean" in summary.numeric_significance.columns
        assert isinstance(summary.categorical_significance, pd.DataFrame)
        assert not summary.categorical_significance.empty
        assert "count" in summary.categorical_significance.columns

    def test_to_excel(
        self,
        mocker: MockerFixture,
        features: pd.DataFrame,
        target_categorical: pd.Series,
    ):
        # Given
        relationship_summary = RelationshipSummary(features, target_categorical)
        mocker.patch.object(SummaryOutput, "to_excel")

        # When
        relationship_summary.to_excel("test_path")

        # Then
        SummaryOutput.to_excel.assert_called_once_with("test_path")

    def test__build_numeric_distribution(
        self, features: pd.DataFrame, target_categorical: pd.Series
    ):
        # Given
        relationship_summary = RelationshipSummary(features, target_categorical)

        # When
        numeric_distribution = relationship_summary._build_numeric_distribution()

        # Then
        expected_numeric_distribution = distribution.numeric_distribution(
            relationship_summary.numeric_features
        )
        assert numeric_distribution.equals(expected_numeric_distribution)

    def test__build_categorical_distribution(
        self, features: pd.DataFrame, target_categorical: pd.Series
    ):
        # Given
        relationship_summary = RelationshipSummary(features, target_categorical)

        # When
        categorical_distribution = relationship_summary._build_categorical_distribution()

        # Then
        expected_categorical_distribution = distribution.categorical_distribution(
            relationship_summary.categorical_features
        )
        assert categorical_distribution.equals(expected_categorical_distribution)

    def test__build_numeric_significance_with_categorical_target(
        self, features: pd.DataFrame, target_categorical: pd.Series
    ):
        # Given
        relationship_summary = RelationshipSummary(features, target_categorical)

        # When
        numeric_significance = relationship_summary._build_numeric_significance()

        # Then
        assert isinstance(numeric_significance, pd.DataFrame)
        pd.testing.assert_index_equal(
            numeric_significance.index,
            pd.MultiIndex.from_tuples(
                [["num1", 0], ["num1", 1], ["num2", 0], ["num2", 1]],
                names=["Variable", "Target"],
            ),
        )
        pd.testing.assert_index_equal(
            numeric_significance.columns,
            pd.Index(["influence", "pvalue", "statistic", "message", "significance"]),
        )

    def test__build_numeric_significance_with_numeric_target(
        self, features: pd.DataFrame, target_numeric: pd.Series
    ):
        # Given
        relationship_summary = RelationshipSummary(features, target_numeric)

        # When
        numeric_significance = relationship_summary._build_numeric_significance()

        # Then
        assert isinstance(numeric_significance, pd.DataFrame)
        pd.testing.assert_index_equal(
            numeric_significance.index, pd.Index(["num1", "num2"], name="Variable")
        )
        pd.testing.assert_index_equal(
            numeric_significance.columns,
            pd.Index(["influence", "pvalue", "statistic", "message", "significance"]),
        )

    def test__build_numeric_significance_empty(
        self, features: pd.DataFrame, target_numeric: pd.Series
    ):
        # Given
        relationship_summary = RelationshipSummary(
            features.select_dtypes(exclude=[np.number]), target_numeric
        )

        # When
        numeric_significance = relationship_summary._build_numeric_significance()

        # Then
        assert isinstance(numeric_significance, pd.DataFrame)
        assert numeric_significance.empty

    def test__build_categorical_significance_with_categorical_target(
        self, features: pd.DataFrame, target_categorical: pd.Series
    ):
        # Given
        relationship_summary = RelationshipSummary(features, target_categorical)

        # When
        categorical_significance = relationship_summary._build_categorical_significance()

        # Then
        assert isinstance(categorical_significance, pd.DataFrame)
        pd.testing.assert_index_equal(
            categorical_significance.index,
            pd.MultiIndex.from_tuples(
                [
                    ["cat1", 0, "a"],
                    ["cat1", 0, "b"],
                    ["cat1", 0, "c"],
                    ["cat1", 1, "a"],
                    ["cat1", 1, "b"],
                    ["cat1", 1, "c"],
                    ["cat2", 0, "a"],
                    ["cat2", 0, "b"],
                    ["cat2", 0, "c"],
                    ["cat2", 1, "a"],
                    ["cat2", 1, "b"],
                    ["cat2", 1, "c"],
                ],
                names=["Variable", "Target", "Value"],
            ),
        )
        pd.testing.assert_index_equal(
            categorical_significance.columns,
            pd.Index(["influence", "pvalue", "statistic", "message", "significance"]),
        )

    def test__build_categorical_significance_with_numeric_target(
        self, features: pd.DataFrame, target_numeric: pd.Series
    ):
        # Given
        relationship_summary = RelationshipSummary(features, target_numeric)

        # When
        categorical_significance = relationship_summary._build_categorical_significance()

        # Then
        assert isinstance(categorical_significance, pd.DataFrame)
        pd.testing.assert_index_equal(
            categorical_significance.index,
            pd.MultiIndex.from_tuples(
                [
                    ["cat1", "a"],
                    ["cat1", "b"],
                    ["cat1", "c"],
                    ["cat2", "a"],
                    ["cat2", "b"],
                    ["cat2", "c"],
                ],
                names=["Variable", "Value"],
            ),
        )
        pd.testing.assert_index_equal(
            categorical_significance.columns,
            pd.Index(["influence", "pvalue", "statistic", "message", "significance"]),
        )

    def test__build_categorical_significance_empty(
        self, features: pd.DataFrame, target_numeric: pd.Series
    ):
        # Given
        relationship_summary = RelationshipSummary(
            features.select_dtypes(include=[np.number]), target_numeric
        )

        # When
        categorical_significance = relationship_summary._build_categorical_significance()

        # Then
        assert isinstance(categorical_significance, pd.DataFrame)
        assert categorical_significance.empty

    def test__build_distribution_by_target_numeric(
        self, features: pd.DataFrame, target_numeric: pd.Series
    ):
        # Given
        relationship_summary = RelationshipSummary(features, target_numeric)

        # When
        distribution_by_target_numeric = (
            relationship_summary._build_distribution_by_target()
        )

        # Then
        for output in distribution_by_target_numeric:
            assert isinstance(output, pd.DataFrame)
            assert output.empty

    def test__build_distribution_by_target_categorical(
        self, features: pd.DataFrame, target_categorical: pd.Series
    ):
        # Given
        relationship_summary = RelationshipSummary(features, target_categorical)

        # When
        distribution_by_target_categorical = (
            relationship_summary._build_distribution_by_target()
        )

        # Then
        (
            expected_numeric_distribution_by_target_class,
            expected_categorical_distribution_by_target_class,
        ) = distribution.summary_distribution_by_target(features, target_categorical)
        pd.testing.assert_frame_equal(
            distribution_by_target_categorical[0],
            expected_numeric_distribution_by_target_class,
        )
        pd.testing.assert_frame_equal(
            distribution_by_target_categorical[1],
            expected_categorical_distribution_by_target_class,
        )
