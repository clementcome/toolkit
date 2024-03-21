import numpy as np
import pandas as pd
import pandera as pa
import pytest
from scipy import stats

from cc_tk.relationship.significance import (
    SignificanceEnum,
    SignificanceOutput,
    _compute_group_info,
    significance_categorical_categorical,
    significance_numeric_categorical,
    significance_numeric_numeric,
)


class TestSignificanceNumericNumeric:
    @pytest.fixture
    def valid_numeric_values(self):
        return pd.Series([1, 2, 3, 4, 5])

    @pytest.fixture
    def invalid_numeric_values(self):
        return pd.Series(["a", "b", "c"])

    def test_valid_numeric_values(self, valid_numeric_values):
        result = significance_numeric_numeric(
            valid_numeric_values, valid_numeric_values
        )
        assert isinstance(result, SignificanceOutput)
        assert isinstance(result.significance, SignificanceEnum)
        assert isinstance(result.pvalue, float)
        assert len(result.influence) == 1
        assert result.influence.isin(["--", "-", " ", "", "+", "++"]).all()
        assert isinstance(result.statistic, float)

    def test_invalid_numeric_values(self, invalid_numeric_values):
        with pytest.raises(TypeError):
            significance_numeric_numeric(
                invalid_numeric_values, invalid_numeric_values
            )


def test__compute_group_info():
    # Generate test data
    numeric_values = pd.Series([1, 2, 3, 4, 4])
    categorical_values = pd.Series(["A", "A", "A", "B", "B"])

    # Call the function
    result = _compute_group_info(numeric_values, categorical_values)

    # Assert the output
    assert isinstance(result, dict)
    assert len(result) == 2

    # Assert group A information
    group_a_info = result["A"]
    assert isinstance(group_a_info, dict)
    pd.testing.assert_series_equal(
        group_a_info["values"], numeric_values.iloc[:3]
    )
    assert group_a_info["mean"] == 2
    assert "normalized_values" in group_a_info
    assert "std" in group_a_info
    assert "ks_test" in group_a_info

    # Assert group B information
    group_b_info = result["B"]
    assert isinstance(group_b_info, dict)
    pd.testing.assert_series_equal(
        group_b_info["values"], numeric_values.loc[3:]
    )
    assert group_b_info["mean"] == 4
    assert "normalized_values" in group_b_info
    assert "std" in group_b_info
    assert "ks_test" in group_b_info


class TestSignificanceNumericCategorical:
    def setup_method(self):
        np.random.seed(22)
        group_desc = {
            "group_0": {"size": 100, "mean": -5, "std": 2},
            "group_1": {"size": 80, "mean": 0, "std": 0.1},
            "group_2": {"size": 50, "mean": 5, "std": 10},
        }
        self.categorical_values = pd.concat(
            [
                pd.Series([group_name] * group["size"], name="category")
                for group_name, group in group_desc.items()
            ],
            ignore_index=True,
        )
        mean_array = np.concatenate(
            [
                [group["mean"]] * group["size"]
                for _, group in group_desc.items()
            ]
        )
        std_array = np.concatenate(
            [[group["std"]] * group["size"] for _, group in group_desc.items()]
        )
        # Numeric values for the case where the groups are normally distributed
        # And have the same variance
        self.numeric_values_normal = pd.Series(
            np.random.randn(len(self.categorical_values)) + mean_array,
            name="number",
        )
        # Numeric values for the case where the groups are normally distributed
        # but have different variances
        self.numeric_values_normal_diff_var = pd.Series(
            np.random.randn(len(self.categorical_values)) * std_array
            + mean_array,
            name="number",
        )

    def test_significance_numeric_categorical_anova(self):
        # Call the function
        result = significance_numeric_categorical(
            self.numeric_values_normal, self.categorical_values
        )

        # Assert the output
        assert isinstance(result, SignificanceOutput)
        assert isinstance(result.significance, SignificanceEnum)
        assert isinstance(result.pvalue, float)
        assert len(result.influence) == 3
        assert result.influence.isin(["--", "-", " ", "", "+", "++"]).all()
        assert isinstance(result.statistic, float)
        assert "ANOVA" in result.message

    def test_significance_numeric_categorical_kw(self):
        # Call the function
        result = significance_numeric_categorical(
            self.numeric_values_normal_diff_var, self.categorical_values
        )

        # Assert the output
        assert isinstance(result, SignificanceOutput)
        assert isinstance(result.significance, SignificanceEnum)
        assert isinstance(result.pvalue, float)
        assert len(result.influence) == 3
        assert result.influence.isin(["--", "-", " ", "", "+", "++"]).all()
        assert isinstance(result.statistic, float)
        assert "Kruskal-Wallis" in result.message


class TestSignificanceCategoricalCategorical:
    def setup_method(self):
        np.random.seed(22)
        self.categorical_values_2_group = pd.Series(
            np.random.randint(2, size=200), name="2 groups"
        ).astype(str)
        self.categorical_values_4_group = pd.Series(
            np.random.randint(4, size=200), name="4 groups"
        ).astype(str)

    def test_significance_categorical_categorical(self):
        # Call the function
        result = significance_categorical_categorical(
            self.categorical_values_4_group, self.categorical_values_2_group
        )

        # Assert the output
        assert isinstance(result, SignificanceOutput)
        assert isinstance(result.significance, SignificanceEnum)
        assert isinstance(result.pvalue, float)
        assert len(result.influence) == 8
        assert result.influence.isin(["--", "-", " ", "", "+", "++"]).all()
        assert isinstance(result.statistic, float)


class TestSignificanceOutput:
    def test_significance_property_weak(self):
        output = SignificanceOutput(
            pvalue=0.1, influence=pd.Series(), statistic=1.0
        )
        assert output.significance == SignificanceEnum.WEAK_VALUE

    def test_significance_property_medium(self):
        output = SignificanceOutput(
            pvalue=0.06, influence=pd.Series(), statistic=1.0
        )
        assert output.significance == SignificanceEnum.MEDIUM_VALUE

    def test_significance_property_strong(self):
        output = SignificanceOutput(
            pvalue=0.01, influence=pd.Series(), statistic=1.0
        )
        assert output.significance == SignificanceEnum.STRONG_VALUE

    def test_to_dataframe(self):
        output = SignificanceOutput(
            pvalue=0.05, influence=pd.Series(["+", ""]), statistic=1.0
        )
        df = output.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        pd.testing.assert_frame_equal(
            df,
            pd.DataFrame(
                {
                    "influence": ["+", ""],
                    "pvalue": 0.05,
                    "statistic": 1.0,
                    "message": "",
                    "significance": SignificanceEnum.MEDIUM_VALUE,
                }
            ),
        )
