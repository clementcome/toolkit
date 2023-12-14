"""Defines the schema for the caracterisation module."""
import numpy as np
from pandera import Check, DataFrameSchema


def all_columns_numeric(df):
    return df.select_dtypes(include=[np.number]).shape[1] == df.shape[1]


def all_columns_categorical(df):
    return df.select_dtypes(exclude=[np.number]).shape[1] == df.shape[1]


OnlyNumericSchema = DataFrameSchema(checks=Check(all_columns_numeric))

OnlyCategoricalSchema = DataFrameSchema(checks=Check(all_columns_categorical))
