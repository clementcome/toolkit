"""Defines the schema for the caracterisation module."""
import inspect
from enum import Enum, unique
from functools import wraps
from typing import Callable, Tuple

import numpy as np
import pandas as pd
from pandera import Check, DataFrameSchema
from pydantic import validate_call


def all_columns_numeric(df):
    return df.select_dtypes(include=[np.number]).shape[1] == df.shape[1]


def all_columns_categorical(df):
    return df.select_dtypes(exclude=[np.number]).shape[1] == df.shape[1]


OnlyNumericSchema = DataFrameSchema(checks=Check(all_columns_numeric))
OnlyCategoricalSchema = DataFrameSchema(checks=Check(all_columns_categorical))


@unique
class SeriesType(str, Enum):
    """Defines the type of a series."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


def check_series_in_signature(
    func: Callable, *arg_names: str
) -> inspect.Signature:
    """Checks that the specified arguments are pd.Series.

    Parameters
    ----------
    func : Callable
        The function to check.
    *arg_names : str
        The names of the arguments to check.

    Returns
    -------
    Signature
        The signature.

    Raises
    ------
    ValueError
        If an argument does not exist.
    TypeError
        If an argument is not a pd.Series.
    """
    signature = inspect.signature(func)
    for arg_name in arg_names:
        if arg_name not in signature.parameters:
            raise ValueError(f"Argument '{arg_name}' does not exist")
        elif signature.parameters[arg_name].annotation != pd.Series:
            raise TypeError(f"Argument '{arg_name}' must be a pd.Series")
    return signature


@validate_call
def check_input_types(*type_specs: Tuple[str, SeriesType]) -> Callable:
    """Checks the types of the arguments of the decorated function.

    Parameters
    ----------
    *type_specs : Tuple[str, SeriesType]
        A tuple of tuples, each tuple contains the name of the argument and the
        expected type of the argument.

    Returns
    -------
    Callable
        The decorator.
    """

    def decorator(func: Callable) -> Callable:
        signature = check_series_in_signature(
            func, *[arg_name for arg_name, _ in type_specs]
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_arguments = signature.bind(*args, **kwargs)
            bound_arguments.apply_defaults()

            for arg_name, expected_type in type_specs:
                series = bound_arguments.arguments[arg_name]

                if (
                    expected_type == SeriesType.NUMERIC
                    and not pd.api.types.is_numeric_dtype(series)
                ):
                    raise TypeError(f"Argument '{arg_name}' must be numeric")
                elif (
                    expected_type == SeriesType.CATEGORICAL
                    and pd.api.types.is_numeric_dtype(series)
                ):
                    raise TypeError(
                        f"Argument '{arg_name}' must be categorical"
                    )

            return func(*args, **kwargs)

        return wrapper

    return decorator


@validate_call
def check_input_index(*arg_names: str) -> Callable:
    """Checks that the specified arguments have the same index.

    Parameters
    ----------
    *arg_names : str
        The names of the arguments to check.

    Returns
    -------
    Callable
        The decorator.
    """

    def decorator(func):
        signature = check_series_in_signature(func, *arg_names)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_arguments = signature.bind(*args, **kwargs)
            bound_arguments.apply_defaults()

            series_list = [
                bound_arguments.arguments[arg_name] for arg_name in arg_names
            ]

            first_series_index = series_list[0].index
            for series in series_list[1:]:
                if not series.index.equals(first_series_index):
                    raise ValueError(
                        "All specified Series must have the same index."
                    )

            return func(*args, **kwargs)

        return wrapper

    return decorator
