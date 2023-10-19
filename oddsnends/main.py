# utils.py

# General utilities

from __future__ import annotations
import ast
import datetime
import sys
import itertools
from collections.abc import Callable, Collection, Hashable, Iterable
from typing import Any, Annotated, Union

import pandas as pd
from numpy import nan
from pandas.core.generic import NDFrame

__all__ = [
    "LoggingLevels",
    "NoneType",
    "Numeric",
    "TwoTupleInts",
    "default",
    "defaults",
    "msg",
    "now",
    "parse_literal_eval",
]


LoggingLevels = {
    "NOTSET": 0,
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}
NoneType = type(None)
Numeric = Union[int, float]
TwoTupleInts = tuple[int, int]



def default(x: Any, default_value: Any = None,
            has_value: Any = lambda x: x,
            null_values: Annotated[Union[Any, Collection[Any], str], 'null'] = None,
            func_args: Collection = None,
            func_kws: dict = None):
    """General function for checking/returning null/non-null objects.

    Arguments:
        x:  Any. Object to test
    Optional:
        default_value: Any, default None. Value to return if x is null
        has_value: Any, default lambda x: x
            Return this if x is not null. if has_value is Callable, then
            it should take x as the first arg and any *args, **kwargs.
        null_values: value, list of values considered as null, or 'null'. default None.
            If null='null', then any null-like value is considered, i.e. None,
            nan, empty strings or 0-length collections.
            Note: can't handle generators.
            If the literal value 'null' or non-str collections (e.g. tuple())
            are to considered as null values, they should be wrapped in a
            list-like (e.g. [tuple()] or ['null', tuple()]).
        *args, **kwargs passed to has_value (if it is a function)
    """

    none_objs = [None, nan]
    try:
        if null_values == 'null':
            if hasattr(x, '__len__'):
                assert len(x) > 0
            assert x not in none_objs
        else:
            if isinstance(null_values, Hashable):
                null_values = [null_values]

            for n in null_values:
                # 'nan == nan' is False, but 'nan is nan' is True
                assert (x != n) and not ((x in none_objs) and (n in none_objs))

    except AssertionError:  # is null value
        return default_value

    if isinstance(has_value, Callable):
        if func_args is None:
            func_args = []
        if func_kws is None:
            func_kws = {}
        return has_value(x, *func_args, **func_kws)

    else:
        return has_value



def defaults(x: Any, *default_values, **kwargs) -> Any:
    """Returns value (via. has value) or first non-null default value

    Note: can only check if a pd.NDFrame is empty. Cannot check values

    Parameters
    ----------
    x: Any
        Value to check
    *default_values: Any
        Default values to check (in order) and return the first non-null
        (or the last value all null). Default returns None.

    **kwargs takes:
    has_value: Any, Callable, default x
        What to return if value is non-null. If Callable, it should take x as
        the first (positional) arg, then any *func_args, **func_kwargs.
        Default returns x.
    empty_as_null: bool, default True
        Consider empty collections (lists, strs, etc.) as null
    null_values: Collection[Any]
        Values to consider null. Default includes None, np.nan, and empty
        Collections and strings
    func_args:  list-like
        Args to pass to has_value (if callable)
    func_kws:   dict
        Kwargs to pass to has_value (if callable)

    """
    def _check_if_null(_x: Any, _dvs: Collection[Any], _null_values: Collection[Any], do_raise: bool = False):
        try:
            if len(_dvs) == 0:
                return _x

            if hasattr(_x, "__len__") and empty_as_null:
                assert len(_x) > 0
            else:
                assert _x not in null_values

        except AssertionError:
            if do_raise:
                raise AssertionError(_dvs[0])
            return _check_if_null(_dvs[1:], _dvs, _null_values)

        return _x


    null_values = kwargs.get("null_values", [None, nan])
    if isinstance(null_values, Hashable):
        null_values = [null_values]
    empty_as_null = kwargs.get("empty_as_null", True)

    # check if x is null
    try:
        _ = _check_if_null(x, [default_values[0]], null_values, do_raise=True)

    except AssertionError as error:  # null value
        # get first non-null default value
        return _check_if_null(error.args[0], default_values[1:], null_values)

    # x is not null, so return has_value
    try:
        has_value = kwargs["has_value"]
        assert callable(kwargs["has_value"])
    except KeyError:
        return x
    except AssertionError:
        return has_value

    return has_value(
        x, *kwargs.get("func_args", []), **kwargs.get("func_kws", {}))




def msg(*args, stream=sys.stdout, sep=" ", end="\n", flush=True) -> None:
    """Writes message to stream"""
    stream.write(str(sep).join(str(x) for x in args) + end)
    if flush:
        stream.flush()

def now(fmt: str = '%c') -> str:
    """Get and format current datetime"""
    return datetime.datetime.now().strftime(fmt)


def parse_literal_eval(val: str) -> Any:
    """Wrapper for ast.literal_eval, returning val if malformed input"""
    try:
        return ast.literal_eval(val)
    except (ValueError, TypeError, SyntaxError, MemoryError):
        return val




