# utils.py

# General utilities

from __future__ import annotations
import ast
import datetime
import os
import sys
import itertools
from collections.abc import Callable, Collection, Hashable, Iterable
from typing import Any, Annotated, Union

import pandas as pd
from numpy import nan
from pandas.core.generic import NDFrame

__all__ = [
    "NoneType",
    "TwoTupleInts",
    "default",
    "defaults",
    "isnull",
    "notnull",
    "msg",
    "now",
    "parse_literal_eval",
    "xor",
]


NoneType = type(None)
TwoTupleInts = tuple[int, int]



def default(x: Any, default_value: Any = None,
            has_value: Any = lambda x: x,
            null_values: Annotated[Any | Collection[Any] | str, 'null'] = None,
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


def _isnull(x: Any,
            null_values: Collection[Any] = None,
            empty: bool = True,
            do_raise: bool = False) -> bool:
    """Default null_values is [None, nan]"""
    
    if null_values is None:
        null_values = [None, nan]
    
    try:
        if hasattr(x, "__len__") and empty:
            assert len(x) > 0
        
        if isinstance(x, NDFrame):
            raise ValueError("Cannot check values in NDFrames", x)
        
        for n in null_values:   # "nan is nan" returns True but nan != nan
            assert (x != n) and (x is not n)
    
    except AssertionError as error:
        if do_raise:
            raise error
        return True

    return False

def _notnull(x: Any,
             null_values: Collection[Any] = None,
             empty: bool = True,
             do_raise: bool = False) -> bool:
    """Wrapper for _isnull"""
    return not _isnull(
        x, null_values=null_values, empty=empty, do_raise=do_raise)

isnull = _isnull
notnull = _notnull

def defaults(value: Any, *default_values, **kwargs) -> Any:
    """Returns value (via. has value) or first non-null default value

    Note: can only check if a pd.NDFrame is empty. Cannot check values

    Parameters
    ----------
    value: Any
        Value to check
    *default_values: Any
        Default values to check (in order) and return the first non-null
        (or the last value all null). Default returns None.

    **kwargs takes:
    has_value: Any, Callable, default x
        What to return if value is non-null. If Callable, it should take x as
        the first (positional) arg, then any *func_args, **func_kwargs.
        Default returns x.
    empty: bool, default True
        Consider empty collections (lists, strs, etc.) as null
    null_values: Collection[Any]
        Values to consider null. Default includes None, np.nan, and empty
        Collections and strings
    func_args:  list-like
        Args to pass to has_value (if callable)
    func_kws:   dict
        Kwargs to pass to has_value (if callable)

    """

    # figure out what counts as a null value
    null_values = kwargs.get("null_values", [None, nan])
    if isinstance(null_values, Hashable):
        null_values = [null_values]
        
    empty = kwargs.get("empty", True)

    # check if x is null
    try:
        isnull = _isnull(value, null_values, empty=empty, do_raise=True)
        
        i = 0
        while isnull:
            isnull = _isnull(default_values[i], null_values, empty=empty)
            i += 1
        return default_values[i]
    
    except AssertionError:
        # x is not null, so return has_value
        try:
            has_value = kwargs["has_value"]
            assert callable(has_value), has_value
            
        except KeyError:  # just return x
            return value                 
        
        except AssertionError as error:  # return the fixed value
            return error.args[0]

        return has_value(
            value, *kwargs.get("func_args", []), **kwargs.get("func_kws", {}))
            
    except IndexError:  # x and all default values are null
        return default_values[-1]
    


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



def xor(expr1, expr2):
    if isinstance(expr1, NDFrame) or isinstance(expr2, NDFrame):
        return (expr1 | expr2) & ~(expr1 & expr2)
    return (expr1 or expr2) and not (expr1 and expr2)
