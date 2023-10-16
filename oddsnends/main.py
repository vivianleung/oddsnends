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
    "agg",
    "calc_intervals",
    "default",
    "defaults",
    "dict2list",
    "msg",
    "now",
    "parse_literal_eval",
    "pops",
    "ranges2locs",
    "setops_ranges",
    "strictcollection",
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


def agg(*args):
    """Aggregate args by addition (a + b)"""
    res = ""
    for arg in itertools.chain.from_iterable(args):
        res = res + arg
    return res


def calc_intervals(values: Collection[int], *other_values,
                   sort: bool = True,
                   sort_key: Callable = None, 
                   sort_reverse: bool = False,
                   ) -> list[int]:
    """Calculate spanning interval ranges given list of values (with sorting)
    
    Returns a list of 2-tuples of ints
    """
    # Based on https://stackoverflow.com/a/72043030

    
    if not (isinstance(values, Collection) or isinstance(values, int)):
        raise ValueError(values)
    
    flattened = list(itertools.chain.from_iterable([values]))
    
    if sort:
        values = sorted(values, key=sort_key, reverse=sort_reverse)


    if isinstance(values, int):
        return [(values, values + 1)]

    if len(values) == 0:
        return []
    
    if len(values) == 1:
        return [(values[0], values[0] + 1)]
            
    try:
        not_int = list(filter(lambda x: not isinstance(x, int), flattened))
        assert len(not_int) == 0, not_int
    except AssertionError as err:
        raise ValueError(f"Not int: {not_int}") from err 
        
    intervals = []
    try:
        for _, g in itertools.groupby(
                enumerate(flattened), lambda k: k[0] - k[1]):
            g_start = next(g)[1]
            g_end = list(v for _, v in g) or [g_start]
            intervals.append((g_start, g_end[-1] + 1))
    except TypeError as error:
        raise ValueError(f"Bad value in provided values.") from error
    return intervals


    # return
def ranges2locs(values: Collection[TwoTupleInts],
                col_start: str = None, 
                col_end: str = None,
                name: str = "loc",
                ignore_index: bool = True,
                drop_duplicates: bool = True,
                dropna: bool = True) -> pd.Series:
    """Breaks down list of ranges into individual positions"""
    
    colnames = [default(col_start, "left"), default(col_end, "right")]
    if isinstance(values, pd.DataFrame):
        # get left and right pos
        left = default(col_start, values.iloc[:, 0], values[col_start])
        right = default(col_end, values.iloc[:, 1], values[col_end])
        
        # concat as columns
        values = pd.concat([left, right], axis=1, keys=colnames)
    else:
        # convert to dataframe
        values = pd.DataFrame(values, columns=colnames)
     

    res = (
        values
        .apply(lambda ser: range(ser[colnames[0]], ser[colnames[1]]), axis=1)
        .explode(ignore_index=ignore_index)
        .rename(name)
    )
    if drop_duplicates:
        res.drop_duplicates(inplace=True)
    if dropna:
        res.dropna(inplace=True)
    
    return res

def setops_ranges(interval_1: Union[pd.DataFrame, Collection[TwoTupleInts]],
                  interval_2: Union[pd.DataFrame, Collection[TwoTupleInts]],
                  col_start: str = "start",
                  col_end: str = "end",
                  how=Annotated[str, "both", "right_only", "left_only"]):
    """Compare two list of ranges and calculate different sets of intervals.
    
    Returns: pd.DataFrame of col_start: int, col_end: int
    """
    
    if isinstance(interval_1, Hashable):
        interval_1 = [interval_1]
    if isinstance(interval_2, Hashable):
        interval_2 = [interval_2]
    
    ranges2locs_kws = {
        "col_start": col_start,
        "col_end": col_end,
        "name": "loc",
    }
    locs_1 = ranges2locs(interval_1, **ranges2locs_kws)
    locs_2 = ranges2locs(interval_2, **ranges2locs_kws)

    keep_locs = (
        pd.merge(locs_1, locs_2, how="outer", indicator=True)
        .pipe(lambda df: df.loc[df["_merge"] == how])
        .drop("_merge", axis=1)
        .squeeze(axis=1)
        )

    intervals = pd.DataFrame(
        calc_intervals(keep_locs), columns=[col_start, col_end])
    
    return intervals




def default(x: Any, default_value: Any = None,
            has_value: Any = lambda x: x,
            null_values: Annotated[Union[Any, Collection[Any], str], 'null'] = None,
            func_args: Collection = None,
            **kwargs):
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
        return has_value(x, *func_args, **kwargs)
    
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
            



def dict2list(dct: dict) -> list:
    """Flatten dictionary to list [key1, val1, key2, val2, ...]"""
    return list(itertools.chain.from_iterable(dct.items()))

def pops(dct: dict, *keys, **kws) -> list[Any]:
    """Pop multiple keys from a dict. if kw 'd' is given, uses d as default"""
    try:
        d = kws.pop('d')
    except KeyError:
        return [dct.pop(k) for k in keys]
    return [dct.pop(k, d) for k in keys]


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
    

def strictcollection(value: Any) -> bool:
    """Check if value is a Collection and not a string-like"""
    return isinstance(value, Collection) and not isinstance(value, str)


