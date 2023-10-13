# utils.py

# General utilities

from __future__ import annotations
import ast
import datetime
import sys
import itertools
from collections.abc import Callable, Collection, Hashable, Iterable
from typing import Any, Annotated, Union

from numpy import nan

__all__ = ["LoggingLevels",
           "Numeric",
           "calc_ranges",
           "calc_intervals",
           "default",
           "dict2list",
           "msg",
           "now",
           "parse_literal_eval",
           "pops",
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

Numeric = Union[int, float]


def calc_ranges(values: Iterable[int]) -> list[tuple[int, int]]:
    """calculate intervals without sorting input values

    From https://stackoverflow.com/a/72043030
    Returns a list of 2-tuples of ints
    """
    ranges = []
    for _, g in itertools.groupby(enumerate(values), lambda k: k[0] - k[1]):
        g_start = next(g)[1]
        g_end = list(v for _, v in g) or [g_start]
        ranges.append((g_start, g_end[-1] + 1))
    return ranges


def calc_intervals(values: Union[int, Iterable[int]], *other_values
                   ) -> list:
    """Calculate spanning interval ranges given list of values (with sorting)"""

    flattened = []
    # flatten list of values
    for val in [values, *other_values]:
        if isinstance(val, Iterable):
            flattened.extend(list(val))
        else:
            flattened.append(val)

    return calc_ranges(sorted(flattened))


# def merge_intervals(ranges: Iterable[Iterable[int]],
#                     start: int = None,
#                     end: int = None):
#     """ranges: Iterable of 2-length iterables"""
#     merged = []

#     sorted_ranges = sorted(ranges, key=lambda x: (x[0], x[1]))

#     curr_start = default(start, sorted_ranges[0][0])
#     prev_end = sorted_ranges[0][1]

#     for i, j in sorted_ranges:
#         if i + 1 < prev_end:

    # return


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


