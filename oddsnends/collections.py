"""Manipulating collections"""
from __future__ import annotations

import itertools
from collections.abc import Collection
from typing import Any, Annotated

from pandas import notnull

__all__ = [
    "agg",
    "dict2list",
    "drop_duplicates",
    "dropna",
    "pops",
    "simplify",
    "strictcollection",
]

def agg(*args):
    """Aggregate args by addition (a + b)"""
    res = ""
    for arg in itertools.chain.from_iterable(args):
        res = res + arg
    return res

def dict2list(dct: dict) -> list:
    """Flatten dictionary to list [key1, val1, key2, val2, ...]"""
    return list(itertools.chain.from_iterable(dct.items()))


def _drop_duplicates(values: Any) -> Any:
    try:
        dct = dict.fromkeys(values, True)
    except TypeError:
        return values
    return list(dct.keys())

drop_duplicates = _drop_duplicates  # for overloading in simplify()

def _dropna(values: Any) -> Any:
    try:
        assert values is not None
        filtered = filter(lambda x: notnull(x), values)
    except (AssertionError, TypeError):
        return values
    else:
        return type(values)(filtered)

dropna = _dropna  # for overloading in simplify()

def pops(dct: dict, *keys, **kws) -> list[Any]:
    """Pop multiple keys from a dict. if kw 'd' is given, uses d as default
    **kws takes: 
        d: Any. Returns a default value.
        errors: 'ignore' or 'raise'. Default "ignore
    """
    def _pop_(_dct, _k, _errors):
        try:
            return _dct.pop(_k)
        except KeyError as error:
            if _errors == "raise":
                raise error
        
    errors = kws.pop("errors", "ignore")
    try:
        d = kws.pop('d')
    except KeyError:
        return [_pop_(dct, k, errors) for k in keys]
    
    return [dct.pop(k, d) for k in keys]


def simplify(value: Any):

    try:
        assert len(value) != 1, 1
        assert len(value) != 0, 0
        assert not isinstance(value, (str, dict)), None
        return list(value)

    except TypeError:
        return value
    except AssertionError as error:
        match error.args[0]:
            case 0:
                return None
            case 1:
                return value[0]
            case _:
                return value

def strictcollection(value: Any) -> bool:
    """Check if value is a Collection and not a string-like"""
    return isinstance(value, Collection) and not isinstance(value, str)

