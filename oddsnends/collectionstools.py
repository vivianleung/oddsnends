"""Manipulating collections"""
from __future__ import annotations

import itertools
from collections.abc import Collection, Hashable
from typing import Any

from pandas import notnull

__all__ = [
    "AttrDict",
    "agg",
    "dict2list",
    "drop_duplicates",
    "dropna",
    "pops",
    "simplify",
    "strictcollection",
]

class AttrDict(dict):
    """Convenient way to store and access dict as attributes"""
    def __init__(self, **kws):
        for k, v in kws.items():
            setattr(self, k, v)

    def __getattr__(self, attr: Hashable, **d):
        return self.__getitem__(attr, **d)

    def __setattr__(self, attr: Hashable, value: Any):
        return self.__setitem__(attr, value)

    def __delattr__(self, attr: Hashable) -> None:
        return self.__delitem__(attr)


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
        filtered = filter(notnull, values)
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


def simplify(value: Any) -> Any:
    """Simplify value to a single value, if collection has only one value
    
    
    """
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
