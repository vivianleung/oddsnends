"""main.py"""

# General utilities

from __future__ import annotations
import ast
import datetime
import sys
from collections.abc import Callable, Collection, Hashable, Sequence
from typing import Annotated, Any

from numpy import nan
from pandas.core.generic import NDFrame

__all__ = [
    "OptionsMetaType",
    "default",
    "defaults",
    "isnull",
    "notnull",
    "msg",
    "now",
    "parse_literal_eval",
    "xor",
]


class OptionsMetaType(type):
    """Metaclass"""

    options: list

    def __repr__(cls):
        bases = " | ".join((getattr(x, "__name__", str(x)) for x in cls.bases))
        options = ", ".join(
            (f"'{x}'" if isinstance(x, str) else str(x) for x in cls.options)
        )
        return f"{cls.__name__}[{bases}, {options}]"

    def __instancecheck__(cls, instance):
        return instance in cls.options

    def __new__(mcs, name: str, bases: tuple, attrs: dict):
        return super().__new__(mcs, name, bases, attrs)


def default(
    x: Any,
    default_value: Any = None,
    has_value: Any = lambda x: x,
    null_values: Annotated[Any | Collection[Any] | str, "null"] = None,
    func_args: Collection = None,
    func_kws: dict = None,
):
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
        if null_values == "null":
            if hasattr(x, "__len__"):
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


def _isnull(
    x: Any,
    null_values: Collection[Any] = None,
    empty: bool = True,
    do_raise: bool = False,
) -> bool:
    """Default null_values is [None, nan]"""

    if null_values is None:
        null_values = [None, nan]

    try:
        if hasattr(x, "__len__") and empty:
            assert len(x) > 0

        if isinstance(x, NDFrame):
            raise ValueError("Cannot check values in NDFrames", x)

        for n in null_values:  # "nan is nan" returns True but nan != nan
            assert (x != n) and (x is not n)

    except AssertionError as error:
        if do_raise:
            raise error
        return True

    return False


def _notnull(
    x: Any,
    null_values: Collection[Any] = None,
    empty: bool = True,
    do_raise: bool = False,
) -> bool:
    """Wrapper for _isnull"""
    return not _isnull(x, null_values=null_values, empty=empty, do_raise=do_raise)


isnull = _isnull
notnull = _notnull


def defaults(*values: Any, **kwargs) -> Any:
    """Returns value (via. has value) or first non-null default value

    Note: can only check if a pd.NDFrame is empty. Cannot check values

    Parameters
    ----------
    *values: Any
        Values to check (in order) and return the first non-null
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

    Usage:
    >>> defaults(None, None, 1)
    1
    >>> defaults(1, None, nan)
    1
    >>> defaults(None, 1, None)
    1
    >>> defaults("", 1, empty=True)
    1
    >>> defaults("", 1, empty=False)
    ''
    >>> defaults("", "foo", "bar", "baz", null_values=[None, nan, "foo"])
    'bar'
    >>> defaults("hello", "world")
    'hello'
    >>> defaults("hello", "world", has_value=lambda s: f"{s} and good night")
    'hello and good night'
    """
        
    # figure out what counts as a null value
    null_values = kwargs.get("null_values", [None, nan])
    if isinstance(null_values, Hashable):
        null_values = [null_values]

    empty = kwargs.get("empty", True)
    
    # initialize vars
    isnull = True  
    
    try:
        val = values[0]
        
    except IndexError as error:
        raise TypeError("Must provide at least one value") from error
    
    for val in values:
        isnull = _isnull(val, null_values, empty=empty, do_raise=False)
        if not isnull:
            break

    if isnull:  # reached the end
        return val
    else:
        # value is not null, so return has_value
        try:
            has_value = kwargs["has_value"]
            assert callable(has_value), has_value

        except KeyError:  # just return x
            return val

        except AssertionError as error:  # return the fixed value
            return error.args[0]

        return has_value(
            val, *kwargs.get("func_args", []), **kwargs.get("func_kws", {})
        )


#%%

def msg(*args, stream=sys.stdout, sep=" ", end="\n", flush=True) -> None:
    """Writes message to stream"""
    stream.write(str(sep).join(str(x) for x in args) + end)
    if flush:
        stream.flush()


def now(fmt: str = "%c") -> str:
    """Get and format current datetime"""
    return datetime.datetime.now().strftime(fmt)


def parse_literal_eval(val: str) -> Any:
    """Wrapper for ast.literal_eval, returning val if malformed input"""
    try:
        return ast.literal_eval(val)
    except (ValueError, TypeError, SyntaxError, MemoryError):
        return val


def xor(
    expr1: bool | Sequence[bool], expr2: bool | Sequence[bool]
) -> bool | Sequence[bool]:
    """Exclusive or operatory"""
    if isinstance(expr1, NDFrame) or isinstance(expr2, NDFrame):
        return (expr1 | expr2) & ~(expr1 & expr2)
    return (expr1 or expr2) and not (expr1 and expr2)
