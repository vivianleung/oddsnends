"""math.py"""

__all__ = [
    "Numeric",
    "rounding",
    "ceiling",
    "floor",
    ]

# Math utilities
import math
from collections.abc import Callable
from typing import Union

Numeric = Union[int, float]

# Math
def rounding(x: Numeric, func: Callable = round, ndigits: int = 0,
             **kwargs) -> Numeric:
    """Returns value rounded by func
    func: takes x as first arg and then any *args, **kwargs
    """
    if ndigits == 0:
        return func(x, **kwargs)
    multiplier = 10 ** min(ndigits, len(str(func(x, **kwargs))))
    return func(x / multiplier, **kwargs) * multiplier

def ceiling(x: Numeric, ndigits: int = 0) -> Numeric:
    """Returns ceiling of x to ndigits decimal places"""
    return rounding(x, math.ceil, ndigits=ndigits)

def floor(x: Numeric, ndigits: int = 0) -> Numeric:
    """Returns floor of x to ndigits decimal places"""
    return rounding(x, math.floor, ndigits=ndigits)
