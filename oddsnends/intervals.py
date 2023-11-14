"""Manipulating intervals and ranges"""

from __future__ import annotations
import itertools
from collections.abc import Callable, Collection, Hashable, Sequence
from typing import Annotated

import pandas as pd

from oddsnends.main import OptionsMetaType, default

__all__ = [
    "IntervalType",
    "RangeType",
    "calc_intervals",
    "intervals2locs",
    "setops_ranges",
]


RangeType = tuple[int, int]


class IntervalType(metaclass=OptionsMetaType):
    """Type of interval (left/lower/up/right refers to closed X)"""

    __doc__ = """
    IntervalType describes the closed state of interval bounds.

    Options                    lower    upper   notation
    -----------------------  -------  -------  ---------
    'lower', 'left'          closed    open     [l, u)
    'upper', 'right'          open    closed    (l, u]
    'closed', 'both', True   closed   closed    [l, u]
    'open', None, False       open     open     (l, u)
    """

    bases = (str, bool)
    options = ("lower", "left", "upper", "right", "closed", "both", True, "open", False)

    @classmethod
    def isclosed(cls, option: str | bool, lower: bool = True):
        """Check if `option` indicates bound is closed.

        Parameters
        ----------
        option: IntervalType
            Value to check
        lower: bool
            Checks lower bound if True, and upper bound if False. Default True

        Returns
        -------
        bool of whether bound is closed
        """
        if not isinstance(option, cls):
            raise TypeError(option, f"`option` must be one of {repr(cls)}")

        if lower:
            return option in ["lower", "left", "closed", "both", True]
        else:
            return option in ["upper", "right", "closed", "both", True]


def calc_intervals(
    *locs: Collection[int],
    sort: bool = True,
    sort_key: Callable = None,
    sort_reverse: bool = False,
    interval_type: IntervalType = "lower",
) -> list[int]:
    """Calculate spanning interval ranges given list of values (with sorting)

    loc: list-like of ints
        Values to merge into intervals
    sort: bool
        Sort values before making intervals
    sort_key: callable
        Passed to sorted() function
    sort_reverse: bool,
        Passed to sorted() function
    interval_type: IntervalType, optional
        See help(IntervalType). Default "lower" (native python range system)
    Returns a list of 2-tuples of ints
    """
    # Based on https://stackoverflow.com/a/72043030

    # shift bounds by this much,
    shift_lower, shift_upper = shift_interval(True, interval_type)

    flattened = list(itertools.chain.from_iterable(locs))

    if sort:
        flattened = sorted(flattened, key=sort_key, reverse=sort_reverse)

    if len(flattened) == 0:
        return []

    if len(flattened) == 1:
        return [(flattened[0], flattened[0] + 1)]

    try:
        not_int = list(filter(lambda x: not isinstance(x, int), flattened))
        assert len(not_int) == 0, not_int
    except AssertionError as err:
        raise ValueError(f"Not int: {not_int}") from err

    intervals = []
    try:
        for _, g in itertools.groupby(enumerate(flattened), lambda k: k[0] - k[1]):
            g_start = next(g)[1]
            g_end = list(v for _, v in g) or [g_start]
            intervals.append((g_start + shift_lower, g_end[-1] + shift_upper))

    except TypeError as error:
        raise ValueError("Bad value in provided values.") from error
    return intervals


def intervals2locs(
    intervals: pd.DataFrame | Sequence[RangeType],
    col_from: str = "POS",
    col_to: str = "END_POS",
    name: str = "loc",
    interval_type: IntervalType = "lower",
    # indexing: Annotated[int, 0, 1] = 0,
    ignore_index: bool = True,
    drop_duplicates: bool = True,
    dropna: bool = True,
) -> pd.Series:
    """Breaks down list of ranges into individual positions

    Parameters
    ----------
    values: pd.DataFrame or list-like of 2-tuple ints
        Contains lower and upper bounds of the ranges
    col_from: str
        Name of column containing lower bounds (if values is pd.DataFrame).
        If None, the first column is used
    col_to: str
        Name of column containing lower bounds (if values is pd.DataFrame).
        If None, the second column is used.
    name: str
        Name of returned pd.Series. Default "loc"
    interval_type: IntervalType, optional
        See help(IntervalType). Default "lower" (native python range system)
    # indexing: 0 or 1
    #     Intervals are based on 0- or 1-indexing system. Default 1

    ignore_index: bool
        If True, returned pd.Series index will be [0, 1, ...]. Default True
    drop_duplicates: bool
        If True, drop duplicate loc values. Default True
    dropna: bool
        If True, drop na values. Default True

    Returns: pd.Series of locs covered by the ranges
    """
    colnames = [col_from, col_to]

    if isinstance(intervals, pd.DataFrame):
        # get left and right pos

        lower = default(col_from, intervals.iloc[:, 0], intervals[col_from])
        upper = default(col_to, intervals.iloc[:, 1], intervals[col_to])

        # concat as columns
        intervals = pd.concat([lower, upper], axis=1, keys=colnames)
    else:
        # convert to dataframe
        intervals = pd.DataFrame(intervals, columns=colnames)

    # adjust left and right bounds for calling range()
    shift_start, shift_end = shift_interval(interval_type, "left")

    res = (
        intervals.apply(
            lambda ser: range(ser[col_from] + shift_start, ser[col_to] + shift_end),
            axis=1,
        )
        .explode(ignore_index=ignore_index)
        .rename(name)
    )
    if drop_duplicates:
        res.drop_duplicates(inplace=True)
    if dropna:
        res.dropna(inplace=True)
    if ignore_index:
        res.reset_index(inplace=True, drop=True)
    return res


def setops_ranges(
    interval_1: pd.DataFrame | Sequence[RangeType],
    interval_2: pd.DataFrame | Sequence[RangeType],
    col_from: str = "start",
    col_to: str = "end",
    how: Annotated[str, "right_only", "left_only", "both"] = "both",
    interval_type: IntervalType = "left",
    indexing: Annotated[int, 0, 1] = 0,
    **intervals2locs_kws,
):
    """Compare two list of ranges and calculate different sets of intervals.

    Parameters
    ----------

    how: 'right_only', 'left_only', or 'both'
        Subset of locs to keep
    interval_type:  'left', 'lower', 'right', 'upper', 'both', True, False, or None
        Treat ranges as interval_type on the
        - 'left' or 'lower':  left/lower bound only, i.e. [from, to)
        - 'right' or 'upper': right/upper bound only, i.e. (from, to]
        - 'both' or True:     both bounds (interval is interval_type), i.e. [from, to]
        - None or False:      interval is open, i.e. (from, to)
        This is used by intervals2locs() and calc_intervals()
        Default 'lower' (which is the native python range interpretation)
    indexing: 0 or 1
        Intervals are based on 0- or 1-indexing system. Default 1

    **intervals2locs_kws are additional kws passed to intervals2locs(). ** takes:
        - name
        - drop_duplicates
        - dropna
    These are passed to intervals2locs(). See documentation.

    Returns: pd.DataFrame of col_start: int, col_end: int
    """

    if isinstance(interval_1, Hashable):
        interval_1 = [interval_1]
    if isinstance(interval_2, Hashable):
        interval_2 = [interval_2]

    # set defaults
    intervals2locs_kws = {
        "col_from": col_from,
        "col_to": col_to,
        "interval_type": interval_type,
        "indexing": indexing,
    } | intervals2locs_kws

    locs_1 = intervals2locs(interval_1, **intervals2locs_kws)
    locs_2 = intervals2locs(interval_2, **intervals2locs_kws)

    keep_locs = (
        pd.merge(locs_1, locs_2, how="outer", indicator=True)
        .pipe(lambda df: df.loc[df["_merge"] == how])
        .drop("_merge", axis=1)
        .squeeze(axis=1)
    )

    # provide calc_intervals with interval_type=interval_type so it outputs the same
    # interval format as our input
    intervals = pd.DataFrame(
        calc_intervals(keep_locs, sort=True, interval_type=interval_type),
        columns=[col_from, col_to],
    )

    return intervals


def shift_interval(
    interval_type_from: IntervalType,
    interval_type_to: IntervalType,
    index_from: Annotated[int, 0, 1] = None,
    index_to: Annotated[int, 0, 1] = None,
    left: int = None,
    right: int = None,
) -> RangeType:
    """Returns values to shift left/lower and right/upper bounds"""
    match interval_type_from:  # shift from this to left closed, right open
        case "right" | "upper":  # left
            shift_left = 1
            shift_right = 1

        case "both" | "closed" | True:  # POS, POS + len(REF) - 1
            shift_left = 0
            shift_right = 1

        case "open" | None | False:  # POS - 1, POS + len(REF)
            shift_left = 1
            shift_right = -1

        case "left" | "lower":  # POS, POS + len(REF)
            shift_left = 0
            shift_right = 0

        case _:
            raise ValueError("closed_from", interval_type_from)

    match interval_type_to:  # shift from left-closed to this
        case "right" | "upper":  # POS - 1, POS + len(REF) - 1
            shift_left += -1
            shift_right += -1

        case "both" | "closed" | True:  # POS, POS + len(REF) - 1
            shift_left += 0
            shift_right += -1

        case "open" | None | False:  # POS - 1, POS + len(REF)
            shift_left += -1
            shift_right += 0

        case "left" | "lower":  # POS, POS + len(REF)
            shift_left += 0
            shift_right += 0

        case _:
            raise ValueError("closed_to", interval_type_to)

    if (index_from is not None) and (index_to is not None):
        shift_indexing = index_to - index_from
        shift_left += shift_indexing
        shift_right += shift_indexing

    if (left is None) and (right is None):
        return shift_left, shift_right

    # new_left = default(left, has_value=lambda x: x + shift_left)
    # new_right = default(right, has_value=lambda x: x + shift_right)
