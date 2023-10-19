"""Manipulating intervals and ranges"""

from __future__ import annotations
import itertools
import sys
from collections.abc import Callable, Collection, Hashable
from typing import Annotated, Union

import pandas as pd

from oddsnends.main import NoneType, TwoTupleInts, default

all = [
    "ClosedIntervalType",
    "calc_intervals",
    "intervals2locs",
    "setops_ranges",

]

ClosedIntervalType = Annotated[Union[str, bool, NoneType],
                               "lower", "left",
                               "upper", "right",
                               "both", True,
                               None, False]


def calc_intervals(*locs: Collection[int],
                   sort: bool = True,
                   sort_key: Callable = None,
                   sort_reverse: bool = False,
                   closed: ClosedIntervalType = "lower",
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
    closed:  str or bool or None
        Output intervals. Options:
        - 'left' or 'lower':  left/lower bound only, i.e. [from, to)
        - 'right' or 'upper': right/upper bound only, i.e. (from, to]
        - 'both' or True:     both bounds (interval is closed), i.e. [from, to]
        - None or False:      interval is open, i.e. (from, to)
        Default 'lower' (which is the native python range interpretation)
    Returns a list of 2-tuples of ints
    """
    # Based on https://stackoverflow.com/a/72043030
    
    # shift bounds by this much,     
    shift_lower, shift_upper = shift_interval(True, closed)

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
        for _, g in itertools.groupby(
                enumerate(flattened), lambda k: k[0] - k[1]):
            g_start = next(g)[1]
            g_end = list(v for _, v in g) or [g_start]
            intervals.append((g_start + shift_lower, g_end[-1] + shift_upper))
            
    except TypeError as error:
        raise ValueError(f"Bad value in provided values.") from error
    return intervals


def intervals2locs(intervals: Union[pd.DataFrame, Collection[TwoTupleInts]],
                col_from: str = None,
                col_to: str = None,
                name: str = "loc",
                closed: ClosedIntervalType = 'lower',
                indexing: Annotated[int, 0, 1] = 0,
                ignore_index: bool = True,
                drop_duplicates: bool = True,
                dropna: bool = True) -> pd.Series:
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
    closed:  'left', 'lower', 'right', 'upper', 'both', True, False, or None
        Treat ranges as closed on the
        - 'left' or 'lower':  left/lower bound only, i.e. [from, to)
        - 'right' or 'upper': right/upper bound only, i.e. (from, to]
        - 'both' or True:     both bounds (interval is closed), i.e. [from, to]
        - None or False:      interval is open, i.e. (from, to)
        Default 'lower' (which is the native python range interpretation)
    indexing: 0 or 1
        Intervals are based on 0- or 1-indexing system. Default 1

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
    shift_start, shift_end = shift_interval(closed, "left")

    res = (
        intervals
        .apply(lambda ser: range(ser[col_from] + shift_start,
                                 ser[col_to] + shift_end), axis=1)
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



def setops_ranges(interval_1: Union[pd.DataFrame, Collection[TwoTupleInts]],
                  interval_2: Union[pd.DataFrame, Collection[TwoTupleInts]],
                  col_from: str = "start",
                  col_to: str = "end",
                  how: Annotated[str, "right_only", "left_only", "both"] = "both",
                  closed: ClosedIntervalType = "left",
                  indexing: Annotated[int, 0, 1] = 0,
                  **intervals2locs_kws):
    """Compare two list of ranges and calculate different sets of intervals.

    Parameters
    ----------
    
    how: 'right_only', 'left_only', or 'both'
        Subset of locs to keep
    closed:  'left', 'lower', 'right', 'upper', 'both', True, False, or None
        Treat ranges as closed on the
        - 'left' or 'lower':  left/lower bound only, i.e. [from, to)
        - 'right' or 'upper': right/upper bound only, i.e. (from, to]
        - 'both' or True:     both bounds (interval is closed), i.e. [from, to]
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
    intervals2locs_kws = {"col_from": col_from,
                       "col_to": col_to,
                       "closed": closed,
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

    # provide calc_intervals with closed=closed so it outputs the same
    # interval format as our input
    intervals = pd.DataFrame(
        calc_intervals(keep_locs,
                       sort=True,
                       closed=closed),
        columns=[col_from, col_to])

    return intervals



def shift_interval(closed_from: ClosedIntervalType,
                   closed_to: ClosedIntervalType,
                   index_from: Annotated[int, 0, 1] = None,
                   index_to: Annotated[int, 0, 1] = None,
                   left: int = None,
                   right: int = None,
                   ) -> Union[TwoTupleInts, tuple[TwoTupleInts, TwoTupleInts]]:
    
    match closed_from:  # shift from this to left closed, right open
        case "right" | "upper":     # left 
            shift_left = 1
            shift_right = 1

        case "both" | True:         # POS, POS + len(REF) - 1
            shift_left = 0
            shift_right = 1
            
        case None | False:          # POS - 1, POS + len(REF)
            shift_left = 1
            shift_right = -1

        case "left" | "lower":      # POS, POS + len(REF)
            shift_left = 0
            shift_right = 0

        case _:
            raise ValueError("closed_from", closed_from)


    match closed_to:  # shift from left-closed to this
        case "right" | "upper":     # POS - 1, POS + len(REF) - 1
            shift_left += -1
            shift_right += -1

        case "both" | True:         # POS, POS + len(REF) - 1
            shift_left += 0
            shift_right += -1
            
        case None | False:          # POS - 1, POS + len(REF)
            shift_left += -1
            shift_right += 0

        case "left" | "lower":      # POS, POS + len(REF)
            shift_left += 0
            shift_right += 0

        case _:
            raise ValueError("closed_to", closed_to)

    if (index_from is not None) and (index_to is not None):
        shift_indexing = index_to - index_from
        shift_left += shift_indexing
        shift_right += shift_indexing
    
    if (left is None) and (right is None):
        return shift_left, shift_right
    
    new_left = default(left, has_value=lambda x: x + shift_left)
    new_right = default(right, has_value=lambda x: x + shift_right)
        
        