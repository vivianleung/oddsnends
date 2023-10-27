# pdtools.py

# pandas tools
from __future__ import annotations
import logging
from collections.abc import Hashable, MutableSequence, Sequence
from typing import Annotated, TypeVar, Union

import numpy as np
import pandas as pd
from pandas.core.generic import NDFrame
from oddsnends.main import default


__all__ = ["SeriesType",
           "assign",
           "check_if_exists",
           "dedup_alias",
           "pipe_concat",
           "get_level_uniques",
           "pivot_indexed_table",
           "reorder_cols",
           "sort_levels",
           "swap_index",
           ]


SeriesType = TypeVar("SeriesType")

def assign(ser: pd.Series, **kwargs) -> pd.DataFrame:
    """Wrapper for assigning values to series and converting to DataFrame"""
    return ser.to_frame().assign(**kwargs)
    

def check_if_exists(labels: list, index: pd.Index,
                    errors: Annotated[str, 'ignore', 'warn', 'raise'] = 'ignore'
                    ) -> list:
    """"check if label exists"""
    if min(len(labels), len(index)) == 0:
        return labels
    try:
        is_in_index = np.array([el in index for el in labels])
        assert all(is_in_index) or (errors == 'ignore'), \
            np.choose(np.where(~is_in_index), labels)[0]
                    
    except AssertionError as err:
        msg = f"Labels not in index: {list(err.args[0])}"
        if errors == 'warn':
            logging.warning(msg)
        elif errors == 'raise':
            raise ValueError(msg) from err
        else:
            raise ValueError(f"errors arg '{errors}' is invalid.") from err
    
    which_labels = np.where(is_in_index)
    
    return np.array(labels)[which_labels]


def dedup_alias(data: pd.DataFrame,
                value: Hashable,
                id_col: Hashable,
                columns: Hashable | Sequence[Hashable],
                alias_name: str = "ALIAS"
                # ) -> tuple[pd.DataFrame, pd.DataFrame]:
                ) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Deduplicates and aliases entries (by 'id_col') with identical values
    
    Returns 
    - pd.Series: id_xref with index ID_COL and values ALIAS
    - pd.DataFrame: alias_xref with index ALIAS, columns COLUMNS, values value
    - pd.DataFrame: aliased values with columns [*COLUMNS, ALIAS, value]
    """
    # - pd.DataFrame: alias xrefs with ["KEY", id_col, N_id_col]
    if len(data) == 0:
        id_xref = pd.Series(name=alias_name).rename_axis(id_col)
        alias_xref = pd.DataFrame(columns=columns).rename_axis(alias_name)
        
        xrefs = pd.DataFrame(columns=["KEY", id_col, f"N_{id_col}"])
        aliased = pd.DataFrame(columns=[*columns, alias_name, value])
    
    else:
        pivoted = data.reset_index().pivot_table(
            value, id_col, columns, aggfunc=lambda x: x)

        id_col_idx = pivoted.columns
        groups = pivoted.reset_index().groupby(id_col_idx.to_list())

        xrefs = (
            pd.DataFrame(list(groups[id_col]), columns=["KEY", id_col])
            .assign(**{f"N_{id_col}": lambda df: df[id_col].apply(len)})
            .sort_values([f"N_{id_col}", "KEY"], ascending=[False, True],
                         ignore_index=True)
            .rename_axis(alias_name)
        )
        
        id_xref = xrefs[id_col].explode().pipe(swap_index)
        
        alias_xref = pd.DataFrame.from_records(
            xrefs["KEY"], columns=id_col_idx).rename_axis(alias_name)
        
        aliased = (
            pd.DataFrame
            .from_records(xrefs["KEY"], columns=id_col_idx)
            .rename_axis(alias_name)
            .melt(value_name=value, ignore_index=False)
            .reset_index()
            .pipe(reorder_cols, last=alias_name)
        )
    return id_xref, alias_xref, xrefs, aliased
    # return xrefs, aliased



def get_level_uniques(df: pd.MultiIndex | NDFrame,
                      name: str, axis: int | str = 0) -> pd.Index:
    """Faster way of getting unique values from an index level"""

    if isinstance(df, pd.MultiIndex):
        index = df
    elif axis in [1, "columns"]:
        index = df.columns
    else:
        index = df.index

    level = np.where([s == name for s in index.names])[0]
    try:
        assert len(level) == 1, len(level)
    except AssertionError as err:
        if err.args[0] == 0:
            raise ValueError(f"Level {name} not found.") from err
        print("Non-unique level", name)

    return index.levels[level[0]].values


def pipe_concat(*objs: NDFrame, **kws):
    """A pipe-able way to concat"""
    return pd.concat(objs, **kws)


def pivot_indexed_table(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """Pivot indexed table, preserving index in original df

    Parameters
    ----------
    df:     pd.DataFrame
    *args, **kwargs
        Passed to df.pivot_table. Do not use 'index' as arg or kwarg
    """
    reset_df = df.reset_index()

    # col names corresponding to index levels in df
    index_cols = list(reset_df.columns[:df.index.nlevels])

    pivoted = (reset_df
               .pivot_table(*args, index=index_cols, **kwargs)
               .rename_axis(df.index.names, axis=0)
               )

    return pivoted


def reorder_cols(df: pd.DataFrame,
                 first: Hashable | Sequence[Hashable] | pd.Index | None = None,
                 last: Hashable | Sequence[Hashable] | pd.Index | None = None,
                 inplace: bool = False,
                 ascending: bool = None,
                 sort_kws: dict = None,
                 errors: Annotated[str, 'ignore', 'warn', 'raise'] = 'ignore',
                 ) -> pd.DataFrame | None:
    """Reorders columns of dataframe.

    Arguments:
        df: pd.DataFrame
        first: column label, list of labels or pd.Index to put first
        last: column label, list of labels or pd.Index to put last
        inplace: bool, default False
        ascending: bool, default None
            how to sort remaining columns, where True is ascending, False is descending, and None is not sorted
        sort_kws: dict, default None
            kwargs to pass to pd.DataFrame.sort_index() func

    Returns: pd.DataFrame if inplace is False, else None.
    """
    # check input df object is not empty
    if len(df.columns) == 0:
        if errors == 'raise':
            raise ValueError('Empty DataFrame')
        elif errors == 'warn':
            logging.warning('Warning: Empty DataFrame')
        return df
    
    # check if first and last cols are in df (filter out ones that aren't)
    if first is None:
        first = []
    else:
        if not isinstance(first, (pd.Index, MutableSequence)):
            first = default(first, [], has_value=[first])
        first = check_if_exists(first, df.columns, errors=errors)
    
    if last is None:
        last = []
    else:
        if not isinstance(last, (pd.Index, MutableSequence)):
            last = default(last, [], has_value=[last])
        last = check_if_exists(last, df.columns, errors=errors)
    
    # list of other (unspecified) columns
    mid = df.drop(first, axis=1).drop(last, axis=1)
        
    if isinstance(ascending, bool):
        mid = mid.sort_index(axis=1, ascending=ascending,
                             **default(sort_kws, {}))

    if inplace:
        # arrange and store a copy of left side before dropping below
        left = pd.concat([df.loc[:, first], mid], axis=1)

        # reverse to insert at 0 during .apply
        left = left[list(reversed(left.columns))]

        # drop left columns inplace
        df.drop([*first, *mid.columns], axis=1, inplace=True)
        
        # put left columns back into the df in the new order
        left.apply(lambda col: df.insert(0, col.name, col))

    else:
        # simply concat and return a new copy
        return pd.concat([df.loc[:, first], mid, df.loc[:, last]], axis=1)
    
    
    
def sort_levels(df: NDFrame | pd.MultiIndex, axis: int = 0, **sort_kws
                ) -> NDFrame | pd.MultiIndex:
    """Sort index or column levels of a MultiIndex (in a Series or DataFrame)

    axis: sort index (0) or column(1) levels (if DataFrame is passed)
    **sort_kws passed to sort_values function, excluding 'by' and 'inplace'
    """
    def _order_levels(_multiidx: pd.MultiIndex, **_sort_kws) -> pd.Series:
        """Core function for sorting levels"""
        names = pd.DataFrame(_multiidx.names)
        return names.sort_values(by=names.columns.to_list(), **_sort_kws
                                 ).apply(tuple, axis=1)

    # get the appropriate index object
    if isinstance(df, pd.Series):
        index = df.index
    elif isinstance(df, pd.DataFrame):
        index = df.columns if axis == 1 else df.index
    else:
        index = df
    
    if not isinstance(index, pd.MultiIndex):  # non-hierarchical index
        return df
    
    # sort levels
    sorted_index = _order_levels(index, **sort_kws)
    
    if isinstance(index, pd.DataFrame):
        return df.reorder_levels(sorted_index, axis=axis)
    else:
        return df.reorder_levels(sorted_index)
    


def swap_index(frame: NDFrame, keys: Hashable | Sequence[Hashable] = None,
               inplace: bool = False, **kws) -> NDFrame | None:
    """Swaps a column in for an index without dropping the reset index
    
    frame: pd.Series or pd.DataFrame
    keys:  keys or list of keys
        Column name(s) to set as new index
    
    """
    def _swap_index(_frame, _keys, **_kws):
        return _frame.reset_index(**_kws).set_index(_keys)

    def _get_set_names(_frame_names, _fmt: str = "index_{}"):
        return [default(
            name, _fmt.format(i)) for i, name in enumerate(_frame_names)]
    
    # set reset index names
    names = kws.pop("names", _get_set_names(frame.index.names))
    
    if keys is not None:
        pass
    elif isinstance(frame, pd.DataFrame):
        keys = frame.columns.to_list()
    else:
        keys = frame.name
        
    match frame:
        case pd.DataFrame() if inplace:
            frame.reset_index(inplace=inplace, names=names, **kws)
            frame.set_index(keys, inplace=inplace)
            
        case pd.DataFrame():
            return _swap_index(frame, keys, names=names, **kws)
        
        case pd.Series() if inplace:
            frame.rename_axis(names, inplace=True)
            swapped = _swap_index(frame, keys, *kws).squeeze(axis=1)
            frame.rename(frame, inplace=True)
            frame.replace(swapped, inplace=True)
            # swap names
            frame.rename_axis(swapped.index.names, inplace=True)
            frame.rename(swapped.name, inplace=True)
        
        case pd.Series():
            return frame.rename_axis(names).pipe(
                _swap_index, keys, *kws).squeeze(axis=1)
            
        case _:
            raise TypeError("Bad 'frame' type", type(frame))

