"""pdtools.py"""

# pandas tools
from __future__ import annotations
import logging
from collections.abc import Callable, Hashable, MutableSequence, Sequence
from typing import Annotated, TypeVar

import numpy as np
import pandas as pd
from pandas.core.generic import NDFrame
from oddsnends.main import default


__all__ = [
    "SeriesType",
    "alias_crossref",
    "assign",
    "check_if_exists",
    "drop_labels",
    "group_identical_rows",
    "pipe_concat",
    "get_level_uniques",
    "ordered_fillna",
    "pivot_indexed_table",
    "rank_sort",
    "reorder_cols",
    "sort_levels",
    "swap_index",
]


SeriesType = TypeVar("SeriesType")


def alias_crossref(data: pd.DataFrame,
                   alias_name: Hashable = "ALIAS",
                   alias_prefix: str = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign and reindex `data` with alias (with prefix) to rows

    Please sort your data beforehand.

    Parameters
    ----------
    data: pd.Series or pd.DataFrame
        The data to alias
    alias_name : Hashable, optional
        Name of the output alias column
    alias_prefix : formatter string
        Prefix for alias. Default None.

    Returns
    -------
    pd.DataFrame
        `data` indexed with new aliases
    pd.DataFrame
        Cross-ref with index as aliases, values as (lists of) `data` indices
    """

    # make aliases
    if alias_prefix is None:
        aliases = range(len(data))

    else:
        pad = len(str(len(data)))
        formatter = f"{alias_prefix}{{0:0{pad}d}}".format
        aliases = [formatter(x) for x in range(len(data))]

    # make new dataframe (index.to_frame is faster than reset_index)
    aliased = data.assign(
        **{alias_name: aliases}).set_index(alias_name, append=True)

    aliased_genotypes = (
        aliased.index.to_frame()
        .droplevel(data.index.names, axis=0)
        .drop(alias_name, axis=1)
    )

    # re-format columns index
    if isinstance(aliased_genotypes.columns, pd.MultiIndex):
        aliased_genotypes.columns = pd.MultiIndex.from_tuples(
            aliased_genotypes.columns, names=aliased_genotypes.columns.names
        )
    
    # also drops the 'rows' index levels
    xrefs = aliased.reset_index(alias_name).set_index(alias_name)

    return aliased_genotypes, xrefs




def assign(ser: pd.Series, **kwargs) -> pd.DataFrame:
    """Wrapper for assigning values to series and converting to DataFrame"""
    return ser.to_frame().assign(**kwargs)


def check_if_exists(
    labels: list,
    index: pd.Index,
    errors: Annotated[str, "ignore", "warn", "raise"] = "ignore",
) -> list:
    """Checks if a given list of labels exists in a given index and returns
    the labels that do exist.

    Parameters
    ----------
    labels : list
        A list of labels that you want to check if they exist in the index.
    index : pd.Index
        Represents the index values against which the labels will be checked
    errors : Annotated[str, "ignore", "warn", "raise"], optional
        How to handle errors when a label is not found in the index

    Returns
    -------
        a list of labels that exist in the given index.

    """
    if min(len(labels), len(index)) == 0:
        return labels
    try:
        is_in_index = np.array([el in index for el in labels])
        assert all(is_in_index) or (errors == "ignore"), np.choose(
            np.where(~is_in_index), labels
        )[0]

    except AssertionError as err:
        msg = f"Labels not in index: {list(err.args[0])}"
        if errors == "warn":
            logging.warning(msg)
        elif errors == "raise":
            raise ValueError(msg) from err
        else:
            raise ValueError(f"errors arg '{errors}' is invalid.") from err

    which_labels = np.where(is_in_index)

    return np.array(labels)[which_labels]


def drop_labels(data: pd.DataFrame | pd.Series,
                drop: Hashable | Sequence[Hashable] = None,
                keep: Hashable | Sequence[Hashable] = None,
                inplace: bool = False
                ) -> pd.DataFrame | pd.Series | None:
    """Drop labels from index or columns of a DataFrame or Series

    Specify either `keep` or `drop`, but not both.
    """
    if keep is not None:
        if drop is not None:
            raise TypeError("Can't specify both `keep` and `drop`.")

        drop_cols = set(data.columns).difference(keep)
        drop_levels = set(data.index.names).difference(keep)

    elif drop is not None:
        drop = set(drop)
        drop_cols = drop.intersection(data.columns)
        drop_levels = drop.intersection(data.index.names)

    else:
        raise TypeError('`keep` and `drop` were both None.')

    drop_cols = list(drop_cols)
    drop_levels = list(drop_levels)
    if inplace:
        data.drop(drop_cols, axis=1, inplace=True)
        data.reset_index(drop_levels, drop=True, inplace=True)

    else:
        return data.drop(drop_cols, axis=1).reset_index(drop_levels, drop=True)




#TODO tests
def group_identical_rows(
    data: pd.DataFrame,
    id_col: Hashable,
    value_col: Hashable,
    colnames: Hashable | Sequence[Hashable] = None,
    **sort_kws,
) -> pd.Series:
    """Generates a dataframe of unique rows and lists of indices corresponding
    to groups of identical rows and each group assigned a unique alias

    Parameters
    ----------
    data : pd.DataFrame
        A pandas DataFrame containing the data to be grouped and deduplicated.
    id_col: Hashable
        Label of column containing ids to group (e.g. "SAMPLE") (as row index)
    value_col: Hashable
        Label of column containing values for comparing entries  pivot_cols: Hashable or sequence of Hashables  Column label(s) to use as columns in pivot table
    colnames: Hashable or Sequence of Hashables
        names in columns/index to unstack
    **sort_kws: passed to sort_index on columns (axis=1)

    Returns
    -------
    pd.Series
        Deduplicated data with index as the unique rows in `data` and values as
        lists of ids
    """
    def _sort_data(_data, _key, **_sort_kws):
        return _data.sort_values(sort_key=_key, **_sort_kws)

    if not isinstance(id_col, Hashable):
        raise TypeError(
            "`id_col` should be a single (hashable) label",
            type(id_col), id_col)
    
    if not isinstance(value_col, Hashable):
        raise TypeError(
            "`value_col` should be a single (hashable) label",
            type(value_col), value_col)

    if colnames is None:
        colnames = [*data.index.names, *data.columns]
        colnames.remove(id_col)
        colnames.remove(value_col)

    # put id_col in index to accommodate multiindex columns
    extra_index = list(set(colnames).difference([*data.index.names, id_col]))

    broad = (
        data
        .set_index(extra_index, append=True)
        .unstack(colnames)
        .droplevel(0, axis=1)
        .sort_index(axis=0)  # sort in prep for pivot_table
        .reset_index()
        .sort_index(axis=1)
        .pipe(lambda df: df.pivot_table(
            id_col, df.columns.drop(id_col).to_list(), aggfunc=lambda x: x
        ))
        .sort_index(axis=1, **sort_kws)
        .squeeze(axis=1)
        .rename(id_col)
    )
    # broad.rename({broad.columns[0]: id_col}, axis=1, inplace=True)
    return broad



def get_level_uniques(
    df: pd.MultiIndex | NDFrame, name: str, axis: int | str = 0
) -> pd.Index:
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


def ordered_fillna(
    df: pd.DataFrame,
    label: Hashable,
    order: Hashable | Sequence[Hashable],
    inplace: bool = False,
) -> pd.DataFrame | None:
    """Progressive fillna according to an ordered `order` list

    gff3: pd.DataFrame
        DataFrame must contain columns in `order`
    label:  Hashable
        Column label to fill, either existing or new. Default "name"
    order : Hashable or sequence of Hashables
        Field or ordered list of fields by which to populate the column
    inplace: bool, optional
        Do inplace

    Returns:
    pd.DataFrame or None
    """
    # get order priority
    if isinstance(order, Hashable):
        order = [order]

    if not inplace:
        df = df.copy()

    # set 'name' field
    if label not in df:
        df[label] = df[order[0]]

    for field in order:
        df[label].fillna(field, inplace=True)

    if not inplace:
        return df


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
    index_cols = list(reset_df.columns[: df.index.nlevels])

    pivoted = reset_df.pivot_table(*args, index=index_cols, **kwargs).rename_axis(
        df.index.names, axis=0
    )

    return pivoted

def rank_sort(
    data: pd.Series | pd.DataFrame,
    func: Callable = None,
    func_kws: dict = None,
    keep_col: bool = False,
    inplace: bool = False,
    **sort_kws
):
    """Ranks data based on `func` and sorts based on rank metric and index

    data: pd.Series | pd.DataFrame
        Data with values to rank
    func: Callable, optional
        How to rank. function takes a row from `data` and returns a Hashable.
        Default lambda x: 1 if isinstance(x, Hashable) else len(x)
    keep: bool, optional
        Keep computed "_rank" metric column. Default False.
    inplace: bool, optional
        Modify `data` inplace. Default False
    func_kws: dict, optional
        Kws to pass to func
    sort_kws: dict, optional
        passed to pandas `sort_values()` function. Note that values are sorted
        by '_rank' first, followed by the other columns

    Returns:
    Rank-sorted pd.Series or pd.DataFrame or None
        `data` as pd.Series if `data` is a pd.Series and `keep_col=False`.
        Else if inplace=True, returns None
        Else, `data` is returned as a pd.DataFrame.
    """
    if isinstance(data, pd.DataFrame) and "_rank" in data.columns:
        raise ValueError("'_rank' cannot be a column in data." )

    if inplace and keep_col and isinstance(data, pd.Series):
        raise TypeError("Cannot modify pd.Series inplace if keep_col=True")

    # TODO: need checks or explicit handling of weird datatypes

    # set defaults
    if func is None:
        func = lambda x: 1 if isinstance(x, Hashable) else len(x)

    # calculate rank values on which to sort
    if func_kws is None:
        func_kws = {}

    if isinstance(data, pd.DataFrame):
        func_kws["axis"] = 1

    ranking_metric = data.apply(func, **func_kws).rename("_rank")

    sort_kws = {"level": ["_rank", *data.index.names]} | sort_kws

    if isinstance(data, pd.DataFrame):
        if not inplace:
            data = data.copy()

        # insert ranking col into dataframe and sort
        data.insert(0, "_rank", ranking_metric)
        data.set_index("_rank", inplace=True, append=True)
        data.sort_index(inplace=True, **sort_kws)

        if not keep_col:
            data.droplevel("_rank", inplace=True)

        if not inplace:
            return data

    elif isinstance(data, pd.Series):

        # put data and ranks together and sort
        data_rank = (
            pd.concat([ranking_metric, data], axis=1)
            .set_index("_rank", append=True)
            .sort_index(**sort_kws)
        )

        if inplace:
            # generate hash where index corresponds to `data` index and
            # values are the rank metric
            rank_hash = pd.Series(range(len(data_rank)), index=data_rank.index)
            data.sort_values(key=lambda x: rank_hash.loc[x], inplace=True)

        elif not keep_col:  # not inplace
            return data_rank.droplevel("_rank")

        else:
            return data_rank




def reorder_cols(
    df: pd.DataFrame,
    first: Hashable | Sequence[Hashable] | pd.Index | None = None,
    last: Hashable | Sequence[Hashable] | pd.Index | None = None,
    inplace: bool = False,
    sort: bool = False,
    key: Callable = None,
    reverse: bool = None,
    errors: Annotated[str, "ignore", "warn", "raise"] = "ignore",
    **kws
) -> pd.DataFrame | None:
    """Reorders columns of dataframe.

    Arguments:
        df: pd.DataFrame
        first: column label, list of labels or pd.Index to put first
        last: column label, list of labels or pd.Index to put last
        inplace: bool, default False
            Note: this is deprecated. Use 'reverse' and 'key' kwargs
            
        sort: bool, optional
            Sort middle columns. Default False.
        key: Callable, optional
            Sort middle cols by this. (passed to `sorted`). Default None
        reverse: Callable, optional
            passed to `sorted. Default None

        **kws:
        
        Deprecated:
        ----------
        ascending: bool, default None
            how to sort remaining columns, where True is ascending, False is
            descending, and None is not sorted.
                        
        sort_kws: dict, default None
            kwargs to pass to `sorted()` on middle columns. Use `reverse` and
            `key`
            
            
    Returns: pd.DataFrame if inplace is False, else None.
    """
    # check input df object is not empty
    if len(df.columns) == 0:
        if errors == "raise":
            raise ValueError("Empty DataFrame")
        elif errors == "warn":
            logging.warning("Warning: Empty DataFrame")
        return df

    # defaults
    try:
        assert key is None
        key = kws["sort_kws"]["key"]
    except (AssertionError, KeyError):
        pass
    
    try:
        assert reverse is None
        reverse = kws["sort_kws"].get(
            "reverse", kws["sort_kws"].get("ascending"))
    except (AssertionError, KeyError):
        pass
            

    # check if first and last cols are in df (filter out ones that aren't)
    if first is None:
        first = []
    else:
        if isinstance(first, Hashable):
            first = default(first, [], [first])
        elif not isinstance(first, (pd.Index, MutableSequence)):
            first = list(first)
            first = default(first, [])

        first = check_if_exists(first, df.columns, errors=errors)

    if last is None:
        last = []
    else:
        if isinstance(last, Hashable):
            last = default(last, [], [last])
        elif not isinstance(last, (pd.Index, MutableSequence)):
            last = list(last)
            last = default(last, [])

        last = check_if_exists(last, df.columns, errors=errors)

    # list of other (unspecified) columns
    mid = df.columns.drop([*first, *last])

    if sort:
        mid = sorted(mid, key=key, reverse=reverse)

    if inplace:
            
        left = [*first, *mid]
        right = last

        # check sizes to minimize the number of columns we need to move
        if len(left) < len(right):
            
            # left has fewer than right, so move the left side columns
            # store left side before dropping from df
            left_df = df.loc[:, left]

            # drop left columns inplace
            df.drop(left, axis=1, inplace=True)
            
            # reverse to insert at 0 during .apply
            for label in reversed(left):

                # put columns back into the df in the new order
                df.insert(0, label, left_df[label])
                
        else:  
            # prefer to keep left in place so id(df) is the same
            # store left side before dropping from df
            
            # store right side before dropping
            right_df = df.loc[:, right]

            # drop right columns inplace
            df.drop(right, axis=1, inplace=True)

            # put columns back into the df in the new order
            i = len(df.columns)
            
            for label in right:
                df.insert(i, label, right_df[label])
                i += 1
    else:
        return pd.concat([df.loc[:, first], df.loc[:, mid], df.loc[:, last]],
                        axis=1)



def sort_levels(
    df: NDFrame | pd.MultiIndex, axis: int = 0, **sort_kws
) -> NDFrame | pd.MultiIndex:
    """Sort index or column levels of a MultiIndex (in a Series or DataFrame)

    axis: sort index (0) or column(1) levels (if DataFrame is passed)
    **sort_kws passed to sort_values function, excluding 'by' and 'inplace'
    """

    def _order_levels(_multiidx: pd.MultiIndex, **_sort_kws) -> pd.Series:
        """Core function for sorting levels"""
        names = pd.DataFrame(_multiidx.names)
        return names.sort_values(by=names.columns.to_list(), **_sort_kws).apply(
            tuple, axis=1
        )

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


def swap_index(
    frame: NDFrame,
    keys: Hashable | Sequence[Hashable] = None,
    inplace: bool = False,
    **kws,
) -> NDFrame | None:
    """Swaps a column in for an index without dropping the reset index

    frame: pd.Series or pd.DataFrame
    keys:  keys or list of keys
        Column name(s) to set as new index

    """

    def _swap_index(_frame, _keys, **_kws):
        return _frame.reset_index(**_kws).set_index(_keys)

    def _get_set_names(_frame_names, _fmt: str = "index_{}"):
        return [default(name, _fmt.format(i)) for i, name in enumerate(_frame_names)]

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
            return (
                frame.rename_axis(names).pipe(_swap_index, keys, *kws).squeeze(axis=1)
            )

        case _:
            raise TypeError("Bad 'frame' type", type(frame))
