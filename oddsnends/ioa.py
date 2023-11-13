"""ioa.py"""

# Input, output and args-related
import gzip
import os
import pickle
from argparse import ArgumentParser
from colllections.abc import Callable, Generator
from typing import Annotated, Any

__all__ = [
    "LoggingLevels",
    "assertfile",
    "assertexists",
    "filepath",
    "find_fpaths",
    "read_pkl_gz",
]

LoggingLevels = {
    "NOTSET": 0,
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}


def assertfile(
    fpath: str | None, *err_args, null: Annotated[str, "ignore", "raise"] = "ignore"
) -> None:
    """Asserts isfile. Raises FileNotFoundError only if fpath is non-null."""
    try:
        assert os.path.isfile(fpath)
    except AssertionError as error:
        raise FileNotFoundError(fpath, *err_args) from error
    except TypeError as error:
        if (fpath is not None) or (null == "raise"):
            raise TypeError(fpath, type(fpath)) from error


def assertexists(
    fpath: str | None, *err_args, null: Annotated[str, "ignore", "raise"] = "ignore"
) -> None:
    """Asserts exists. Raises FileNotFoundError only if fpath is non-null."""
    try:
        assert os.path.exists(fpath)
    except AssertionError as error:
        raise FileNotFoundError(fpath, *err_args) from error
    except TypeError as error:
        if (fpath is not None) or (null == "raise"):
            raise TypeError(fpath, type(fpath)) from error


def filepath(parser: ArgumentParser, arg: str | None) -> str:
    """Check if arg given to parser is a valid file, raising ArgumentError
    otherwise
    """
    arg = arg.strip()
    try:
        assert os.path.isfile(arg)
    except (AssertionError, TypeError):
        parser.error(f"file does not exist: {arg}")
    return arg


def find_fpaths(
    rootdir: str,
    suffix: str = None,
    prefix: str = None,
    filter_fn: Callable = None,
    filter_kws: dict = None,
    **kws
) -> Generator[tuple[str, str]]:
    """Returns filenames and paths in root directory
    Parameters
    ----------
    rootdir:  str
        Directory to search
    suffix: str, optional
        Filter by suffix. Can use in combination with prefix. Default None
    prefix:
        Filter by prefix. Can use in combination with prefix. Default None
    filter_fn: functionn, optional
        Filter files. If given, the function should take the file name and dir
        as the first two positional arguments, and return a bool. suffix and 
        prefix args are ignored if this is given.
    filter_kws: dict, optional
        Passed to filter_fn
    **kws
        Passed to os.walk
    
    Returns
    -------
    Generator of (filename, filepath)
    """
    # defaults
    if filter_kws is None:
        filter_kws = {}
    
    # check/construct filter function
    if filter_fn is not None:
        pass

    elif (suffix is not None) and (prefix is not None):
        filter_fn = lambda s: s.startswith(prefix) and s.endswith(prefix)

    elif suffix is not None:
        filter_fn = lambda s: s.endswith(suffix)

    elif prefix is not None:
        filter_fn = lambda s: s.startswith(prefix)

    else:  # no filter
        filter_fn = lambda _: True

    # walk
    return (
        (fname, f"{d}/{fname}")
        for d, _, files in os.walk(rootdir, **kws)
        for fname in files
        if filter_fn(fname)
    )


def read_pkl_gz(fpath: str) -> Any:
    """Read in data from a gzipped pickle"""
    with gzip.open(fpath, "rb") as fin:
        return pickle.load(fin)
