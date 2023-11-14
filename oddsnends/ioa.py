"""ioa.py"""

# Input, output and args-related
import os
from argparse import ArgumentParser
from typing import Annotated

__all__ = [
    "LoggingLevels",
    "assertfile",
    "assertexists",
    "argtype_filepath",
]

LoggingLevels = {
    "NOTSET": 0,
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}

def assertfile(fpath: str | None, *err_args,
               null: Annotated[str, "ignore", "raise"] = "ignore") -> None:
    """Asserts isfile. Raises FileNotFoundError only if fpath is non-null."""
    try:
        assert os.path.isfile(fpath)
    except AssertionError as error:
        raise FileNotFoundError(fpath, *err_args) from error
    except TypeError as error:
        if (fpath is not None) or (null == "raise"):
            raise TypeError(fpath, type(fpath)) from error

def assertexists(fpath: str | None, *err_args,
               null: Annotated[str, "ignore", "raise"] = "ignore") -> None:
    """Asserts exists. Raises FileNotFoundError only if fpath is non-null."""
    try:
        assert os.path.exists(fpath)
    except AssertionError as error:
        raise FileNotFoundError(fpath, *err_args) from error
    except TypeError as error:
        if (fpath is not None) or (null == "raise"):
            raise TypeError(fpath, type(fpath)) from error

def argtype_filepath(parser: ArgumentParser, arg: str | None) -> str:
    """Check if arg given to parser is a valid file, raising ArgumentError
    otherwise
    """
    arg = arg.strip()
    try:
        assert os.path.isfile(arg)
    except (AssertionError, TypeError):
        parser.error(f"file does not exist: {arg}")
    return arg
