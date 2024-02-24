"""Provides convenience methods to handle parsed arguments."""

import numpy as np


def handle_steps(arg: str) -> np.ndarray:
    """
    Handles the -s (--step) argument.
    Transform the argument into an np.ndarray.

    Parameters
    ----------
    arg : str
        A string argument.
    """
    return np.array(arg.split(","), dtype=int)


def handle_infile(arg: object) -> str:
    """
    Handles the INFILE argument.
    Transforms the argument into a string.

    Parameters
    ----------
    arg : str
        A object argument.
    """
    if arg is None:
        raise ValueError("infile needed")
    else:
        infile = arg.name
        arg.close
    return infile


def handle_fileargs(arg: str) -> list:
    """
    Handles the masks and interpolation masks arguments -mf and -imf.
    Transfroms the string into a list.

    Parameters
    ----------
    arg : str
        A string argument.
    """
    if not isinstance(arg, str):
        l = []
        return l

    l = arg.split(",")
    if arg.endswith(","):
        l.pop()
    return l
