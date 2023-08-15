import os

import numpy as np
from astropy.io import fits


def add_psf_head(val):
    for file in os.listdir("."):
        if file.endswith(".fits") and file.startswith(("TS", "field")):
            fits.setval(file, "PSF", value=val)


def handle_steps(arg: str):
    return np.array(arg.split(","), dtype=int)


def handle_infile(arg: str):
    if arg == None:
        raise ValueError("infile needed")
    else:
        infile = arg.name
        arg.close
    return infile


def handle_fileargs(arg: str):
    if not isinstance(arg, str):
        l = []
        return l

    l = arg.split(",")
    if arg.endswith(","):
        l.pop()
    return l


def handle_colormask_arg(arg: str, cf_num: int):
    l = []

    if not isinstance(arg, str):
        for _ in range(cf_num):
            l.append([])
        return l

    l = arg.split(",")

    for i, val in enumerate(l):
        val = val.replace(":", ",")
        if val == "":
            l[i] = []
            continue
        val = handle_fileargs(val)
        l[i] = val

    return l


def handel_num_pair(arg: str) -> list:
    l = arg.split(",")
    if arg.endswith(","):
        # TODO good error message
        raise ValueError
    return list(map(int, l))
