# TODO docstring
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits

from astrostreampy.BuildModel.aperture import fwhm_mask_from_paramtab
from astrostreampy.Image.measure import StreamProperties

__all__ = ["measure"]


class _ColorProperties:
    # TODO docstring
    def __init__(self, measurement: str):
        raise NotImplementedError
        try:
            with open(measurement, encoding="utf-8") as f:
                header = f.readline().lstrip("#").strip()
                self._data = pd.read_csv(f, names=header.split(), sep="\s+")
        except:
            # only a small error handling if the measurement txt could not be read
            # maybe unnecessary, as the measurement file should be correctly created by astrostreampy
            raise IOError(
                f"Measurement file {measurement} is corrupt, not found or theres something wrong with the content."
            )

        # set properties
        self.mapp = self._data["m"]
        self.mapp_err = [self._data["m_err+"], self._data["m_err-"]]
        self.mtot = self._data["M"]
        self.mtot_err = [self._data["M_err+"], self._data["M_err-"]]
        self.filter = self._data["filter"]


def _mv_func(g: float, gr: float):
    raise NotImplementedError
    # Computes the visual magnitude from the g band brightness
    # and the g-r color.
    return g - 0.59 * gr + 0.0269


def _lum_func(mv: float):
    raise NotImplementedError
    # Computes the solar luminosities from the visual magnitude
    # and the visiual birghtness of the sun.
    return 10 ** (0.4 * (4.83 - mv))


def _select_colorfile():
    # TODO docstring
    # TODO create plot to interactively choose the best
    # TODO aperture from the models in different filters
    raise NotImplementedError


def _measure_color(measurements: list[str], filename: str):
    # TODO docstring
    raise NotImplementedError
    cps = [_ColorProperties(m) for m in measurements]  # cps = color properties
    res = ""
    header = "#"
    i = 1
    # TODO add check if filenaem already exists
    with open(filename, "w", encoding="utf-8") as f:
        while i < len(cps):
            # load data in pairs for the color calculation
            a = cps[i - 1]
            b = cps[i]
            header += f"{a.filter}-{b.filter} "
            res += f"{a.mtot}-{b.mtot} "
            i += 1
            if f"{a.filter}-{b.filter}" == "g-r":
                mv = _mv_func(a.mtot, a.mtot - b.mtot)
                lum = _lum_func(mv)
                header += "M_V L_S "
                res += f"{mv} {lum} "


def measure(
    multifitsfile: str,
    parameterfile: str,
    color_multifitsfile: str,
    color_parameterfile: str,
    maskfiles: list[str],
    errorfile: list[str],
    output: str = "streampy",
    zeropoint: float = None,
    zeropoint_type: str = "ZP",
    redshift: float = None,
    pixelscale: float = None,
    overwrite: bool = False,
):
    # TODO
    # create stream instances from multifits and param file (TODO write class)
    # measure in that class with the __init__
    # pass both instances into the 'measure_color' method
    # there select the best aperture, merges masks, measure the color!
    # for Mv and Lsolar the people have to do it themselfes beacaus of different filter systems /conversions

    stream = StreamProperties(
        multifitsfile,
        parameterfile,
        maskfiles,
        redshift=redshift,
        zeropoint=zeropoint,
        zero_point_type=zeropoint_type,
        pixelscale=pixelscale,
    )

    stream.measure(
        errorfile=errorfile,
        color_multifits_file=color_multifitsfile,
        color_parameter_file=color_parameterfile,
    )

    stream.writeto(filename=output, overwrite=overwrite)
