"""
Provides a class to measure and extract brightness and shape
properties of extragalactic tidal streams.

Example
-------
>>> from measure import StreamProperties
"""

import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
from astropy.io import fits
from scipy.stats import sem

from ..BuildModel.aperture import effective_mask_from_paramtab, fwhm_mask_from_paramtab
from ..BuildModel.utilities import fit_func
from .track import StreamTrack
from .utilities import effective_width, fwhm_width

_H_0 = 69.6  # km / s / Mpc
_H = _H_0 / 100.0
_OMEGA_M = 0.286
_OMEGA_VAC = 0.714
_OMEGA_R = 4.165e-5 / _H**2
_OMEGA_K = 1 - _OMEGA_M - _OMEGA_R - _OMEGA_VAC
_C = 299792.458  # km/s
_N = 1000  # precision of the co-moving radial distance integral

__all__ = ["StreamProperties"]


class StreamProperties:
    # TODO refactor as dataclass with subclasses
    """
    A class to analyze properties of a stream from astronomical data.

    Parameters
    ----------
    multifits_file : str
        Path to the multifits file containing astronomical data.
    parameter_file : str
        Path to the parameter file containing stream parameters.
    maskfiles : list of str, optional
        List of mask files for masking irrelevant data. Defaults to None.
    zero_point_type : {'ZP', 'ZP50'}, optional
        Type of zero point for calculating absolute magnitudes. Defaults to 'ZP'.

    Attributes
    ----------
    multifits_file : str
        Path to the multifits file containing astronomical data.
    parameter_file : str
        Path to the parameter file containing stream parameters.

    Methods
    -------
    measure(errorfile=None)
        Measures various properties of the stream from the data.
    writeto(filename, overwrite=False, addsuffix=True)
        Writes the measured properties to a .txt file.

    Example
    -------
    >>> measurement = StreamProperties('multifits_file.fits','paramfile.fits',
    >>>                                ['sourcemask.fits','interpolationmask.fits'],
    >>>                                'ZP')
    >>> measurement.measure('errorfile.fits')
    >>> measurement.writeto('results.txt')
    >>> print(measurement) # displays formatted stream properties to the console
    """

    def __init__(
        self,
        multifits_file: str,
        parameter_file: str,
        maskfiles: list[str] = None,
        zero_point_type: Literal["ZP,ZP50"] = "ZP",
        redshift: float = None,
        pixelscale: float = None,
        zeropoint: float = None,
    ):
        if isinstance(maskfiles, list):
            self._maskfiles = maskfiles
        else:
            self._maskfiles = []

        if zero_point_type not in ["ZP", "ZP50"]:
            raise ValueError("invalid option for zero_point_type")
        self._zero_point_type = zero_point_type
        self.multifits_file = multifits_file
        self.parameter_file = parameter_file
        self._data: np.ndarray = np.nan_to_num(
            fits.getdata(multifits_file, ext=2), nan=0
        )
        self._error_data = np.zeros(self._data.shape)
        self._header = fits.getheader(multifits_file, ext=0)
        self._aperture: np.ndarray = fits.getdata(multifits_file, ext=5)
        self._center_aperture: np.ndarray = np.zeros(self._aperture.shape)
        self._border_aperture: np.ndarray = np.zeros(self._aperture.shape)
        self._aperture_type: str = "1FWHM"
        self._model: np.ndarray = fits.getdata(multifits_file, ext=4)
        self._fillmask: np.ndarray = np.ones(self._aperture.shape)
        self._parameter_table = fits.getdata(parameter_file, ext=1)

        # TODO add try block for proper header key handling
        if pixelscale is None:
            self._pixel_scale = self._header["PXSCALE"]
        else:
            self._pixel_scale = pixelscale

        if zeropoint is None:
            self._zero_point = self._header[zero_point_type]
        else:
            self._zero_point = zeropoint

        self._color_measurement = False
        # init all properties
        self._filter = fits.getval(filename=multifits_file, keyword="FILTER", ext=0)
        self._surface_brightness: float = np.nan
        self._surface_brightness_error: list[float, float] = [
            np.nan,
            np.nan,
        ]  # +error, -error
        self._apparent_magnitude: float = np.nan
        self._apparent_magnitude_error: list[float, float] = [
            np.nan,
            np.nan,
        ]  # +error, -error
        self._total_apparent_magnitude: float = np.nan
        self._total_apparent_magnitude_error: list[float, float] = [
            np.nan,
            np.nan,
        ]  # +error, -error
        self._absolute_magnitude: float = np.nan
        self._absolute_magnitude_error: list[float, float] = [
            np.nan,
            np.nan,
        ]  # +error, -error
        self._total_absolute_magnitude: float = np.nan
        self._total_absolute_magnitude_error: list[float, float] = [
            np.nan,
            np.nan,
        ]  # +error, -error
        self._effective_radius: float = np.nan
        self._effective_radius_error: float = np.nan
        self._mean_effective_surface_brightness: float = np.nan
        self._mean_effective_surface_brightness_error: list[float, float] = [
            np.nan,
            np.nan,
        ]
        self._effective_surface_brightness: float = np.nan
        self._effective_surface_brightness_error: list[float, float] = [
            np.nan,
            np.nan,
        ]
        self._length: float = np.nan
        self._length_error: float = np.nan
        self._width: float = np.nan
        self._width_error: float = np.nan
        if redshift is None:
            self._redshift: float = self._header["ZSTREAM"]
        else:
            self._redshift = redshift
        (
            self._distance_modulus,
            self._luminosity_distance,
            self._kpc_scale,
        ) = self._calc_distmod()

        self._background: float = 0.0
        self._background_error: float = 0.0
        self._data_flux: float = 0.0
        self._dimmed_data_flux: float = 0.0
        self._error_flux: float = 0.0
        self._aperture_type = "1FWHM"

    def __str__(self):
        # TODO use a string formatter
        f = self._val_to_str
        out_string = ""
        out_string += "o surface brightness (cosmic dimming)\n"
        out_string += (
            f"\t SB\t= {f(self._surface_brightness)} {self._filter} mag/arcsec\u00b2\n"
        )
        out_string += f"\t \u0394SB\t= {f(self._surface_brightness_error)} {self._filter} mag/arcsec\u00b2\n"
        out_string += "o apparent magnitude\n"
        out_string += (
            f"\t m_tot\t= {f(self._total_apparent_magnitude)} {self._filter} mag\n"
        )
        out_string += f"\t \u0394m_tot\t= {f(self._total_apparent_magnitude_error)} {self._filter} mag\n"
        out_string += "o absolute magnitude\n"
        out_string += (
            f"\t M_tot\t= {f(self._total_absolute_magnitude)} {self._filter} mag\n"
        )
        out_string += f"\t \u0394M_tot\t= {f(self._total_absolute_magnitude_error)} {self._filter} mag\n"
        out_string += "o effective radius\n"
        out_string += f"\t r\t= {f(self._effective_radius)} kpc\n"
        out_string += f"\t \u0394r\t= \u00b1{f(self._effective_radius_error)} kpc\n"
        out_string += "o effective surface brightness (cosmic dimming)\n"
        out_string += f"\t <SBe>\t= {f(self._mean_effective_surface_brightness)} {self._filter} mag/arcsec\u00b2\n"
        out_string += f"\t \u0394<SBe>\t= {f(self._mean_effective_surface_brightness_error)} {self._filter} mag/arcsec\u00b2\n"
        out_string += "o surface brightness at effective radius (cosmic dimming)\n"
        out_string += f"\t SBe\t= {f(self._effective_surface_brightness)} {self._filter} mag/arcsec\u00b2\n"
        out_string += f"\t \u0394SBe\t= {f(self._effective_surface_brightness_error)} {self._filter} mag/arcsec\u00b2\n"
        out_string += "o width\n"
        out_string += f"\t w\t= {f(self._width)} kpc\n"
        out_string += f"\t \u0394w\t= \u00b1{f(self._width_error)} kpc\n"
        out_string += "o length\n"
        out_string += f"\t l\t= {f(self._length)} kpc\n"
        out_string += f"\t \u0394l\t= \u00b1{f(self._length_error)} kpc"
        print(out_string)
        return ""

    def _calc_distmod(self):
        z = self._redshift
        z_time = 0  # time from z to now
        cmrd = 0  # co-moving radial distance
        a_z = 1.0 / (1.0 + z)

        for i in range(_N):
            a = a_z + (1 - a_z) * (i + 0.5) / _N
            a_dot = np.sqrt(
                _OMEGA_K + (_OMEGA_M / a) + (_OMEGA_R / (a * a)) + (_OMEGA_VAC * a * a)
            )
            z_time += 1.0 / a_dot
            cmrd += 1.0 / (a * a_dot)

        z_time *= (1 - a_z) / _N
        cmrd *= (1 - a_z) / _N
        x = np.sqrt(abs(_OMEGA_K)) * cmrd
        if x > 0.1:
            if _OMEGA_K > 0:
                ratio = 0.5 * (np.exp(x) - np.exp(-x)) / x
            else:
                ratio = np.sin(x) / x
        else:
            y = x**2
            if _OMEGA_K < 0:
                y = -y
            ratio = 1.0 + y / 6.0 + y * y / 120.0

        cmtd = ratio * cmrd
        dA = a_z * cmtd
        dA_Mpc = dA * _C / _H_0
        kpc_scale = dA_Mpc / 206.264806
        dL = dA / a_z**2
        dL_Mpc = dL * _C / _H_0
        mu = 5 * np.log10(dL_Mpc * 1e6) - 5
        return mu, dL_Mpc, kpc_scale

    def _val_to_str(self, val: Any) -> str:
        if isinstance(val, list):
            return f"+{self._val_to_str(val[0])}|-{self._val_to_str(abs(val[1]))}"
        if np.isnan(val):
            return "Nan"
        return str(val)

    def _create_fillmask(self) -> None:
        for file in self._maskfiles:
            maskdata = fits.getdata(file)
            self._fillmask *= maskdata
        self._fillmask = (self._fillmask * self._aperture) + np.invert(
            self._aperture.astype(bool)
        )

    def _fill_zero_pixels(self) -> None:
        empty_pixels = np.where(self._fillmask == 0)
        self._data[empty_pixels] = self._model[empty_pixels]

    def _flux_inside_aperture(self) -> None:
        self._data_flux = np.sum((self._data - self._background) * self._aperture)
        self._dimmed_data_flux = self._data_flux * (1 + self._redshift) ** 4
        self._error_flux = np.sqrt(np.sum(np.square(self._error_data * self._aperture)))
        # TODO add future logger to log something like this
        print(f"[INFO] Data Flux = {self._data_flux}")
        print(f"[INFO] Dimmed Data Flux = {self._dimmed_data_flux}")
        print(f"[INFO] Error Flux = {self._error_flux}")

    def _calc_bg_from_offsets(self):
        if np.isnan(np.mean(self._parameter_table[f"offset_{self._filter}_err"])):
            self._background = np.median(
                self._parameter_table[f"offset_{self._filter}"]
            )
        else:
            self._background = np.average(
                self._parameter_table[f"offset_{self._filter}"],
                weights=1 / self._parameter_table[f"offset_{self._filter}_err"],
            )

        self._background_error = sem(self._parameter_table[f"offset_{self._filter}"])

    def _calc_log_flux(self, val, zeropoint: float = None):
        if zeropoint is None:
            zeropoint = self._zero_point
        return -2.5 * np.log10(val) + zeropoint

    def _calc_magnitude_error(self, flux, error_flux, background_error_flux) -> list:
        error_minus = self._calc_log_flux(
            (flux - background_error_flux - error_flux) / flux,
            zeropoint=0,
        )
        error_plus = self._calc_log_flux(
            (flux + background_error_flux + error_flux) / flux,
            zeropoint=0,
        )
        return error_minus, error_plus

    def _calc_apparent_magnitude(self):
        apparent_magnitude = self._calc_log_flux(self._data_flux)
        apparent_magnitude_error = self._calc_magnitude_error(
            self._data_flux,
            self._error_flux,
            np.sum(self._aperture) * self._background_error,
        )
        if self._aperture_type == "1FWHM":
            self._apparent_magnitude = apparent_magnitude
            self._apparent_magnitude_error[:] = apparent_magnitude_error
        if self._aperture_type == "3FWHM":
            self._total_apparent_magnitude = apparent_magnitude
            self._total_apparent_magnitude_error[:] = apparent_magnitude_error

    def _calc_absolute_magnitude(self):
        if self._aperture_type == "1FWHM":
            self._absolute_magnitude = self._apparent_magnitude - self._distance_modulus
            self._absolute_magnitude_error = self._apparent_magnitude_error.copy()
        if self._aperture_type == "3FWHM":
            self._total_absolute_magnitude = (
                self._total_apparent_magnitude - self._distance_modulus
            )
            self._total_absolute_magnitude_error = (
                self._total_apparent_magnitude_error.copy()
            )

    def _calc_surface_brightness(self):
        npix = np.sum(self._aperture)
        surface_brightness = self._calc_log_flux(
            self._dimmed_data_flux / npix / self._pixel_scale**2
        )
        surface_brightness_error = self._calc_magnitude_error(
            self._dimmed_data_flux,
            self._error_flux,
            npix * self._background_error,
        )
        if self._aperture_type == "1FWHM":
            self._surface_brightness = surface_brightness
            self._surface_brightness_error[:] = surface_brightness_error

        if self._aperture_type == "EFF":
            self._mean_effective_surface_brightness = surface_brightness
            self._mean_effective_surface_brightness_error[:] = surface_brightness_error

    def _calc_effective_surface_brightness(
        self,
    ) -> None:  # the effective surface brightness at the effective radius
        npix = np.sum(self._aperture)

        self._effective_surface_brightness = self._calc_log_flux(
            self._dimmed_data_flux / npix / self._pixel_scale**2
        )
        self._effective_surface_brightness_error[:] = self._calc_magnitude_error(
            self._dimmed_data_flux,
            self._error_flux,
            npix * self._background_error,
        )

    def _calc_length(self):
        stream_track = StreamTrack(self.parameter_file, self.multifits_file)
        stream_track.fit_spline()
        self._length = stream_track.length
        self._length = self._length * self._pixel_scale * self._kpc_scale
        self._length_error = (
            (2 + stream_track.bin) * self._pixel_scale * self._kpc_scale
        )

    def _calc_widths(self):
        fwhm_widths = []
        eff_widths = []
        for row in self._parameter_table:
            w, h, sigma, norm, h2, skew, h4 = np.array(row)[
                np.array([2, 3, 6, 8, 16, 18, 20])
            ]
            model = fit_func(np.max([w, h]), sigma, norm, 0, h2, skew, h4)  # 0 = offset
            fwhm_widths.append(fwhm_width(model))
            eff_widths.append(effective_width(model))
        self._width = np.median(fwhm_widths) * self._pixel_scale * self._kpc_scale
        self._width_error = 2 * self._pixel_scale * self._kpc_scale
        self._effective_radius = (
            np.median(eff_widths) * self._pixel_scale * self._kpc_scale
        )
        self._effective_radius_error = 2 * self._pixel_scale * self._kpc_scale

    def _set_error_data(self, errorfile: str = None) -> None:
        try:
            print(f"[INFO] Got error data from {errorfile}")
            self._error_data = np.nan_to_num(fits.getdata(errorfile), nan=0)
        except:
            print("[INFO] No errorfile given!")

    def _prepare(self, measure_model: bool = False):
        if not measure_model:
            self._create_fillmask()
            if self._color_measurement:
                print(
                    "[INFO] Masking pixel from source and int masks and multiply onto aperture"
                )
                self._aperture *= self._fillmask
            self._fill_zero_pixels()
        self._flux_inside_aperture()

    def _measure_shape(self) -> None:
        self._calc_length()
        self._calc_widths()

    def _measure_brightness(self):
        if self._aperture_type == "EFF":
            self._calc_surface_brightness()
            self._aperture = self._border_aperture
            self._prepare()
            self._calc_effective_surface_brightness()
        if self._aperture_type == "1FWHM":
            self._calc_apparent_magnitude()
            self._calc_absolute_magnitude()
            self._calc_surface_brightness()
        if self._aperture_type == "3FWHM":
            self._calc_apparent_magnitude()
            self._calc_absolute_magnitude()

    def _set_aperture(
        self, aperture_type: Literal["FWHM", "EFF"], k: int = 1, smoothing: int = 10
    ) -> None:
        if aperture_type == "FWHM":
            self._aperture, self._border_aperture, self._center_aperture = (
                fwhm_mask_from_paramtab(
                    self.parameter_file, self.multifits_file, k, smoothing
                )
            )
            self._aperture_type = f"{k}FWHM"
        if aperture_type == "EFF":
            self._aperture, self._border_aperture = effective_mask_from_paramtab(
                self.parameter_file, self.multifits_file, smoothing
            )
            self._aperture_type = "EFF"

    def measure(
        self,
        errorfile: str = None,
        bg: float = None,
        bg_err: float = 0,
        measure_model: bool = False,
        color_parameter_file: str = None,
        color_multifits_file: str = None,
    ) -> None:
        """
        Method to measure the stream with a given aperture.
        It measures:
            - surface brightness (inside a 1xFWHM aperture)
            - apparent magnitude (inside a 3xFWHM aperture)
            - absolute magnitude (inside a 3xFWHM aperture)
            - effective surface brightness (inside a 1xEFF aperture)
            - effective surface brightness (at effective radius aperture)
            - effective radius
            - length
            - width

        Parameters
        ----------
        errorfile : str, optional
            Filename of the error file. Default is None.
            If none is parsed the error data is set to a
            numpy ndarray filled with zeros.

        bg : float, optional
            Value of the local background. Default is None.
            Use this argument if the background is known, or
            to measure on the model.

        bg_err : float, optional
            Uncertainty of the local background. Default is 0.

        measure_model : bool, optional
            Activates measuring on the model in addition
            to the real stream data. Default is False.
            If ``True`` `writeto()` writes the measured
            properties to a seperate TXT file.
        """
        if measure_model:
            self._data = self._model.copy()

        self._set_error_data(errorfile=errorfile)
        if bg is None:
            self._calc_bg_from_offsets()
        else:
            self._background = bg
            self._background_error = bg_err
        self._measure_shape()

        if isinstance(color_parameter_file, (str, Path)) and isinstance(
            color_multifits_file, (str, Path)
        ):
            print(f"measuring {self.multifits_file} ...")
            self.parameter_file = color_parameter_file
            self.multifits_file = color_multifits_file
            print(f"... using {self.multifits_file} as colorfile!")
            self._color_measurement = True
        print(f"measuring with pixel scale: {self._pixel_scale}")
        for aperture_properties in [
            ["FWHM", 1, 10],
            ["FWHM", 3, 10],
            ["EFF", None, 10],
        ]:
            aperture_type, k, smoothing = aperture_properties
            print(
                f"measuring brightnesses with a {smoothing}x smoothed k={k} {aperture_type} aperture"
            )
            self._set_aperture(aperture_type, k, smoothing)
            self._prepare(measure_model)
            self._measure_brightness()

    def writeto(self, filename: str, overwrite: bool = False, addsuffix: bool = True):
        """
        Write the measured properties to a TXT file.

        Parameters
        ----------
        filename : str
            Filename of the output file.
            If filename does not end with .txt it is added as a suffix.

        overwrite : bool, optional
            Overwrites an already existing file with the same name.
            Default is False.

        addsuffix : bool, optional
            Adds '_measurements_FILTER_ZEROPOINTTYPE' suffix to `filename`.
            Default is True.

        """
        filename = filename.removesuffix(".txt")
        if addsuffix:
            filename += f"_measurements_{self._filter}_{self._zero_point_type}"
        if self._color_measurement:
            filename += "_color"
        filename += ".txt"

        if filename in os.listdir(".") and not overwrite:
            raise FileExistsError("File already exists. Use overwrite==True.")

        header = "# <SB> <SB>_err+ <SB>_err- "
        header += "m m_err+ m_err- "
        header += "M M_err+ M_err- "
        header += "m_tot m_tot_err+ m_tot_err- "
        header += "M_tot M_tot_err+ M_tot_err- "
        header += "<SB_eff> <SB_eff>_err+ <SB_eff>_err- "
        header += "SB_eff SB_eff_err+ SB_eff_err- "
        header += "r_eff r_eff_err "
        header += "l l_err "
        header += "w w_err "
        header += "z mu dL kpcscale "
        header += "filter\n"

        data = f"{self._surface_brightness} {self._surface_brightness_error[0]} {self._surface_brightness_error[1]} "
        data += f"{self._apparent_magnitude} {self._apparent_magnitude_error[0]} {self._apparent_magnitude_error[1]} "
        data += f"{self._absolute_magnitude} {self._absolute_magnitude_error[0]} {self._absolute_magnitude_error[1]} "
        data += f"{self._total_apparent_magnitude} {self._total_apparent_magnitude_error[0]} {self._total_apparent_magnitude_error[1]} "
        data += f"{self._total_absolute_magnitude} {self._total_absolute_magnitude_error[0]} {self._total_absolute_magnitude_error[1]} "
        data += f"{self._mean_effective_surface_brightness} {self._mean_effective_surface_brightness_error[0]} {self._mean_effective_surface_brightness_error[1]} "
        data += f"{self._effective_surface_brightness} {self._effective_surface_brightness_error[0]} {self._effective_surface_brightness_error[1]} "
        data += f"{self._effective_radius} {self._effective_radius_error} "
        data += f"{self._length} {self._length_error} "
        data += f"{self._width} {self._width_error} "
        data += f"{self._redshift} {self._distance_modulus} {self._luminosity_distance} {self._kpc_scale} "
        data += f"{self._filter}"
        file = open(filename, "w", encoding="utf-8")
        file.writelines([header, data])
        file.close()
