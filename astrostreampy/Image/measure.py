import os
from typing import Any, Literal

import numpy as np
from astropy.io import fits
from astrostreampy.BuildModel.aperture import fwhm_mask_from_paramtab
from astrostreampy.Image.point import Point

_H_0 = 69.6  # km / s / Mpc
_H = _H_0 / 100.0
_OMEGA_M = 0.286
_OMEGA_VAC = 0.714
_OMEGA_R = 4.165e-5 / _H**2
_OMEGA_K = 1 - _OMEGA_M - _OMEGA_R - _OMEGA_VAC
_C = 299792.458  # km/s
_N = 1000  # precision of the co-moving radial distance integral


class StreamProperties:
    """
    Measures the Stream Properties.
    """

    def __init__(
        self,
        multifits_file: str,
        parameter_file: str,
        zero_point_type: Literal["ZP,ZP50"] = "ZP",
    ):
        if zero_point_type not in ["ZP", "ZP50"]:
            raise ValueError("invalid option for zero_point_type")
        self._zero_point_type = zero_point_type
        self.multifits_file = multifits_file
        self.parameter_file = parameter_file
        self._data: np.ndarray = np.nan_to_num(
            fits.getdata(multifits_file, ext=2), nan=0
        )
        self._error_data = np.zeros(
            self._data.shape
        )  # None #np.nan_to_num(fits.getdata(multifits_files,ext=6),nan=0)
        self._header = fits.getheader(multifits_file, ext=0)
        self._aperture = fits.getdata(multifits_file, ext=5)
        self._parameter_table = fits.getdata(parameter_file, ext=1)

        self._pixel_scale = self._header["PXSCALE"]
        self._zero_point = 30  # self._header[zero_point_type]

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
        self._absolute_magnitude: float = np.nan
        self._absolute_magnitude_error: list[float, float] = [
            np.nan,
            np.nan,
        ]  # +error, -error
        self._effective_radius: float = np.nan
        self._effective_radius_error: float = np.nan
        self._effective_surface_brightness: float = np.nan
        self._effective_surface_brightness_error: list[float, float] = [np.nan, np.nan]
        self._length: float = np.nan
        self._length_error: float = np.nan
        self._width: float = np.nan
        self._width_error: float = np.nan
        self._redshift: float = 0.023  # None #self._header["ZSTREAM"]
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
        f = self._val_to_str
        out_string = ""
        out_string += "o surface brightness (cosmic dimming)\n"
        out_string += (
            f"\t SB\t= {f(self._surface_brightness)} {self._filter} mag/arcsec\u00b2\n"
        )
        out_string += f"\t \u0394SB\t= {f(self._surface_brightness_error)} {self._filter} mag/arcsec\u00b2\n"
        out_string += "o apparent magnitude\n"
        out_string += f"\t m\t= {f(self._apparent_magnitude)} {self._filter} mag\n"
        out_string += (
            f"\t \u0394m\t= {f(self._apparent_magnitude_error)} {self._filter} mag\n"
        )
        out_string += "o absolute magnitude\n"
        out_string += f"\t M\t= {f(self._absolute_magnitude)} {self._filter} mag\n"
        out_string += (
            f"\t \u0394M\t= {f(self._absolute_magnitude_error)} {self._filter} mag\n"
        )
        out_string += "o effective radius\n"
        out_string += f"\t r\t= {f(self._effective_radius)} kpc\n"
        out_string += f"\t \u0394r\t= \u00b1{f(self._effective_radius_error)} kpc\n"
        out_string += "o effective surface brightness (cosmic dimming)\n"
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

    def _val_to_str(self, val: Any):
        if isinstance(val, list):
            return f"+{self._val_to_str(val[0])}|-{self._val_to_str(abs(val[1]))}"
        if np.isnan(val):
            return "Nan"
        return str(val)

    def _create_fillmask(self):
        raise NotImplementedError

    def _fill_zero_pixels(self):
        raise NotImplementedError

    def _flux_inside_aperture(self, aperture: np.ndarray = None):
        if aperture is None:
            aperture = self._aperture
        self._data_flux = np.sum((self._data - self._background) * aperture)
        self._dimmed_data_flux = self._data_flux * (1 + self._redshift) ** 4
        self._error_flux = np.sqrt(np.sum(np.square(self._error_data * aperture)))

    def _calc_bg_from_offsets(self):
        self._background = np.median(self._parameter_table[f"offset_{self._filter}"])
        self._background_error = np.sqrt(
            np.mean(np.square(self._parameter_table[f"offset_{self._filter}_err"]))
        )

    def _calc_log_flux(self, val):
        return -2.5 * np.log10(val) + self._zero_point

    def _calc_apparent_magnitude(self):
        self._apparent_magnitude = self._calc_log_flux(self._data_flux)
        self._apparent_magnitude_error[0] = (
            self._calc_log_flux(self._data_flux - self._error_flux)
            - self._apparent_magnitude
        )
        self._apparent_magnitude_error[1] = (
            self._calc_log_flux(self._data_flux + self._error_flux)
            - self._apparent_magnitude
        )

    def _calc_absolute_magnitude(self):
        self._absolute_magnitude = self._apparent_magnitude - self._distance_modulus
        self._absolute_magnitude_error = self._apparent_magnitude_error.copy()

    def _calc_surface_brightness(self):
        npix = np.sum(self._aperture)
        self._surface_brightness = self._calc_log_flux(
            self._dimmed_data_flux / npix / self._pixel_scale**2
        )
        self._surface_brightness_error[0] = (
            self._calc_log_flux(
                (self._dimmed_data_flux - self._error_flux)
                / npix
                / self._pixel_scale**2
            )
            - self._surface_brightness
        )
        self._surface_brightness_error[1] = (
            self._calc_log_flux(
                (self._dimmed_data_flux + self._error_flux)
                / npix
                / self._pixel_scale**2
            )
            - self._surface_brightness
        )

    def _calc_length(self):
        x_pos = self._parameter_table["box_x"]
        y_pos = self._parameter_table["box_y"]
        i = 1
        self._length = 0
        while i < len(x_pos):
            self._length += Point(x_pos[i], y_pos[i]).distance_to(
                Point(x_pos[i - 1], y_pos[i - 1])
            )
            i += 1
        self._length = self._length * self._pixel_scale * self._kpc_scale
        self._length_error = 2 * self._pixel_scale * self._kpc_scale

    def writeto(self, filename: str, overwrite: bool = False):
        filename = filename.removesuffix(".txt")
        filename += f"_measurements_{self._filter}_{self._aperture_type}_{self._zero_point_type}"
        filename += ".txt"

        if filename in os.listdir(".") and not overwrite:
            raise FileExistsError("File already exists. Use overwrite==True.")

        header = "# SB SB_err+ SB_err- "
        header += "m m_err+ m_err- "
        header += "M M_err+ M_err- "
        header += "SB_eff SB_eff_err+ SB_eff_err- "
        header += "l l_err "
        header += "w w_err "
        header += "r_eff r_eff_err "
        header += "z mu dL kpcscale\n"

        data = f"{self._surface_brightness} {self._surface_brightness_error[0]} {self._surface_brightness_error[1]} "
        data += f"{self._apparent_magnitude} {self._apparent_magnitude_error[0]} {self._apparent_magnitude_error[1]} "
        data += f"{self._absolute_magnitude} {self._absolute_magnitude_error[0]} {self._absolute_magnitude_error[1]} "
        data += f"{self._effective_surface_brightness} {self._effective_surface_brightness_error[0]} {self._effective_surface_brightness_error[1]} "
        data += f"{self._length} {self._length_error} "
        data += f"{self._width} {self._width_error} "
        data += f"{self._effective_radius} {self._effective_radius_error} "
        data += f"{self._redshift} {self._distance_modulus} {self._luminosity_distance} {self._kpc_scale}"

        file = open(filename, "w")
        file.writelines([header, data])
        file.close()

    def redo_aperture(
        self, criterion: Literal["FWHM, EFF"], k: int = 1, smoothing: int = 10
    ):
        if criterion == "FWHM":
            if k == 1:
                self._aperture = fits.getdata(self.multifits_file, ext=5)
                self._aperture_type = "1FWHM"
            else:
                self._aperture = fwhm_mask_from_paramtab(
                    self.parameter_file, self.multifits_file, k, smoothing
                )
                self._aperture_type = f"{k}FWHM"
            return
        if criterion == "EFF":
            self._aperture = None  # TODO
            self._aperture_type = "EFF"
            return

        print("no valid option for aperture. aperture remains unchanged!")

    def measure(self):
        self._flux_inside_aperture()
        self._calc_bg_from_offsets()
        self._calc_apparent_magnitude()
        self._calc_absolute_magnitude()
        self._calc_surface_brightness()
        self._calc_length()
