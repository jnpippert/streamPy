"""
Provides several useful methods for pixel math and analysis of 1D Gaussians.

Example
-------
>>> from utilities import effective_width
>>> from utilities import fwhm_width
"""

import numpy as np
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from scipy.signal import find_peaks


def interpolate_zero_pixels(
    data: np.ndarray, x_stddev: int = 10, y_stddev: int = 7, theta: float = 0
) -> np.ndarray:
    """
    Interpolates every pixel with value zero.

    Parameters
    ----------
    data : `np.ndarray`
        The image array.

    x_stddev : float
        Standard deviation of the Gaussian in x before rotating by theta.

    y_stddev : float
        Standard deviation of the Gaussian in y before rotating by theta.

    theta : float
        Rotation angle in degree.

    Returns
    -------
    interpolated array
    """

    kernel = Gaussian2DKernel(
        x_stddev=x_stddev, y_stddev=y_stddev, theta=np.radians(theta - 90)
    )
    data[data == 0] = np.nan
    return interpolate_replace_nans(data, kernel)


def check_peaks(gauss: np.ndarray):
    """
    Check for the number and similarity of peaks in a Gaussian array.

    Parameters
    ----------
    gauss : np.ndarray
        The Gaussian array.
    """
    peaks = find_peaks(gauss)[0]
    if len(peaks) == 2:
        if np.isclose(gauss[peaks[0]], gauss[peaks[1]]):
            print("[WARNING] Two peaks with similar height detected!")
            print("          Measurements for 'r' and 'w' might be incorrect.")


def effective_width(gauss: np.ndarray) -> float:
    """
    Calculate the effective width of a Gaussian array.

    Parameters
    ----------
    gauss : np.ndarray
        The Gaussian array.

    Returns
    -------
    float
        The effective width.

    Notes
    -----
    The effective width is defined as the width of the region containing
    half of the total flux of the Gaussian, centered at the peak.
    """
    check_peaks(gauss)
    dx = 0
    total_flux = np.sum(gauss)
    center_pix = np.argmax(gauss)
    while True:
        region = gauss[center_pix - dx : center_pix + dx + 1]
        if np.sum(region) > total_flux / 2:
            break

        dx += 1

    return dx + 0.5


def fwhm_width(gauss: np.ndarray) -> int:
    """
    Calculate the full width at half maximum (FWHM) of a Gaussian.

    Parameters
    ----------
    gauss : np.ndarray
        The Gaussian array.

    Returns
    -------
    fwhm_width : int
        The FWHM width.
    """
    check_peaks(gauss)
    dl = 0
    dr = 0
    left = right = False
    half = np.max(gauss) / 2
    center_pix = np.argmax(gauss)
    while True:
        if gauss[center_pix - dl] >= half:
            dl += 1
        else:
            left = True
        if gauss[center_pix + dr] >= half:
            dr += 1
        else:
            right = True
        if left and right:
            break

    return dl + dr + 1 - 2
