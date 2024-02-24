"""
Provides the Gaussian fit functions for 1D and 2D (grid).
Provides a usefule timing decorator.

Example
-------
>>> from utilities import fit_func_1D
>>> from utilities import fit_func_2D
>>> from utilities import timeit
"""

import time
from functools import wraps

import numpy as np
from astropy.convolution import Gaussian1DKernel, convolve
from scipy.stats import norm as scn

from .BuildModel.constants import c2_1, c2_2, c4_1, c4_2, c4_3
from .BuildModel.utilities import create_grid_arrays


def fit_func_2D(
    angle: float,
    sigma: float,
    norm: float,
    offset: float,
    x0: float,
    y0: float,
    h2v: float,
    skewv: float,
    h4v: float,
    w: int,
    h: int,
    seeing: float = None,
) -> np.ndarray:
    """
    The 2D fit function to fit a Gaussian with higher order moments
    to a 2D grid with a width and height.

    Parameters
    ----------
    angle : float
        Rotation angle of Grid, increasing counter clockwise.
    sigma : float
        Standard deviation of the Gaussian.
    norm : float
        Amplitude of the Gaussian.
    offset : float
        Offset of the Gaussian.
    x0 : float
        X-offset of the Gaussian center.
    y0 : float
        Y-offset of the Gaussian center.
    h2v : float
        Coefficient for the second-order moment term.
    skewv : float
        Skewness of the distribution.
    h4v : float
        Coefficient for the fourth-order moment term.
    w : int
        Width of the 2D grid.
    h : int
        Height of the 2D grid.
    seeing : float, optional
        Standard deviation of the convolution kernel for seeing effects.
        Can also be understood as the standard deviation of the convolution
        kernel (default is None).

    Returns
    -------
    model : ndarray
        The 2D Gaussian model.

    Notes
    -----
    If `seeing` is provided, the function convolves the model with a Gaussian kernel.
    """

    if seeing is not None:
        kernel = Gaussian1DKernel(stddev=seeing)

    y_grid, x_grid = create_grid_arrays(width=w, height=h)
    grid = np.sin(np.radians(angle)) * (y_grid - y0) + np.cos(np.radians(angle)) * (
        x_grid - x0
    )

    ravel_grid = np.ravel(grid)

    vals = ravel_grid / sigma
    h4_comp = h4v * (c4_1 * vals**4 - c4_2 * vals**2 + c4_3)
    h2_comp = h2v * (c2_1 * vals**2 - c2_2)
    model = (
        norm * np.exp(-0.5 * vals**2) * (1 + h2_comp + h4_comp + scn.cdf(skewv * vals))
    ) + offset
    if seeing is None:
        return model
    return convolve(model, kernel=kernel, boundary="extend", nan_treatment="fill")


def fit_func_1D(x, sigma, norm, offset, h2=0, skew=0, h4=0, seeing: float = None):
    """
    The 1D fit function for a Gaussian with higher order moments.

    Parameters
    ----------
    x : array_like
        Array of independent variable values.
    sigma : float
        Standard deviation of the Gaussian.
    norm : float
        Amplitude of the Gaussian.
    offset : float
        Offset of the Gaussian.
    h2 : float, optional
        Coefficient for the second-order moment term (default is 0).
    skew : float, optional
        Skewness of the distribution (default is 0).
    h4 : float, optional
        Coefficient for the fourth-order moment term (default is 0).
    seeing : float, optional
        Standard deviation of the convolution kernel for seeing effects.
        Can also be understood as the standard deviation of the convolution
        kernel (default is None).

    Returns
    -------
    model : ndarray
        The 1D Gaussian model.

    Notes
    -----
    If `seeing` is provided, the function convolves the model with a Gaussian kernel.
    """
    if seeing is not None:
        kernel = Gaussian1DKernel(stddev=seeing)

    vals = x / sigma
    h4_comp = h4 * (c4_1 * vals**4 - c4_2 * vals**2 + c4_3)
    h2_comp = h2 * (c2_1 * vals**2 - c2_2)
    model = (
        norm * np.exp(-0.5 * vals**2) * (1 + h2_comp + h4_comp + scn.cdf(skew * vals))
    ) + offset
    if seeing is None:
        return model
    return convolve(model, kernel=kernel, boundary="extend", nan_treatment="fill")


def timeit(
    func=None,
    ## /,
    ## *,
    ## print_input: bool = False,
    ## print_output: bool = False,
    ## use_print: bool = False,
):
    """
    A decorator to time the execution of a function, written by
    Raphael Spiekermann (https://github.com/raphaelspiekermann).
    Some parts are modified or removed, which are marked as double comments.
    """

    def decorator_timeit(func):
        @wraps(func)
        def wrapper_timeit(*args, **kwargs):
            print_fn = print  ## print_fn = print if use_print else logging.debug
            start_time = time.perf_counter()  ## time.time()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()  ## time.time()
            ## if print_input:
            ##    print_fn(f"@Timeit({func.__name__}) -> Input: {args}, {kwargs}")
            ## if print_output:
            ##    print_fn(f"@Timeit({func.__name__}) -> Output: {result}")
            ## print_fn(
            ##    f"@Timeit({func.__name__}) -> Execution time: {end_time - start_time} seconds"
            ## )
            print_fn(
                f"\tfinished after {np.round(end_time - start_time,2)} seconds"
            )  ## added
            return result

        return wrapper_timeit

    if func is None:  # @timeit() -> @timeit
        return decorator_timeit

    return decorator_timeit(func)
