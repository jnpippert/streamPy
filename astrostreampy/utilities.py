import numpy as np
from astropy.convolution import Gaussian1DKernel, convolve
from scipy.stats import norm as scn

from .BuildModel.constants import c2_1, c2_2, c4_1, c4_2, c4_3
from .BuildModel.utilities import create_grid_arrays


def fit_func_2D(
    _: None,
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
        norm
        * np.exp(-0.5 * vals**2)
        * (1 + h2_comp + h4_comp + scn.cdf(skewv * vals))
    ) + offset
    if seeing is None:
        return model
    return convolve(model, kernel=kernel, boundary="extend", nan_treatment="fill")


def fit_func_1D(x, sigma, norm, offset, h2=0, skew=0, h4=0, seeing: float = None):
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
