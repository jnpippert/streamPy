import numpy as np
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans


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
