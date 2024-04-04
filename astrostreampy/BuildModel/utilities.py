import numpy as np
from astropy.convolution import Gaussian1DKernel, convolve
from astropy.io import fits
from scipy.signal import find_peaks
from scipy.stats import norm as scn

from .constants import *


def intro():
    """
    Prints the intro of the stream modelling.
    """
    print(
        "  _____ _______ _____  ______          __  __                       _      _ _           "
        + "\\u00A9"
    )
    print(
        " / ____|__   __|  __ \\|  ____|   /\\   |  \\/  |                     | |    | | |          "
    )
    print(
        "| (___    | |  | |__) | |__     /  \\  | \\  / |  _ __ ___   ___   __| | ___| | | ___ _ __ "
    )
    print(
        " \\___ \\   | |  |  _  /|  __|   / /\\ \\ | |\\/| | | '_ ` _ \\ / _ \\ / _` |/ _ \\ | |/ _ \\ '__|"
    )
    print(
        " ____) |  | |  | | \\ \\| |____ / ____ \\| |  | |_| | | | | | (_) | (_| |  __/ | |  __/ |   "
    )
    print(
        "|_____/   |_|  |_|  \\_\\______/_/    \\_\\_|  |_(_)_| |_| |_|\\___/ \\__,_|\\___|_|_|\\___|_|   "
    )
    print(
        "                             developed by Jan-Niklas Pippert                             "
    )


def create_grid_arrays(width: int, height: int) -> np.ndarray:
    """
    Returns an open multi-dimensional "meshgrid". See numpy.ogrid for
    further information.
    """
    return np.ogrid[0 - height : height + 1, 0 - width : width + 1]


def get_direction_options(angle: float, sectors: list, dictonary: dict) -> tuple:
    """
    Determines the possible directions the nex box might get shifted to.
    """
    angle = np.arctan(np.tan(np.radians(angle))) * 180 / np.pi

    for i, val in enumerate(sectors[0]):
        if val[0] < angle < val[1]:
            return dictonary[i]

    return dictonary[[i + 4 for i, val in enumerate(sectors[1]) if angle == val][0]]


def get_box_direction(current_direction: str, directions: tuple) -> str:
    """
    Returns the correct direction keyword.
    """
    for direction in directions:
        if current_direction[0] == direction[0] or current_direction[1] == direction[0]:
            return direction

    raise ValueError(f"no match for '{current_direction}' found in {directions}")


def calculate_next_boxcenter(
    angle: float,
    x_center: int,
    y_center: int,
    direction: str,
    dictonary: dict,
) -> tuple[int, int]:
    """
    Calculates the next box center.
    """
    slope = np.tan(np.arctan(np.tan(np.radians(angle)))).astype(np.float32)
    y_solution = abs(1 / np.sqrt(slope**2 + 1))
    x_solution = abs(slope * y_solution)

    if int(abs(slope)) == 0:
        x_shift, y_shift = dictonary[direction[0] + "="]

    if slope < 0:
        x_shift, y_shift = dictonary[direction[0] + "<"]

    if slope > 0:
        x_shift, y_shift = dictonary[direction[0] + ">"]

    if np.arctan(np.tan(np.radians(angle))) * 180 / np.pi in [90, 270]:
        x_shift, y_shift = dictonary[direction[0] + "="]

    x_shift = x_shift * x_solution
    y_shift = y_shift * y_solution

    return (int(np.round(x_center + x_shift, 0)), int(np.round(y_center + y_shift, 0)))


def width_function(angle: float, scale_param: float, offset: int) -> float:
    """
    Calculates the function value of the width function. See the 'calculate_new_box_dimensions'
    method for further information on the graph and its parameters.

    Parameters
    ----------
    angle : float
        Angle in degree or radians. If angle in radians, type must
        be parsed with value 'radians'.

    scale_param : float
        Scale value which scales the function into the right shape.
        This value is calculated in the 'calculate_new_box_dimensions' method.

    offset : int
        The positive shift on the y-axis.
        This value is calculated in the 'calculate_new_box_dimensions' method.

    Returns
    -------
    new width : float
    """

    if not offset > 0:
        raise ValueError("'offset' out of range (offset > 0)")

    if not 0 <= angle <= 90:
        raise ValueError("'angle' out of range (0 <= 90 value <= 90)")

    return (
        (-scale_param / np.arctan(45) * np.arctan((angle - 45))) + scale_param
    ) + offset


def height_function(angle: float, scale_param: float, offset: int) -> float:
    """
    Calculates the function value of the height function. See the ``calculate_new_box_dimensions``
    method for further information on the graph and its parameters.

    Parameters
    ----------
    angle : float
        Angle in degree or radians. If angle in radians, type must
        be parsed with value 'radians'.

    scale_param : float
        Scale value which scales the function into the right shape.
        This value is calculated in the 'calculate_new_box_dimensions' method.

    offset : int
        The positive shift on the y-axis.
        This value is calculated in the 'calculate_new_box_dimensions' method.

    Returns
    -------
    new height : float
    """
    if not offset > 0:
        raise ValueError("'offset' out of range (offset > 0)")

    if not 0 <= angle <= 90:
        raise ValueError("'angle' out of range (0 <= 90 angle <= 90)")

    return (
        (scale_param / np.arctan(45) * np.arctan((angle - 45))) + scale_param
    ) + offset


def damping_function(angle: float, sigma: float = 5.0, amplitude: float = 0.8) -> float:
    """
    Calculates a damping factor to downscale the box dimensions near an angle of 45, including it's
    multiples, degrees.

    Parameters
    ----------
    angle : float
        Angle in degree or radians. If angle in radians, type must
        be parsed with value 'radians'.

    sigma : float
        The sigma value of the Gaussian. Default is 5.0.

    amplitude : float
        A stretch factor for the gaussian. Default is 0.8.

    Returns
    -------
    damping value : float
    """
    if not 0 <= angle <= 90:
        raise ValueError("'angle' out of range (0 <= 90 value <= 90)")

    return -amplitude * np.exp(-1 / 2 * (angle - 45) ** 2 / sigma**2) + 1


def calculate_new_box_dimensions(
    angle: float, init_width: int, init_height: int, angle_type: str = "degree"
) -> list:
    """
    Calculates the new box dimensions based on the fitted model angle
    in the current box. 'angle' is transformed such that it's in degree,
    greater than zero and does not contain multiples of pi, i.d. less than
    90 degree. From the inital box dimensions a scaling value and a offset
    are computed which bring the width and height function to the right shape.

    Both are arctan functions. The damping function, which down scales the
    box dimensions near 45 degree, is a Gauss function. It makes sure that
    a stream with a strong curvature could still possibly be modelled.

    Use the following link to have a visualization of the the three functions:
    https://www.desmos.com/calculator/anbl4lrxlj

    Parameters
    ----------
    angle : float
        Angle in degree or radians. If angle in radians, type must
        be parsed with value 'radians'.

    init_height : int
        Height of the initial box.

    init_width : int
        Width of the intital box.

    type : str
        The unit-type of the angle. Default is degree.
        Other valid option is 'radians'.

    Returns
    -------
    dimension list
        List of integers containing the new width and height.
    """

    if angle_type == "radians":
        angle = np.degrees(angle)

    # eliminate multiples of pi and transforming negative values
    angle = abs(np.degrees(np.arctan(np.tan(np.radians(angle)))))

    if not any([opt == type for opt in ["degree", "radians"]]):
        raise ValueError(
            f"invalid option for 'type' keyword (expected 'degree' or 'radians', given {type})"
        )

    scale_param = abs((init_width - init_height) / 2)

    if init_width > init_height:
        offset = init_height
    else:
        offset = init_width

    damp_val = 1  # damping_function(angle) currently deactivated, TODO overwork the dynamic box dimensions.
    width = width_function(angle, scale_param=scale_param, offset=offset)
    height = height_function(angle, scale_param=scale_param, offset=offset)
    new_width = int(np.round(width * damp_val, 0))
    new_height = int(np.round(height * damp_val, 0))
    return new_width, new_height


def half_pos(array, val):
    """
    Returns the half maximum position of a 1d array (distirbution).
    """
    return np.where(np.min(abs(array - val / 2)) == abs(array - val / 2))


def calc_fwhm_pos(array):
    """
    Calculates the positions, array IDs of the points where the values are closest to the
    half maximum.
    """
    # TODO refactor the whole method
    # TODO return peak pos right -> compare height of all found peaks, use the highest
    peaks, _ = find_peaks(
        x=array, prominence=np.max(array) / 5
    )  # TODO prominence might not be the best solution
    size = int(array.size / 2)
    xarr = np.linspace(-size, size, 2 * size + 1)

    if len(peaks) == 0:
        return -1
    left_peak = peaks[0]
    left_id = half_pos(array[:left_peak], val=array[left_peak])[0][0]
    right_id = half_pos(array[left_peak:], val=array[left_peak])[0][0] + left_peak
    if len(peaks) > 1:
        right_peak = peaks[-1]
        right_id = (
            half_pos(array[right_peak:], val=array[right_peak])[0][0] + right_peak
        )
        if array[left_peak] > array[right_peak]:
            return (
                left_peak,
                left_id,
                right_peak,
                right_id,
                xarr[left_peak],
            )  # true peak here
        else:
            return left_peak, left_id, right_peak, right_id, xarr[right_peak]
    return left_peak, left_id, left_peak, right_id, xarr[left_peak]


def correct_box_center_from_peak(
    x: int, y: int, w: int, h: int, peak_pos: float
) -> list:
    """
    Corrects the box center from an peak to center box offset.
    """
    if peak_pos is None:
        print("[WARNING] No peak found, box center correction skipped")
        return x, y
    if w > h:
        return x + peak_pos, y
    return x, y + peak_pos


def get_distances(parameter_file, multifits_file):
    """
    Calculates the distance from the peak to the point where the half maximum
    is reached. It does this independently for the left and right side.

    Parameters
    ----------
    param_file : str
        A ``_paramtab.fits`` file created by the ``streampy.BuildModel.autobuild.build`` method.

    multifits_file : str
        A ``_multifits.fits`` file created by the ``streampy.BuildModel.autobuild.build`` method.

    Returns
    -------
    result : list
    """
    left_dists: list = []
    right_dists: list = []
    center_ids: list = []
    table = fits.getdata(parameter_file, ext=1)
    model = fits.getdata(multifits_file, ext=4)

    for row in table:
        (
            x,
            y,
            w,
            h,
        ) = np.array(row)[
            np.array([0, 1, 2, 3])
        ].astype(int)

        if w > h:
            model_slice = model.copy()[y, x - w : x + w + 1]
        else:
            model_slice = model.copy()[y - h : y + h + 1, x]

        center_id = np.argmax(model_slice)
        model_x = np.arange(0, len(model_slice), 1)
        left_ids = model_x < center_id
        right_ids = model_x > center_id

        half_max = np.max(model_slice) / 2
        left = np.argmin(np.abs(model_slice[left_ids] - half_max))
        right = np.argmin(np.abs(model_slice[right_ids] - half_max)) + center_id + 1

        left_dists.append(center_id - left)
        right_dists.append(right - center_id)
        center_ids.append(center_id)
    return center_ids, left_dists, right_dists


def get_effective_distances(parameter_file, multifits_file):
    dists: list = []
    center_ids: list = []
    table = fits.getdata(parameter_file, ext=1)
    model = fits.getdata(multifits_file, ext=4)

    for row in table:
        (
            x,
            y,
            w,
            h,
        ) = np.array(row)[
            np.array([0, 1, 2, 3])
        ].astype(int)

        if w > h:
            model_slice = model.copy()[y, x - w : x + w + 1]
        else:
            model_slice = model.copy()[y - h : y + h + 1, x]
        total_flux = np.sum(model_slice)
        center_id = np.argmax(model_slice)
        dist = 0

        while True:
            region = model_slice[center_id - dist : center_id + dist + 1]

            if np.sum(region) > total_flux / 2:

                break
            dist += 1

        dists.append(dist)
        center_ids.append(center_id)

    return center_ids, dists


def fit_func(
    size: int,
    sigma: float,
    norm: float,
    offset: float,
    h2: float,
    skew: float,
    h4: float,
    psf: float = None,
) -> np.ndarray:
    """
    The fit function parsed into ``lmfit.Model``. All parameters are varied by LMfit-py.
    Every call a new set of parameters is used to create a model image with the size of the box.
    Before returning the model array it gets convovled by the seeing, this ensure that the intrinsic
    Gaussian parameters are fitted. This is a private method and should not be used outside this class.

    """

    vals = np.arange(-size, size + 1, 1) / sigma
    h4_comp = h4 * (c4_1 * vals**4 - c4_2 * vals**2 + c4_3)
    h2_comp = h2 * (c2_1 * vals**2 - c2_2)
    model = (
        norm * np.exp(-0.5 * vals**2) * (1 + h2_comp + h4_comp + scn.cdf(skew * vals))
    ) + offset

    if isinstance(psf, float):
        return convolve(
            model,
            kernel=Gaussian1DKernel(stddev=psf),
            boundary="extend",
            nan_treatment="fill",
        )
    return model
