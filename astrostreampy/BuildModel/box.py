from datetime import datetime

import numpy as np
from astropy.convolution import Gaussian1DKernel, convolve
from lmfit import Model
from scipy.stats import norm as scn

from .constants import c2_1, c2_2, c4_1, c4_2, c4_3
from .utilities import calc_fwhm_pos, create_grid_arrays


class Box:
    """
    Creates a box object, which represents a small segemnt of the stream.
    """

    def __init__(
        self,
        original_data: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        seeing: float,
        init: list = None,
        h2: bool = False,
        skew: bool = False,
        h4: bool = False,
        fix_bg: float = None,
    ):
        """
        Initializes the box object.

        Parameters
        ----------
        rdata : np.ndarray
            Image array of the original/real data.

        x : int
            Central x position of the box.

        y : int
            Central y position of the box.

        width : int
            Width of the box.

        height : int
            Height of the box.

        seeing : float
            Seeing quality of the image/box.

        init : list, optional
            List of initial guess parameters. Order of parameters are:
            [angle, sigma, norm, offset, x0, y0, h2, skew, h4]. Default is [0,1,1,0,0,0,0,0,0].

        h2 : bool, optional
            Enables h2 fitting. Default is ``False``.

        skew : bool, optional
            Enables skewness fitting. Default is ``False``.

        h4 : bool, optional
            Enables h4 fitting. Default is ``False``.

        fix_bg : float, optional
            Sets the offset/background to a fixed value and turns off offset fitting. Default is ``None``.
        """

        if init is None:
            init = [0, 1, 1, 0, 0, 0, 0, 0, 0]

        self._psf = Gaussian1DKernel(stddev=seeing)
        data = original_data.copy()[y - height : y + height, x - width : x + width]
        data[data == 0] = np.nan
        self.data = data
        self._y_grid, self._x_grid = create_grid_arrays(width, height)
        self.height = height
        self.width = width
        self.params = []
        self.param_errs = []
        self._x = x
        self._y = y

        # initalize model
        iangle, isigma, inorm, ioffset, ix0, iy0, ih2, iskew, ih4 = init
        func_model = Model(self._fitfunc)

        func_params = func_model.make_params(
            angle=iangle,
            sigma=isigma,
            norm=inorm,
            offset=ioffset,
            x0=ix0,
            y0=iy0,
            h2v=ih2,
            skewv=iskew,
            h4v=ih4,
        )

        # set bounds
        func_params["angle"].min = -360
        func_params["angle"].max = 360
        func_params["x0"].min = -width
        func_params["x0"].max = width
        func_params["y0"].min = -height
        func_params["y0"].max = height

        # turn on/off x0 and y0 fitting
        if iangle <= 40:
            func_params["y0"].vary = False
            func_params["y0"].value = 0
        if iangle >= 50:
            func_params["x0"].vary = False
            func_params["x0"].value = 0

        # turn on/off h2, h3, h4
        if not h2:
            func_params["h2v"].vary = False
            func_params["h2v"].value = 0

        if not skew:
            func_params["skewv"].vary = False
            func_params["skewv"].value = 0

        if not h4:
            func_params["h4v"].vary = False
            func_params["h4v"].value = 0

        # fix offset value
        if isinstance(fix_bg, int):
            fix_bg = float(fix_bg)
        if isinstance(fix_bg, float):
            func_params["offset"].vary = False
            func_params["offset"].value = fix_bg

        self._func_params = func_params
        self._func_model = func_model
        self._nan_policy = True

        self.angle = None
        self.sigma = None
        self.norm = None
        self.offset = None
        self.x0 = None
        self.y0 = None
        self.h2v = None
        self.skewv = None
        self.h4v = None
        self.model = None
        self.norm_err = None
        self.peak_pos = None

    def _fitfunc(
        self,
        xy: None = None,
        angle: float = 0,
        sigma: float = 1,
        norm: float = 1,
        offset: float = 0,
        x0: float = 0,
        y0: float = 0,
        h2v: float = 0,
        skewv: float = 0,
        h4v: float = 0,
    ) -> np.ndarray:
        """
        The fit function parsed into ``lmfit.Model``. All parameters are varied by LMfit-py.
        Every call a new set of parameters is used to create a model image with the size of the box.
        Before returning the model array it gets convovled by the seeing, this ensure that the intrinsic
        Gaussian parameters are fitted. This is a private method and should not be used outside this class.

        """
        grid = np.sin(np.radians(angle)) * (self._y_grid - y0) + np.cos(
            np.radians(angle)
        ) * (self._x_grid - x0)

        if self._nan_policy:
            ravel_grid = np.ravel(grid[~np.isnan(self.data)])
        else:
            ravel_grid = np.ravel(grid)

        vals = ravel_grid / sigma
        h4_comp = h4v * (c4_1 * vals**4 - c4_2 * vals**2 + c4_3)
        h2_comp = h2v * (c2_1 * vals**2 - c2_2)
        model = (
            norm
            * np.exp(-0.5 * vals**2)
            * (1 + h2_comp + h4_comp + scn.cdf(skewv * vals))
        ) + offset
        if self._nan_policy:
            return convolve(
                model, kernel=self._psf, boundary="extend", nan_treatment="fill"
            )
        return convolve(
            model, kernel=self._psf, boundary="extend", nan_treatment="fill"
        )

    def fit_model(self):
        """
        Starts the parameter fitting.
        """
        self._nan_policy = True
        ravel_data = np.ravel(self.data[~np.isnan(self.data)])

        try:
            result = self._func_model.fit(
                ravel_data, self._func_params, xy=None
            )  # xy is just a place holder
        except ValueError:
            return -1

        for key in result.params:
            self.params.append(result.params[key].value)
            self.param_errs.append(result.params[key].stderr)

        (
            self.angle,
            self.sigma,
            self.norm,
            self.offset,
            self.x0,
            self.y0,
            self.h2v,
            self.skewv,
            self.h4v,
            *_,
        ) = self.params
        self.norm_err = self.param_errs[2]

    def make_model(self):
        """
        Uses ``_func()`` do create the box model based on the best fit parameters found.
        """
        self._nan_policy = False

        model = self._fitfunc(None, *self.params)
        self.model = model.reshape(self.data.shape)

        if self.width > self.height:
            data_slice = self.model[self.height]
        else:
            data_slice = self.model[:, self.width]

        res = calc_fwhm_pos(array=data_slice)
        if res == -1:
            self.peak_pos = None
        else:
            self.peak_pos = res[4]


class BoxList:
    """
    Class to track and save the model parameters of every box inside a TXT file.
    """

    def __init__(self, filename: str = None):
        """
        Initializes the TXT file.

        Parameters
        ----------
        filename : str, optional
            If ``None``(default) the filename is automatically set.
        """
        time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        if filename is None:
            filename = f"streampy_{time_stamp}.boxlist"
        else:
            filename = str(filename) + ".boxlist"

        self.filename = filename
        try:
            file = open(filename, "x", encoding="utf-8")
        except FileExistsError:
            file = open(filename, "a", encoding="utf-8")
            file.truncate(0)

        file.write(
            "#id xpos ypos width height angle sigma norm offset x0 y0 h2 skew h4\n"
        )
        file.close()

    def write_line(self, data: list = None, comment=None):
        """
        Writes a line of parameters to the file.

        Parameters
        ----------
        data : list, optional
            List of the box parameters.

        comment : str, optional
            If not ``None`` (default), only a comment line is written to the file.
        """
        if isinstance(comment, str):
            line = f"# {comment}"
        else:
            line = " ".join(np.array(data).astype(str))
        with open(self.filename, "a", encoding="utf-8") as file:
            file.write(line + "\n")
            file.close()
