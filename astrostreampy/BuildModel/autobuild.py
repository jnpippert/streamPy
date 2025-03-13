import time
import warnings
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import psutil
from astropy.io import fits
from astropy.wcs import WCS

from ..Image.point import Point
from ..utilities import timeit
from . import utilities as util
from .box import Box
from .constants import direction_dict, sectors, slope_dict

warnings.filterwarnings("ignore")


@dataclass
class Monitor:
    # TODO Docstring

    cpu_usage: list = field(default_factory=lambda: [])
    ram_usage: list = field(default_factory=lambda: [])
    time_stamps: list = field(default_factory=lambda: [])

    def monitor(self, time_stamp: float) -> None:
        # TODO Docstring
        self.cpu_usage.append(psutil.cpu_percent())
        self.ram_usage.append(psutil.virtual_memory().percent)
        self.time_stamps.append(time_stamp)

    def show(self) -> None:
        # TODO Docstring
        _, (ax1, ax2) = plt.subplots(2, figsize=(5, 10))
        ax1.plot(self.time_stamps[1:], self.cpu_usage[1:], ".-", color="k")
        ax2.plot(self.time_stamps[1:], self.ram_usage[1:], ".-", color="k")
        np.save("cpu.npy", self.cpu_usage[1:])
        np.save("time.npy", self.time_stamps[1:])
        plt.show()

    def close(self) -> None:
        # TODO Docstring
        plt.close()

    def save(self, out: str) -> None:
        np.save(f"{out}_cpu.npy", self.cpu_usage)
        np.save(f"{out}_ram.npy", self.ram_usage)
        np.save(f"{out}_time.npy", self.time_stamps)


class SegmentModel:
    # TODO Docstring
    def __init__(self, shape: tuple) -> None:
        self.model = np.full(shape=shape, fill_value=np.nan)

    def paste(
        self,
        data: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        offset: float,
        angle: float,
    ):
        # TODO Docstring
        tmp_model = np.full(shape=self.model.shape, fill_value=np.nan)
        angle = abs(np.degrees(np.arctan(np.tan(np.radians(angle)))))
        if 40 <= angle <= 50:
            tmp_model[y - h : y + h + 1, x] = data[:, w + 1]
            tmp_model[y, x - w : x + w + 1] = data[h + 1]
        if angle < 40:
            tmp_model[y, x - w : x + w + 1] = data[h + 1]
        if angle > 50:
            tmp_model[y - h : y + h + 1, x] = data[:, w + 1]

        self.model = np.nanmean([self.model, tmp_model], axis=0)


class ParamTracker:
    """
    Creates a parameter object. Checks for repetition of similiar (~1e-5) values for
    the given parameter. ``_value``, ``_rep_count`` and ``_temp`` are private attributes and
    should not be used outside of this class.

    Attributes
    ----------
    values : list
        List of values added by ``add_val()``.
    """

    def __init__(self, atol: float = None):
        self.values: list = []
        self._value: float = 0
        self._rep_count: int = 1
        self._none_count: int = 1
        self._atol: float = atol
        self._tmp: float = 0
        self._tmptmp: float = None

    def add_val(self, value: Any):
        """
        Adds a value to the listt of previous values and calls the ``_check()`` method.

        Parameters
        ----------
        value : float
            Value to add.

        Returns
        -------
        int
            0 if ``_count()`` else -1
        """
        self._value = value
        self.values.append(value)
        if self._atol is not None and not self._check_close():
            return -1
        if not self._check_none():
            return -1
        self._tmptmp = self._tmp
        self._tmp = value
        return 0

    def _check_none(self):
        self._none_count += 1
        if not all([self._value == None]):
            self._none_count = 1
        return self._none_count != 3

    def _check_close(self):
        self._rep_count += 1
        if not all([np.isclose(self._tmp, self._value, atol=self._atol)]):
            self._rep_count = 1
        return self._rep_count != 3


class Model:
    """
    Creates a model object for a stream

    """

    def __init__(
        self,
        original_data: np.ndarray,
        masked_data: np.ndarray,
        header: fits.Header,
        sourcemask: np.ndarray,
        init_x: int,
        init_y: int,
        init_width: int,
        init_height: int,
        init_angle: float = 0,
        h2: bool = False,
        skew: bool = False,
        h4: bool = False,
        sn_threshold: float = 5.0,
        fix_bg: float = None,
        vary_box_dim: bool = False,
        head: Point = None,
        tail: Point = None,
        tol: int = 10,
        output: str = "streampy",
    ):
        """
        Initializes the object and creates class attributes. ``_ctrlc``,
        ``_header``, ``_filter``, ``_seeing``, ``_vary_box_dim``,
        ``_fix_bg``, ``_dimx``, ``_dimy``, ``_h2``, ``_skew``, ``_h4``,
        ``_tmp_data`` and ``_sn_threshold`` are private attributes
        and should not be used outside this class.

        Parameters
        ----------
        original_data : np.ndarray
            Image array of the original data.

        masked_original_data : np.ndarray
            Image array of the masked original data.

        sourcemask : np.ndarray
            Binary image array of the sourcemask.

        data : np.ndarray
            Image array of the model. When initialized it only contains zeros and gets later filled
            during the ``build()`` method.

        init_x : int
            Central x position of the initial box.

        init_y : int
            Central y position of the initial box.

        init_w : int
            Width of the initial box.

        init_h : int
            Height of the initial box.

        init_angle : float, optional
            Tilt angle of the stream inside the initial box.

        h2 : bool, optional
            Enables h2 fitting. Default is ``False``.

        skew : bool, optional
            Enables skewness fitting. Default is ``False``.

        h4 : bool, optional
            Enables h4 fitting. Default is ``False``.

        sn_threshold : float, optional
            Sets the signal to noise threshold. Default is 5.0.

        fix_bg : float, optional
            Sets the offset/background to a fixed value
            and turns off offset fitting. Default is ``None``.

        vary_box_dim : bool
            Enables dynamic box dimensions. Default is ``False``.

        output : str, optional
            Name(-prefix) of all output files.

        """
        self._ctrlc = False
        self.original_data = original_data
        self.masked_original_data = masked_data
        self.sourcemask = sourcemask

        try:
            self._filter = header["FILTER"]  # Filter band the image was taken in.
        except KeyError as e:
            print(f"warning: set filter to None. {e}")
            self._filter = "None"

        pxscale = header["PXSCALE"]  # Pixelscale of the image in arcseconds/pixel.

        try:
            psf = header["PSF"]  # Mean FWHM in arcseconds of all image PSF's.
            self._seeing = psf / pxscale / (2 * np.sqrt(2 * np.log(2)))
        except KeyError as e:
            psf = None
            self._seeing = None
            print(f"warning: set psf to None. {e}")
            print("info: disabled convolutional fitting")
        print(f"using psf fhwm: {psf} [arcsec]")
        print(f"using pxscale: {pxscale} [arcsec / pixel]")

        self._header = header
        self._vary_box_dim = vary_box_dim
        self._fix_bg = fix_bg

        self._tmp_sigma = None
        self._tmp_peak_pos = None
        # arrays
        self._dimy, self._dimx = original_data.shape
        self.data = np.zeros((self._dimy, self._dimx)) * np.nan
        self._tmp_data = np.zeros((self._dimy, self._dimx)) * np.nan
        self.init_direction = None
        self.post_direction = None
        self._tmp_norm_err = None
        self._tmp_norm = None
        self.box_prop_data = None
        self.param_data = None
        self._init_box_params = None
        self._init_box_param_errs = None
        self._init_box_model = None
        self._init_box_props = None
        # initial parameters
        self.init_x = init_x
        self.init_y = init_y
        self.init_w = init_width
        self.init_h = init_height
        self.init_angle = init_angle
        self.output = output
        self._h2 = h2
        self._skew = skew
        self._h4 = h4
        self._sn_threshold = sn_threshold
        self._init_params = [self.init_angle, 1, 1, 0, 0, 0, 0, 0, 0]
        self._monitor = None
        # like that, higher order fitting is more robust
        if h2:
            self._init_params[6] = 0.01
        if skew:
            self._init_params[7] = 0.01
        if h4:
            self._init_params[8] = 0.01

        # error monitoring
        self.param_errors = []

        # parameters
        self.params = []
        self.box_properties = []

        if any([val == 0 for val in [init_x, init_y]]):
            raise ValueError(
                f"invalid box center positions (expected values > 0, got {init_x,init_y})"
            )

        if any([val == 0 for val in [init_width, init_height]]):
            raise ValueError(
                f"invalid inital box dimensions (expected values > 0, got {init_width}x{init_height})"
            )

        # termination selfs
        self._tol = tol
        self._head = head
        self._head_dist = int(self._tol * 2)
        self._tail = tail
        self._tail_dist = int(self._tol * 2)

    def _check_termination(
        self,
        peak_pos: float,
        norm: float,
        norm_err: float,
        norm_tracker: ParamTracker,
        peak_tracker: ParamTracker,
        box_center: Point,
        bruteforce: bool = False,
    ):
        """
        Checks various termination conditions.
        This is a private method and should not be used outside this class.

        Returns
        -------
        int
            -1 if any condition is met, otherwise 0.
        """

        if peak_tracker.add_val(value=peak_pos) == -1:
            print("continously found no peak. segment terminated.")
            return -1

        if self._head is not None:
            # print(
            #     f"Head distance = {self._tail_dist} ({box_center.x},{box_center.y} # {self._head.x},{self._head.y})"
            # )
            if box_center.isclose(self._head, tol=self._tol):
                self._head_dist -= 1

            if self._head_dist == 0:
                print("head reached. segment terminated.")
                self._head = None
                return -1

        if self._tail is not None:
            if box_center.isclose(self._tail, tol=self._tol):
                self._tail_dist -= 1
            if self._tail_dist == 0:
                print("tail reached. segment terminated.")
                self._tail = None
                return -1

        if bruteforce:
            return 0

        if isinstance(norm_err, float):
            if self._sn_threshold == 0:
                return 0
            if norm / norm_err < self._sn_threshold:
                print("\n")
                print(
                    f"S/N of {norm/norm_err} below threshold of {self._sn_threshold}. segment terminated."
                )
                return -1

        if norm_tracker.add_val(value=norm) == -1:
            print("repeating parameters detected. segment terminated.")
            return -1

        return 0

    def _segment(self, args):
        if self._monitor:
            t0 = time.perf_counter()
            monitor = Monitor()
            monitor.monitor(time.perf_counter() - t0)
        angle, x, y, w, h, direction, step_number, guess_parameter, bruteforce, name = (
            args
        )
        print(name, "started ...")
        init_w = w
        inti_h = h
        best_fit_parameter: list = []
        box_properties: list = []
        seg_model = SegmentModel(shape=(self._dimy, self._dimx))
        norm_tracker = ParamTracker(atol=1e-10)
        peak_tracker = ParamTracker()
        box_center = Point()
        for _ in range(step_number):

            if self._monitor:
                monitor.monitor(time.perf_counter() - t0)
            x, y = util.calculate_next_boxcenter(
                angle=angle,
                x_center=x,
                y_center=y,
                direction=direction,
                dictonary=slope_dict,
            )
            box_center.x = x
            box_center.y = y

            tmp_box = Box(
                self.masked_original_data,
                x,
                y,
                w,
                h,
                init=guess_parameter,
                seeing=self._seeing,
                h2=self._h2,
                skew=self._skew,
                h4=self._h4,
                fix_bg=self._fix_bg,
            )

            if tmp_box.fit_model() == -1:
                break

            if tmp_box.make_model() == -1:
                break
            best_fit_parameter.append([tmp_box.params, tmp_box.param_errs])
            box_properties.append([x, y, w, h])

            seg_model.paste(
                tmp_box.model.copy(), x, y, w, h, tmp_box.offset, tmp_box.angle
            )
            guess_parameter = tmp_box.params
            possible_directions = util.get_direction_options(
                tmp_box.angle, sectors, direction_dict
            )
            direction = util.get_box_direction(direction, possible_directions)
            if (
                self._check_termination(
                    tmp_box.peak_pos,
                    tmp_box.norm,
                    tmp_box.norm_err,
                    norm_tracker,
                    peak_tracker,
                    box_center,
                    bruteforce,
                )
                == -1
            ):
                break

            x, y = util.correct_box_center_from_peak(
                x=x, y=y, w=w, h=h, peak_pos=tmp_box.peak_pos
            )
            if self._vary_box_dim:
                w, h = util.calculate_new_box_dimensions(
                    tmp_box.angle, init_height=inti_h, init_width=init_w
                )

        if self._monitor:
            monitor.show()
            monitor.close()
            monitor.save(out=name)
        return [best_fit_parameter, seg_model.model, box_properties]

    @timeit
    def build(self, steps: tuple[int, int] = (9999, 9999), monitor: bool = False):
        """
        Does the model building.

        First, the initial box is fitted. From there the two possible
        directions in which the box could be shifted is determined.
        From there two loops shift the box and make a model in every step.
        Each box-model is stored. Between the boxes the mean is taken.
        One iteration does the following:
        - correct the old box center by the fit paramters (x0 and/or y0)
        - calculate the new box center
        - calculate new box dimensions
        - create a new box object
        - fit a model
        - get direction options from fitted angle
        - get the true new direction from comparison with current directio and direction options
        - paste the model
        - save the box parameters
        - store neceassary parameters as variables
        - delete the box object

        Parameters
        ----------
        stepsize : int
            Distance the box will get shifted.

        steps : tuple, int
            First value is the number of iterations the box gets shifted towards the 'init_direction'.
            Second value is the number of iterations the box gets shifted towards the 'post_direction'.
        """
        self._monitor = monitor
        init_box = Box(
            self.masked_original_data,
            x=self.init_x,
            y=self.init_y,
            width=self.init_w,
            height=self.init_h,
            seeing=self._seeing,
            init=self._init_params,
            h2=self._h2,
            skew=self._skew,
            h4=self._h4,
            fix_bg=self._fix_bg,
        )

        init_box.fit_model()
        init_box.make_model()
        self._init_box_params = init_box.params
        self._init_box_param_errs = init_box.param_errs
        self._init_box_props = [self.init_x, self.init_y, self.init_w, self.init_h]
        self._init_box_model = SegmentModel(shape=(self._dimy, self._dimx))
        self._init_box_model.paste(
            init_box.model,
            init_box.x,
            init_box.y,
            init_box.width,
            init_box.height,
            init_box.offset,
            init_box.angle,
        )

        dopts = util.get_direction_options(init_box.angle, sectors, direction_dict)

        x, y = util.correct_box_center_from_peak(
            x=self.init_x,
            y=self.init_y,
            w=self.init_w,
            h=self.init_h,
            peak_pos=init_box.peak_pos,
        )
        with Pool() as pool:

            result = pool.map(
                self._segment,
                [
                    (
                        init_box.angle,
                        x,
                        y,
                        self.init_w,
                        self.init_h,
                        dopts[0],
                        steps[0],
                        init_box.params.copy(),
                        self._head is not None,
                        "head_segment",
                    ),
                    (
                        init_box.angle,
                        x,
                        y,
                        self.init_w,
                        self.init_h,
                        dopts[1],
                        steps[1],
                        init_box.params.copy(),
                        self._tail is not None,
                        "tail_segment",
                    ),
                ],
            )

        self._stitch_segment_models(result[0][1], result[1][1])
        self._stitch_best_fit_parameters(
            result[0][0], result[1][0], result[0][2], result[1][2]
        )
        return

    def _stitch_segment_models(self, model1: np.ndarray, model2: np.ndarray):
        self.data = np.nanmean(
            np.array([model1, model2, self._init_box_model.model]), axis=0
        )

    def _stitch_best_fit_parameters(self, params1, params2, props1, props2):
        params1 = np.flip(np.array(params1), axis=0)
        params2 = np.array(params2)
        props1 = np.flip(np.array(props1), axis=0)
        props2 = np.array(props2)

        self.box_prop_data = np.array([*props1, self._init_box_props, *props2])
        params = np.array([*params1[:, 0], self._init_box_params, *params2[:, 0]])
        param_errs = np.array(
            [*params1[:, 1], self._init_box_param_errs, *params2[:, 1]]
        )
        self.param_data = np.array([params, param_errs])

        self._create_parameter_table()
        self._save()

    def show(self, output: str = None):
        """
        Displays a plot to the screen with the original, model and residual image. Mainly used to check how the
        ``build()`` method performed.

        Parameters
        ----------
        output : str, optional
            If not ``None``(default) the figure is saved as PNG file.

        """
        array = self.masked_original_data.copy()
        array = np.nan_to_num(array, nan=0)
        wcs = WCS(self._header)
        _, (ax1, ax2, ax3) = plt.subplots(
            1,
            3,
            sharex=True,
            sharey=True,
            figsize=(12, 4),
            subplot_kw={"projection": wcs},
        )

        vmin = np.percentile(array, 2)
        vmax = np.percentile(array, 98)

        # image
        ax1.imshow(array, vmin=vmin, vmax=vmax, cmap="YlOrBr", origin="lower")
        ax1.set_xlabel("R.A.")
        ax1.set_ylabel("Dec.")
        ax1.set_title("image")

        # model
        model = np.nan_to_num(self.data, nan=0)
        ax2.imshow(model, vmin=vmin, vmax=vmax, cmap="YlOrBr", origin="lower")
        ax2.set_xlabel("R.A.")
        ax2.set_ylabel("Dec.")
        ax2.set_title("model")

        # residual
        ax3.imshow(
            self.masked_original_data - model,
            vmin=vmin,
            vmax=vmax,
            cmap="YlOrBr",
            origin="lower",
        )
        ax3.set_xlabel("R.A.")
        ax3.set_ylabel("Dec.")
        ax3.set_title("residual")

        if isinstance(output, str):
            plt.savefig(f"{output}" + "_plot.png", dpi=300)
        plt.show()

    def _save(self):
        """
        Saves all data arrays, i.e. masks, images,models in one FITS file. This is a private
        method and should not be used outside this class.
        """
        output = self.output + "_multifits.fits"

        self._header["OBJECT"] = "stream"
        primary_hdu = fits.PrimaryHDU(self.original_data, header=self._header)
        data = np.nan_to_num(self.data.copy(), nan=0)
        self._header["OBJECT"] = "masked intpol stream"
        mrdata_hdu = fits.ImageHDU(self.masked_original_data, header=self._header)
        self._header["OBJECT"] = "sourcemask"
        mask_hdu = fits.ImageHDU(self.sourcemask, header=self._header)
        self._header["OBJECT"] = "model"
        model_hdu = fits.ImageHDU(data, header=self._header)
        self._header["OBJECT"] = "residual"
        residual_hdu = fits.ImageHDU(self.original_data - data, header=self._header)
        hdul = fits.HDUList(
            [primary_hdu, mask_hdu, mrdata_hdu, residual_hdu, model_hdu]
        )

        hdul.writeto(fileobj=output, overwrite=True)

    def _create_parameter_table(self):
        """
        Creates a data table of all fit parameters and their errors.

        Parameter arrays consists of 1d-arrays. Within each array the paramters are stored and resemble
        the fit parameters of one box. This is a private method and should not be used outside this class.

        Order of parameters:
        x,y,w,h,angle,angle_err,sigma,sigma_err,norm,norm_err,offset,offset_err,
        x0, x0_err, y0, y0_err, h2, h2_err, skew, skew_err, h4, h4_err
        """
        output = self.output + "_paramtab.fits"
        pd = self.param_data
        bp = self.box_prop_data

        # x
        bx = bp[:, 0]
        bxc = fits.Column(name="box_x", format="I", array=bx)

        # y
        by = bp[:, 1]
        byc = fits.Column(name="box_y", format="I", array=by)

        # width
        bw = bp[:, 2]
        bwc = fits.Column(name="box_w", format="I", array=bw)

        # height
        bh = bp[:, 3]
        bhc = fits.Column(name="box_h", format="I", array=bh)

        # angle
        angle, angle_err = pd[:, :, 0]
        ac = fits.Column(name=f"angle_{self._filter}", format="E", array=angle)
        aec = fits.Column(name=f"angle_{self._filter}_err", format="E", array=angle_err)

        # sigma / standard deviation
        sigma, sigma_err = pd[:, :, 1]
        sc = fits.Column(name=f"sigma_{self._filter}", format="E", array=sigma)
        sec = fits.Column(name=f"sigma_{self._filter}_err", format="E", array=sigma_err)

        # normalization / amplitude
        norm, norm_err = pd[:, :, 2]
        nc = fits.Column(name=f"norm_{self._filter}", format="E", array=norm)
        nec = fits.Column(name=f"norm_{self._filter}_err", format="E", array=norm_err)

        # offset / background
        offset, offset_err = pd[:, :, 3]
        oc = fits.Column(name=f"offset_{self._filter}", format="E", array=offset)
        oec = fits.Column(
            name=f"offset_{self._filter}_err", format="E", array=offset_err
        )

        # x0 / x shift
        x0, x0_err = pd[:, :, 4]
        xc = fits.Column(name=f"x0_{self._filter}", format="E", array=x0)
        xec = fits.Column(name=f"x0_{self._filter}_err", format="E", array=x0_err)

        # y0 / y shift
        y0, y0_err = pd[:, :, 5]
        yc = fits.Column(name=f"y0_{self._filter}", format="E", array=y0)
        yec = fits.Column(name=f"y0_{self._filter}_err", format="E", array=y0_err)

        # h3 / third gaussian hermite
        h2, h2_err = pd[:, :, 6]
        h2c = fits.Column(name=f"h2_{self._filter}", format="E", array=h2)
        h2ec = fits.Column(name=f"h2_{self._filter}_err", format="E", array=h2_err)

        # h4 / fourth gaussian hermite
        skew, skew_err = pd[:, :, 7]
        skewc = fits.Column(name=f"skew_{self._filter}", format="E", array=skew)
        skewec = fits.Column(
            name=f"skew_{self._filter}_err", format="E", array=skew_err
        )

        h4, h4_err = pd[:, :, 8]
        h4c = fits.Column(name=f"h4_{self._filter}", format="E", array=h4)
        h4ec = fits.Column(name=f"h4_{self._filter}_err", format="E", array=h4_err)

        hdu = fits.BinTableHDU.from_columns(
            [
                bxc,
                byc,
                bwc,
                bhc,
                ac,
                aec,
                sc,
                sec,
                nc,
                nec,
                oc,
                oec,
                xc,
                xec,
                yc,
                yec,
                h2c,
                h2ec,
                skewc,
                skewec,
                h4c,
                h4ec,
            ]
        )
        hdu.writeto(name=output, overwrite=True)
