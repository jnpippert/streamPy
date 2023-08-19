import signal

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from . import utilities as util
from .box import Box, BoxList
from .constants import direction_dict, sectors, slope_dict
from .liveplot import LivePlot


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

    values: list = []
    _value: float = 0
    _rep_count: int = 1
    _tmp: float = 0

    def add_val(self, value: float):
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
        self._tmp = self._value
        self._value = value
        self.values.append(value)
        if not self._check():
            return -1
        return 0

    def _check(self):
        """
        Private method that tracks the repetition of the values in a given range, here 1e-5.
        If new value is close to the previous, increase counter by one, else reset counter to 1.

        Returns
        -------
        bool
            ``False`` if repetion limit reached. Otherwise ``True``.
        """
        if not np.isclose(self._tmp, self._value, atol=1e-20):
            self._rep_count = 1
            return True
        self._rep_count += 1
        if self._rep_count == 3:
            return False
        return True


class Model:

    """
    Creates a model object for a stream

    """

    def __init__(
        self,
        original_data: np.ndarray,
        masked_data: np.ndarray,
        header: dict,
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
        output: str = "model",
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
        # Ctrl+C interrupt handling
        signal.signal(signal.SIGINT, self._keyboard_int)
        self._ctrlc = False

        self.original_data = original_data
        self.masked_original_data = masked_data
        self.sourcemask = sourcemask

        self._filter = header["FILTER"]  # Filter band the image was taken in.
        pxscale = header["PXSCALE"]  # Pixelscale of the image in arcseconds/pixel.
        psf = header["PSF"]  # Mean FWHM in arcseconds of all image PSF's.
        self._seeing = psf / pxscale / (2 * np.sqrt(2 * np.log(2)))

        print(f"using psf fhwm: {psf} [arcsec]")
        print(f"using pxscale: {pxscale} [arcsec / pixel]")

        self._header = header
        self._vary_box_dim = vary_box_dim
        self._fix_bg = fix_bg
        self._norm_tracker = None
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

    def _keyboard_int(self, *args):
        """
        Handles keyboard interrupts. This is a private method and should not be used outside this class.
        """
        print("\n")
        print("Ctrl+C pressed by user, saving current progress")
        self._save()
        self._ctrlc = True

    def _chek_termination(self):
        """
        Checks various termination conditions. This is a private method and should not be used outside this class.

        Returns
        -------
        int
            -1 if any condition is met, otherwise 0.
        """
        if self._tmp_peak_pos is None:
            print("no peak for a gaussian fit found")
            return -1

        if isinstance(self._tmp_norm_err, float):
            if self._sn_threshold == 0:
                return 0
            if self._tmp_norm / self._tmp_norm_err < self._sn_threshold:
                print("\n")
                print(
                    f"S/N of {self._tmp_norm/self._tmp_norm_err} below threshold of {self._sn_threshold}"
                )
                return -1

        if self._norm_tracker.add_val(value=self._tmp_norm) == -1:
            print("repeating parameters detected")
            return -1
        return 0

    def _paste_modelbox(
        self,
        data: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        angle: float,
        offset: float,
    ):
        """
        Fills the model data with 1d slices. Depending on the box dimensions and the stream angle, the correct
        position and fill area on the model data array is determined.

        Parameters
        ----------
        data : np.ndarray
            Model data of a box.

        x : int
            Central x position of the box/data.

        y : int
            Central y position of the box/data.

        w : int
            Width of the box/data.

        h : int
            Height of the box/data.

        angle : float
            Angle of the Gaussian inside the box.

        offset : float
            Offset/background value of the box.

        """
        # TODO this whole method needs to be overworked for near 45 degree angles.
        data -= offset
        angle = abs(np.degrees(np.arctan(np.tan(np.radians(angle)))))

        # w = h
        if self._vary_box_dim:
            if 40 <= angle <= 50:
                self._tmp_data[y - h : y + h, x - w : x + w] = data
                self.data = np.nanmean([self._tmp_data, self.data], axis=0)
                self._tmp_data *= np.nan
                return

        # w < h
        if w < h:
            self._tmp_data[y - h : y + h, x] = data[:, w + 1]

        # w > h
        if w > h:
            self._tmp_data[y, x - w : x + w] = data[h + 1]

        self.data = np.nanmean([self._tmp_data, self.data], axis=0)
        self._tmp_data *= np.nan

    def build(
        self,
        stepsize: int = 1,
        steps: tuple = (9999, 9999),
        liveplot: bool = False,
        verbose: int = 1,
    ):
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

        x: int = self.init_x
        y: int = self.init_y
        w: int = self.init_w
        h: int = self.init_h

        if isinstance(steps, int):
            raise TypeError(f"'steps'of non-tuple type {type(steps)}")

        if len(steps) != 2:
            raise ValueError(
                f"false number of values to unpack (expected 2 got {len(steps)})"
            )

        if verbose == 1:
            util.intro()

        if liveplot:
            live_plot = LivePlot(
                real_data=self.masked_original_data,
                box_width=self.init_w,
                box_height=self.init_h,
            )

        self._norm_tracker = ParamTracker()

        # fit the init box
        init_box = Box(
            self.masked_original_data,
            x=self.init_x,
            y=self.init_y,
            width=self.init_w,
            height=self.init_h,
            seeing=self._seeing,
            init=[self.init_angle, 1, 1, 0, 0, 0, 0, 0, 0],
            h2=self._h2,
            skew=self._skew,
            h4=self._h4,
            fix_bg=self._fix_bg,
        )

        init_box.fit_model()
        init_box.make_model()

        angle = init_box.angle
        self._tmp_sigma = init_box.sigma

        self._tmp_peak_pos = init_box.peak_pos
        params = init_box.params

        errs = init_box.param_errs

        self.param_errors.append(errs)
        self.params.append(params)
        self.box_properties.append([self.init_x, self.init_y, self.init_w, self.init_h])

        # paste init box
        self._paste_modelbox(
            init_box.model,
            self.init_x,
            self.init_y,
            self.init_w,
            self.init_h,
            angle,
            init_box.offset,
        )

        # get direction options
        dopts = util.get_direction_options(init_box.angle, sectors, direction_dict)
        self.init_direction = dopts[0]
        self.post_direction = dopts[1]
        direction = dopts[0]

        box_id = 1
        file = BoxList(filename=self.output)
        file.write_line(data=[box_id, x, y, w, h, *params])

        # model loop, in direction init_direction, starting from init_x and init_y
        if verbose == 1:
            print("\n")
            print("##### FIRST HALF #####")
            print("\n")

        for run in range(2):
            if run == 1:
                if verbose == 1:
                    print("\n")
                    print("##### SECOND HALF #####")
                    print("\n")
                # setup other direction
                angle = init_box.angle
                params = init_box.params
                x = self.init_x
                y = self.init_y
                w = self.init_w
                h = self.init_h
                self.params = self.params[::-1]
                self.param_errors = self.param_errors[::-1]
                self.box_properties = self.box_properties[::-1]
                direction = self.post_direction
                file.write_line(comment="2nd run")

            for _ in range(steps[run]):
                if verbose == 1:
                    print(f"iteration {_+1}", end="\r")

                # x,y = util.correct_boxcenter(x=x,y=y,w=w,h=h,params=params, peak_pos = self._tmp_peak_pos)
                x, y = util.correct_box_center_from_peak(
                    x=x, y=y, w=w, h=h, peak_pos=self._tmp_peak_pos
                )

                x, y = util.calculate_next_boxcenter(
                    angle=angle,
                    x_center=x,
                    y_center=y,
                    direction=direction,
                    stepsize=stepsize,
                    dictonary=slope_dict,
                )

                if self._vary_box_dim:
                    w, h = util.calculate_new_box_dimensions(
                        angle=angle, init_width=self.init_w, init_height=self.init_h
                    )

                tmp_box = Box(
                    self.masked_original_data,
                    x,
                    y,
                    w,
                    h,
                    init=params,
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

                self._tmp_norm_err = tmp_box.norm_err
                self._tmp_norm = tmp_box.norm

                angle = tmp_box.angle
                params = tmp_box.params

                self._tmp_sigma = tmp_box.sigma

                self._tmp_peak_pos = tmp_box.peak_pos

                dopts = util.get_direction_options(angle, sectors, direction_dict)
                direction = util.get_box_direction(direction, dopts)

                if self._chek_termination() == -1:
                    print("\n")
                    print(f"----> {run+1}. half terminated after {_} steps!")
                    print("\n")
                    break

                self._paste_modelbox(tmp_box.model, x, y, w, h, angle, tmp_box.offset)

                self.param_errors.append(tmp_box.param_errs)
                self.params.append(params)
                self.box_properties.append([x, y, w, h])

                file.write_line(data=[(_ + 1) + (run + 1) / 10, x, y, w, h, *params])

                if liveplot:
                    live_plot.plot(data=self.data, center=[x, y], width=w, height=h)
                del tmp_box

                if self._ctrlc:
                    self._ctrlc = False
                    break

        if liveplot:
            live_plot.close()

        self.param_data = np.array([self.params, self.param_errors])
        self.box_prop_data = np.array([self.box_properties])
        self._save()
        self._create_parameter_table()

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

        # TODO correct header manipulation
        self._header["OBJECT"] = "stream"
        primary_hdu = fits.PrimaryHDU(self.original_data, header=self._header)
        data = np.nan_to_num(self.data.copy(), nan=0)
        self._header["OBJECT"] = "masked intpol stream"
        mrdata_hdu = fits.ImageHDU(self.masked_original_data, header=self._header)
        self._header["OBJECT"] = "sourcemask"
        mask_hdu = fits.ImageHDU(self.sourcemask)
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
        bx = bp[:, :, 0][0]
        bxc = fits.Column(name="box_x", format="I", array=bx)

        # y
        by = bp[:, :, 1][0]
        byc = fits.Column(name="box_y", format="I", array=by)

        # width
        bw = bp[:, :, 2][0]
        bwc = fits.Column(name="box_w", format="I", array=bw)

        # height
        bh = bp[:, :, 3][0]
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
