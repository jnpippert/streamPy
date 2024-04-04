"""
Provides two classes as a convenience
inspect the model quality and to inspect images for tidal features.

Example
-------
>>> from inspect import Slice
>>> from inspect import Inspect
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider

__all__ = ["Slice", "Inspect"]


class Stretch:
    """A class for stretching FITS data."""

    def __init__(self, filename: str, ext: int = 0):
        """
        Initialize Stretch object.

        Parameters
        ----------
        filename : str
            The filename of the FITS file.
        ext : int, optional
            The extension number to read from the FITS file, by default 0
        """
        self._data = fits.getdata(filename, ext)
        self._aperture = fits.getdata(filename, ext=5)
        self._data = np.nan_to_num(self._data, nan=0)
        self._lin_scale()
        self._calc_cuts()
        self._stretched_data = None

    def _asinh_scale(self, array):
        """
        Apply the arcsinh stretch to the given array.

        Parameters
        ----------
        array : np.ndarray
            The input array.

        Returns
        -------
        np.ndarray
            The array after applying the arcsinh stretch.
        """
        self._stretched_data = np.arcsinh(self._data)
        if isinstance(array, np.ndarray):
            return np.arcsinh(array)

    def _log_scale(self, array):
        """
        Not implemented.

        Raises
        ------
        NotImplementedError
            This method is not implemented.
        """
        raise NotImplementedError

    def _lin_scale(self, array: np.ndarray = None):
        """
        Apply a linear stretch to the given array.

        Parameters
        ----------
        array : np.ndarray, optional
            The input array, by default None

        Returns
        -------
        np.ndarray
            The array after applying the linear stretch.
        """
        self._stretched_data = self._data
        if isinstance(array, np.ndarray):
            return array

    def _calc_cuts(self):
        """
        Calculate vmin and vmax values for stretching the data.
        """
        self._vmin = np.percentile(self._stretched_data, 10)
        self._vmax = np.percentile(self._stretched_data, 90)


class Slice(Stretch):
    """
    Loads the Slice Inspection tool.

    Its a user convenience to inspect the quality of the model.
    """

    # TODO log and asinh buttons
    # TODO vmin and vmax slider
    def __init__(self, filename: str, paramfile: str):
        super().__init__(filename, ext=2)

        self._model = fits.getdata(filename, ext=4)
        self._model = np.nan_to_num(self._model, nan=0)
        self._params = fits.getdata(paramfile, ext=1)
        init_n = int(len(self._params) / 2)
        self._offset = self._params[init_n][10]
        self._index = init_n
        self._x, self._y, self._width, self._height = self._params[init_n][:4]
        self._lin_scale()
        self._calc_cuts()
        self._stretched_model = self._lin_scale(array=self._model)

        self._fig = plt.figure(figsize=(15, 6))

        self._fig.suptitle(r"$left$ $click$ $to$ $set$ $slice$")
        self._grid = GridSpec(3, 7)

        # image axis
        self._iax = self._fig.add_subplot(self._grid[0:3, 0:3])
        self._img = self._iax.imshow(
            self._stretched_data,
            vmin=self._vmin,
            vmax=self._vmax,
            origin="lower",
            cmap="YlOrBr",
        )
        self._iax.set_xlabel("x [px]")
        self._iax.set_ylabel("y [px]")
        self._iax.set_xticks([100, 200, 300, 400, 500, 600, 700, 800, 900])
        self._iax.set_yticks([100, 200, 300, 400, 500, 600, 700, 800, 900])

        self._slice()
        # image slice axis
        self._isaxd = self._fig.add_subplot(self._grid[0, 3:], aspect=1)
        self._simgd = self._isaxd.imshow(
            self._data_slice_arr,
            vmin=self._vmin,
            vmax=self._vmax,
            origin="lower",
            cmap="YlOrBr",
        )

        # model slice axis
        self._isaxm = self._fig.add_subplot(self._grid[1, 3:], aspect=1)
        self._simgm = self._isaxm.imshow(
            self._model_slice_arr,
            vmin=self._vmin,
            vmax=self._vmax,
            origin="lower",
            cmap="YlOrBr",
        )

        # plot axis
        self._pax = self._fig.add_subplot(self._grid[2, 3:])
        (self._plot_gauss,) = self._pax.plot(
            self._gauss1d + self._offset, label="model"
        )
        self._pax.set_ylim(
            np.nanmin(self._data1d - self._gauss1d) - 1, np.nanmax(self._data1d) + 1
        )
        (self._plot_data,) = self._pax.plot(self._data1d, ".", label="data")
        (self._plot_res,) = self._pax.plot(
            self._data1d - self._gauss1d, color="k", lw=0.8, label="residual"
        )
        self._plot_meanres = self._pax.axhline(
            np.nanmean(self._data1d - self._gauss1d),
            color="k",
            lw=0.8,
            linestyle="dashed",
        )
        (self._plot_aper,) = self._pax.plot(
            self._aper1d, color="k", ls="dotted", label="mask"
        )
        self._plot_hm = self._pax.axhline(np.max(self._gauss1d) / 2, lw=0.8)
        # point
        (self._point,) = self._iax.plot([self._x], [self._y], ".", color="red")

        # rectangle
        self._rect = self._iax.add_patch(
            Rectangle(
                [self._x - self._width, self._y - self._height],
                width=2 * self._width + 1,
                height=2 * self._height + 1,
                fill=False,
                color="red",
                linewidth=0.8,
            )
        )

        # initialize width slider
        ax_width = self._fig.add_axes([0.03, 0.2, 0.01, 0.7])
        self._width_slider = Slider(
            ax=ax_width,
            label="width",
            valmin=0,
            valmax=self._data.shape[1] / 2,
            valinit=self._width,
            valstep=1,
            orientation="vertical",
        )
        self._width_slider.on_changed(self._update)

        # initialize height slider
        ax_height = self._fig.add_axes([0.065, 0.2, 0.01, 0.7])
        self._height_slider = Slider(
            ax=ax_height,
            label="height",
            valmin=0,
            valmax=self._data.shape[0] / 2,
            valinit=self._height,
            valstep=1,
            orientation="vertical",
        )
        self._height_slider.on_changed(self._update)

        # prev button
        ax_prev_button = self._fig.add_axes([0.15, 0.82, 0.05, 0.05])
        self._prev_button = Button(ax_prev_button, "previous")
        self._prev_button.on_clicked(self._previous)

        # next button
        ax_prev_button = self._fig.add_axes([0.35, 0.82, 0.05, 0.05])
        self._next_button = Button(ax_prev_button, "next")
        self._next_button.on_clicked(self._next)

        self._cid = self._point.figure.canvas.mpl_connect(
            "button_press_event", self._mouse_click
        )
        self._pax.legend(ncol=2)
        # plt.savefig("screen_slice_inspection.pdf")
        plt.show()

    def _nearest_index(self):
        # used to update index depending on mouse click events
        raise NotImplementedError

    def _previous(self, event):
        if self._index == 0:
            self._index = len(self._params) - 1
        else:
            self._index -= 1
        self._x = self._params[self._index][0]
        self._y = self._params[self._index][1]
        self._point.set_data(self._x, self._y)
        self._rect.set_xy([self._x - self._width, self._y - self._height])
        self._update_slices()
        self._iax.set_title(f"x = {self._x}, y = {self._y}")

    def _next(self, event):
        if self._index == len(self._params) - 1:
            self._index = 0
        else:
            self._index += 1
        self._x = self._params[self._index][0]
        self._y = self._params[self._index][1]
        self._point.set_data(self._x, self._y)
        self._rect.set_xy([self._x - self._width, self._y - self._height])
        self._update_slices()
        self._iax.set_title(f"x = {self._x}, y = {self._y}")

    def _update_slices(self):
        self._slice()

        xarr = np.arange(0, self._data1d.size, 1)
        mv = np.nanmax(self._data1d)
        self._pax.set_ylim(
            np.nanmin(self._data1d - self._gauss1d) - mv / 5,
            np.nanmax(self._data1d) + mv / 5,
        )
        self._pax.set_xlim(0, self._data1d.size)
        self._plot_data.set_data(xarr, self._data1d)
        self._offset = self._params[self._index][10]
        if np.sum(self._gauss1d) == 0:
            self._offset = 0
        self._plot_gauss.set_data(xarr, self._gauss1d + self._offset)
        self._plot_res.set_data(xarr, self._data1d - self._gauss1d)

        self._simgd.set_data(self._data_slice_arr)
        self._simgm.set_data(self._model_slice_arr)
        self._plot_meanres.set_ydata(np.nanmean(self._data1d - self._gauss1d))
        self._plot_aper.set_ydata(self._aper1d)
        self._plot_hm.set_ydata(np.max(self._gauss1d) / 2)

    def _mouse_click(self, event):
        if event.inaxes != self._point.axes:
            return
        x = int(np.round(event.xdata, 0))
        y = int(np.round(event.ydata, 0))
        self._x = x
        self._y = y
        self._iax.set_title(f"{x = }, {y = }")
        self._point.set_data(x, y)
        self._rect.set_xy([self._x - self._width, self._y - self._height])

        self._update_slices()
        self._fig.canvas.draw()

    def _update(self, val):
        self._rect.set_width(2 * self._width_slider.val + 1)
        self._rect.set_height(2 * self._height_slider.val + 1)
        self._width = self._width_slider.val
        self._height = self._height_slider.val
        self._rect.set_xy([self._x - self._width, self._y - self._height])
        self._update_slices()
        self._fig.canvas.draw()

    def _slice(self):
        self._model_slice_arr = self._stretched_model[
            self._y - self._height : self._y + self._height,
            self._x - self._width : self._x + self._width,
        ]
        self._data_slice_arr = self._stretched_data[
            self._y - self._height : self._y + self._height,
            self._x - self._width : self._x + self._width,
        ]
        self._data_slice_arr = np.nan_to_num(self._data_slice_arr, nan=0)
        # _, self._gauss1d = self._mean_box(self._model[self._y-self._height:self._y+self._height,self._x-self._width:self._x+self._width])
        # _, self._data1d = self._mean_box(self._data[self._y-self._height:self._y+self._height,self._x-self._width:self._x+self._width])
        if self._width > self._height:
            self._gauss1d = self._model[
                self._y, self._x - self._width : self._x + self._width
            ]
            self._aper1d = self._aperture[
                self._y, self._x - self._width : self._x + self._width
            ]
        else:
            self._gauss1d = self._model[
                self._y - self._height : self._y + self._height, self._x
            ]
            self._aper1d = self._aperture[
                self._y - self._height : self._y + self._height, self._x
            ]

        self._data1d = self._stack_data_slices(
            w=self._width, h=self._height, x=self._x, y=self._y
        )
        self._data1d = np.nan_to_num(self._data1d, nan=0)
        self._aper1d *= 100

    def _stack_data_slices(self, w: int, h: int, x: int, y: int):
        box = self._data[y - h : y + h, x - w : x + w]
        box[box == 0] = np.nan
        if w > h:
            return np.nanmean(box, axis=0)
        return np.nanmean(box, axis=1)

    def _mean_box(self, array):
        array[array == 0] = np.nan
        angle = self._params[self._index][4]
        x0 = self._params[self._index][12]
        y0 = self._params[self._index][14]
        yg, xg = np.ogrid[
            0 - self._height : self._height, 0 - self._width : self._width
        ]
        grid = np.sin(np.radians(angle)) * (yg - y0) + np.cos(np.radians(angle)) * (
            xg - x0
        )
        xl = []
        yl = []

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                xl.append(grid[i][j])
                yl.append(array[i][j])

        sort_ids = np.argsort(xl)

        sort_x = np.array(xl)[sort_ids]
        sort_y = np.array(yl)[sort_ids]
        res = 0.5
        binned_x = np.arange(int(np.min(sort_x)) - 1, int(np.max(sort_x)), res)
        binned_y = []

        for lim in binned_x:
            ids = np.where((sort_x > lim) & (sort_x <= lim + res))
            if len(ids) > 0:
                binned_y.append(np.nanmean(sort_y[ids]))

        return np.array(binned_x), np.array(binned_y)


class Inspect:
    """
    Class to load the inspection window on instantiation of the class.


    Initialize Inspect object. Is used to inspect a large wide field image. It
    converts the image into grid cells and loads each cell into a inspection window.
    The user then can by pressing keys and mouse to select detected features.

    Left mouse click marks a feature with a dot.
    Pressing 'e' enters the current feature and store it into the
    ``features`` and ``positions`` list.
    Pressing 'd' loads the next cell and adds the current feature to the lists. If no
    feature is clicked, it only loads the cell.
    Pressing 'a' loads the previous cell and adds the current feature to the lists. If no
    feature is clicked, it only loads the cell.

    The inspection is finished when closing the inspection window.

    Parameters
    ----------
    filename : str
        The filename of the FITS file.
    cell_size : int, optional
        The size of each cell. Default is 2000 pixel.
    vmin : float
        The lower scale limit for the image colormapping.
    vmax : float
        The uper scale limit for the image colormapping.

    Attributes
    ----------
    features : list
        List of the features. For now every feature is assigned the name 'feature'.
    positions : list
        List of the positions of the features. Contains the x and y and the RA and DEC
        position in the whole stack image.
    """

    def __init__(
        self,
        filename: str,
        cell_size: int = 2000,
        vmin: float = None,
        vmax: float = None,
    ):
        self.filename = filename
        self.stackdata, self.stackheader = fits.getdata(filename, header=True)
        self.stackdata = np.nan_to_num(self.stackdata, nan=0)
        self.stackwcs = WCS(self.stackheader)
        self._cell_size = cell_size
        self._stack_height, self._stack_width = self.stackdata.shape
        self._lower_offsets = []
        self._left_offsets = []
        self._feature = None
        self._pos = None
        self._cells = []
        self._cell_id = 0
        self._aquire_cells()
        self.features = []
        self.positions = []
        self.x = None
        self.y = None
        self.ra = None
        self.dec = None
        fig, ax = plt.subplots()
        fig.suptitle(f"cell {self._cell_id + 1}")
        ax.set_title(r"$click$ $to$ $set$ $point$", color="dimgray")
        ax.axis("off")
        (point,) = ax.plot([], [], "o", markersize=10, markeredgecolor="red")
        y, x = self.stackdata.shape
        y = int(y / 2)
        x = int(x / 2)
        if vmin is None:
            vmin = np.percentile(self.stackdata, 2)
        if vmax is None:
            vmax = np.percentile(self.stackdata, 98)
        print(f"vmin = {vmin}, vmax = {vmax}")
        img = ax.imshow(
            self._cells[0],
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            cmap="YlOrBr_r",
            interpolation="bicubic",
        )

        self._fig = fig
        self._ax = ax
        self._point = point
        self._img = img
        fig.canvas.mpl_connect("key_press_event", self._key_press)
        self._cid = point.figure.canvas.mpl_connect(
            "button_press_event", self._mouse_click
        )
        self._fig.suptitle(f"cell {self._cell_id + 1}, feature: {self._feature}")
        plt.show()

    def _key_press(self, event):
        sys.stdout.flush()
        if event.key == "e":
            if isinstance(self._point.get_xdata()[0], int):
                self.features.append("feature")
                self.positions.append((self.x, self.y, self.ra, self.dec))
                self._point.set_data([], [])
                self._feature = None
                print("feature added")
            self._point.figure.canvas.draw()
            self._fig.suptitle(f"cell {self._cell_id}, feature: {self._feature}")
        if event.key == "d":
            self._cell_id += 1
            if self._cell_id == len(self._cells):
                self._cell_id = 0
            self._img.set_data(self._cells[self._cell_id])
            self._fig.suptitle(f"cell {self._cell_id + 1}, feature: {self._feature}")
            if isinstance(self._point.get_xdata()[0], int):
                self.features.append("feature")
                self.positions.append((self.x, self.y, self.ra, self.dec))
                self._point.set_data([], [])
                self._feature = None
                print("feature added")
            self._point.figure.canvas.draw()
            self._fig.suptitle(f"cell {self._cell_id + 1}, feature: {self._feature}")
        if event.key == "a":
            self._cell_id -= 1
            if self._cell_id == -1:
                self._cell_id = len(self._cells) - 1
            self._img.set_data(self._cells[self._cell_id])
            self._fig.suptitle(f"cell {self._cell_id + 1}, feature: {self._feature}")
            if isinstance(self._point.get_xdata()[0], int):
                self.features.append("feature")
                self.positions.append((self.x, self.y, self.ra, self.dec))
                self._point.set_data([], [])
                self._feature = None
                print("feature added")
            self._point.figure.canvas.draw()
            self._fig.suptitle(f"cell {self._cell_id + 1}, feature: {self._feature}")

    def _mouse_click(self, event):
        if event.inaxes != self._point.axes:
            return

        self._feature = "feature"
        x = int(np.round(event.xdata, 0))
        y = int(np.round(event.ydata, 0))
        self.x = x + self._left_offsets[self._cell_id]
        self.y = y + self._lower_offsets[self._cell_id]
        coord = self.stackwcs.pixel_to_world(x, y)
        self.ra = coord.ra.to_value() / 15
        self.dec = coord.dec.to_value()
        self._point.set_data(x, y)
        self._point.figure.canvas.draw()
        self._fig.suptitle(f"cell {self._cell_id + 1}, feature: {self._feature}")

    def _get_cell(self, left_index, right_index, lower_index, upper_index):
        return self.stackdata[lower_index:upper_index, left_index:right_index]

    def _aquire_cells(self):

        left_index = 0
        while left_index < self._stack_width:
            lower_index = 0
            right_index = min(left_index + self._cell_size, self._stack_width)
            while lower_index < self._stack_height:
                upper_index = min(lower_index + self._cell_size, self._stack_height)
                d = self._get_cell(left_index, right_index, lower_index, upper_index)
                if np.median(d) != 0:  # removes low S/N border cells
                    self._cells.append(d)
                    self._lower_offsets.append(lower_index)
                    self._left_offsets.append(left_index)
                lower_index += self._cell_size
            left_index += self._cell_size
