import argparse
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table


class Modifier:
    def __init__(self, multifits_file: str, param_file: str, lower: int, upper: int):
        plt.ion()
        self._param_file = param_file.removeprefix(".\\")
        self._multifits_file = multifits_file.removeprefix(".\\")
        self._model = fits.getdata(multifits_file, ext=4, memmap=False)
        self._tmp_model = self._model.copy()
        self._data = fits.getdata(multifits_file, ext=2, memmap=False)
        self._table_data, self._table_header = fits.getdata(
            param_file, ext=1, header=True, memmap=False
        )
        self._lower = lower
        self._upper = upper

        # plt.ion()
        self._data[np.isnan(self._data)] = 0
        vmin = np.percentile(self._data, 2)
        vmax = np.percentile(self._data, 98)

        fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
        for a in ax:
            a.axis("off")

        ax[0].imshow(self._data, origin="lower", vmin=vmin, vmax=vmax, cmap="YlOrBr")
        mod = ax[1].imshow(
            self._model, origin="lower", vmin=vmin, vmax=vmax, cmap="YlOrBr"
        )
        res = ax[2].imshow(
            self._data - self._model,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap="YlOrBr",
        )

        self.fig = fig
        self.ax = ax

        self._mod = mod
        self._res = res

        plt.show(block=False)

    def _update(self):
        """
        Plots/updates the figure.

        Parameters
        ----------
        data : np.ndarray
            Image array of the stream model.

        center : list
            List of current box center coordinates.

        width : int
            Width of the current box.

        height : int
            Height of the current box.
        """
        self._mod.set_data(self._tmp_model)
        self._res.set_data(self._data - self._tmp_model)
        self.fig.canvas.draw()
        # self.fig.canvas.flush_events()

    def _modify(self):
        self._tmp_model = self._model.copy()
        ids = np.array([0, 1, 2, 3])
        for row in range(len(self._table_data) - self._upper, len(self._table_data)):
            x, y, w, h = np.array(self._table_data[row])[ids].astype(int)
            if w > h:
                self._tmp_model[y, x - w : x + w] = 0
            else:
                self._tmp_model[y - h : y + h, x] = 0

        for row in range(self._lower):
            x, y, w, h = np.array(self._table_data[row])[ids].astype(int)
            if w > h:
                self._tmp_model[y, x - w : x + w] = 0
            else:
                self._tmp_model[y - h : y + h, x] = 0

        self._update()

    def _close(self):
        """
        Closes the figure.
        """
        plt.close()
        plt.ioff()

    def _save(self):
        table = Table(self._table_data)
        table.remove_rows(np.arange(len(table) - self._upper, len(table), 1))
        table.remove_rows(np.arange(0, self._lower, 1))
        fits.BinTableHDU(table, header=self._table_header).writeto(
            f"mod_{self._param_file}", overwrite=True
        )

        hdul = fits.open(self._multifits_file)
        hdul[4].data = self._tmp_model
        hdul[3].data = self._data - self._tmp_model
        hdul.writeto(f"mod_{self._multifits_file}", overwrite=True)
        hdul.close()

    def do(self):
        # self._modify()

        while True:
            inp = input("New limit ('lower,upper', blank if finished):")
            if len(inp) == 0:
                self._close()
                self._save()
                break
            self._lower, self._upper = np.array(inp.split(","), dtype=int)
            self._modify()
