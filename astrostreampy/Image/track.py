import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.ticker import MultipleLocator
from scipy import stats
from scipy.interpolate import BSpline, splrep

from .point import Point

__all__ = ["StreamTrack"]


class StreamTrack:
    """
    Class representing a stream track with methods for analysis and visualization.

    Parameters
    ----------
    paramfile : str
        Path to the FITS file containing parameter data.
    multifits_file : str
        Path to the FITS file containing stream data.

    Attributes
    ----------
    points : numpy.ndarray
        Array of points representing the stream track.
    stream : numpy.ndarray
        Array representing the stream data.
    length : float
        Length of the stream track.
    bin : int
        Binning factor for the stream track.

    Methods
    -------
    show(out: str = None, dpi: int = 300)
        Display the stream data with the stream track overlay.
    fit_spline(k: int = 2, s: float = 1e6)
        Fit a B-spline to the stream track points.

    Example
    -------
    >>> track = StreamTrack('paramfile.fits', 'multifits_file.fits')
    >>> track.fit_spline()
    >>> track.show(out='output.png', dpi=300)
    """

    def __init__(self, paramfile, multifits_file):
        """
        Initialize a StreamTrack instance.

        Parameters
        ----------
        paramfile : str
            Path to the FITS file containing parameter data.
        multifits_file : str
            Path to the FITS file containing stream data.

        Attributes
        ----------
        points : numpy.ndarray
            Array of points representing the stream track.
        stream : numpy.ndarray
            Array representing the stream data.
        length : float
            Length of the stream track.
        bin : int
            Binning factor for the stream track.
        """
        table_data = fits.getdata(paramfile, ext=1)
        self.points = np.array(
            [[x, table_data["box_y"][i]] for i, x in enumerate(table_data["box_x"])]
        )
        self._flipped = False
        self.stream: np.ndarray = fits.getdata(multifits_file, ext=2)
        self.length: float = np.nan
        self.bin = 5
        self.spoints = None
        self.spline = None

    def _flip(self):
        """
        Flip the x and y coordinates if the mode of 'box_y' in parameter data is greater than 1.
        """
        if stats.mode(self.points).count[0][0] > 1:
            self._flipped = True
            self.points = np.flip(self.points)[::-1]

    def _sort(self):
        """
        Sort the points of the stream track based on the x-coordinate.
        """
        self.points = self.points[np.argsort(self.points[:, 0])]

    def _bin(self):
        """
        Bin the stream track points with a specified binning factor.

        Parameters
        ----------
        bin : int, optional
            Binning factor for the stream track. Default is 5.
        """

        points = []
        binned_points = []
        for i, point in enumerate(self.points):
            points.append(point)
            if i % self.bin == 0:
                binned_points.append(np.mean(points, axis=0))
                points = []
        self.points = np.array(binned_points)

    def _length(self):
        """
        Calculate and set the length of the stream track.
        """
        i = 1
        l = 0
        while i < len(self.spoints):
            l += Point(*self.spoints[i]).distance_to(Point(*self.spoints[i - 1]))
            i += 1
        self.length = l

    def show(self, out: str = None, dpi: int = 300):
        """
        Display the stream data with the stream track overlay.

        Parameters
        ----------
        out : str, optional
            Output file path. If provided, saves the plot to the specified file.
        dpi : int, optional
            Dots per inch for the saved plot. Default is 300.
        """
        vmin, vmax = np.percentile(self.stream, (2, 98))
        _, ax = plt.subplots(1, figsize=(10, 10))
        im = ax.imshow(
            self.stream,
            vmin=vmin,
            vmax=vmax,
            interpolation="bicubic",
            cmap="inferno",
            origin="lower",
        )
        ax.tick_params(
            which="major",
            direction="in",
            bottom=True,
            labelbottom=True,
            top=True,
            labeltop=False,
            length=10,
            width=1.5,
            left=True,
            labelleft=True,
            right=True,
            labelright=False,
        )
        ax.tick_params(
            which="minor",
            direction="in",
            bottom=True,
            labelbottom=True,
            top=True,
            labeltop=False,
            length=5,
            width=1,
            left=True,
            labelleft=True,
            right=True,
            labelright=False,
        )
        ax.xaxis.set_minor_locator(MultipleLocator(25))
        ax.yaxis.set_minor_locator(MultipleLocator(25))
        ax.set_xticks([i * 100 for i in range(1, self.stream.shape[1] // 100)])
        ax.set_yticks([i * 100 for i in range(1, self.stream.shape[0] // 100)])
        ax.set_xlabel("x [px]")
        ax.set_ylabel("x [px]")
        plt.colorbar(
            im,
            ax=ax,
            location="top",
            orientation="horizontal",
            shrink=0.75,
            label="Flux",
            fraction=0.05,
        )
        if self._flipped:

            ax.plot(self.spoints[:, 1], self.spoints[:, 0], lw=2, color="k")
        else:
            ax.plot(self.spoints[:, 0], self.spoints[:, 1], lw=2, color="k")
        if out is not None:
            if not out.endswith((".png", ".jpg")):
                out += ".png"
            plt.savefig(out, dpi=dpi)
        plt.show()

    def fit_spline(self, k: int = 2, s: float = 1e6, bin_factor: int = 5):
        """
        Fit a B-spline to the stream track points.

        Parameters
        ----------
        k : int, optional
            Degree of the B-spline. Default is 2.
        s : float, optional
            Smoothness parameter. Default is 1e6.
        """
        self.bin = bin_factor
        self._bin()
        self._flip()
        self._sort()
        tck = splrep(self.points[:, 0], self.points[:, 1], s=s, k=k)
        self.spline = BSpline(*tck)
        self.spoints = np.array([[x, self.spline(x)] for x in self.points[:, 0]])
        self._length()
