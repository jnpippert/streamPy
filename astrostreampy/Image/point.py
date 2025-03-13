"""
Provides two classes as a convenience
to set the intial box and endpoints of the stream.

Example
-------
>>> from point import Point
>>> from point import InitBox
"""

import sys
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider

__all__ = ["Point", "InitBox"]


@dataclass(eq=True, unsafe_hash=True)
class Point:
    """
    Represents a point in 2D space.

    Attributes
    ----------
    x : int
        The x-coordinate of the point.
    y : int
        The y-coordinate of the point.
    color : str
        The color associated with the point (default is "red").
    """

    x: int = field(default=0, repr=True, compare=True, hash=True)
    y: int = field(default=0, repr=True, compare=True, hash=True)
    color: str = field(
        default="red", metadata={"repr": False, "compare": False, "hash": False}
    )

    def vector_to(self, point: "Point") -> list[int]:
        """
        Computes the vector from this point to another point.

        Parameters
        ----------
        point : Point
            The other point.

        Returns
        -------
        List[int]
            The vector from this point to the given point.
        """
        return [point.x - self.x, point.y - self.y]

    def distance_to(self, point: "Point") -> float:
        """
        Computes the Euclidean distance from this point to another point.

        Parameters
        ----------
        point : Point
            The other point.

        Returns
        -------
        float
            The Euclidean distance between this point and the given point.
        """
        vec = self.vector_to(point)
        return np.sqrt(vec[0] ** 2 + vec[1] ** 2)

    def isclose(self, point: "Point", tol: int = 10) -> bool:
        """
        Checks if another point is close to this point within a tolerance.

        Parameters
        ----------
        point : Point
            The other point.
        tol : int, optional
            The tolerance within which points are considered close, by default 10.

        Returns
        -------
        bool
            True if the other point is close within the tolerance, False otherwise.
        """
        if abs(self.x - point.x) <= tol and abs(self.y - point.y) <= tol:
            return True
        return False

    def toarray(self) -> np.ndarray:
        """
        Converts the point to a NumPy array.

        Returns
        -------
        np.ndarray
            A NumPy array representing the coordinates of the point.
        """
        return np.array([self.x, self.y])


class InitBox(Point):
    """
    Represents an interactive initialization box.

    Parameters
    ----------
    data : np.ndarray
        The data to be visualized.
    cmap : str, optional
        The colormap string, by default "YlOrBr".
    color : str, optional
        The color of the point, by default "red".
    """

    def __init__(self, data: np.ndarray, cmap: str = "YlOrBr", color: str = "red"):
        super().__init__()

        # set nans to zero
        data = np.nan_to_num(data, nan=0)

        # calculate cut levels
        vmin, vmax = np.nanpercentile(data, (5, 95))

        # initialize plot
        fig, ax = plt.subplots()

        ax.set_title(
            r"$to$ $switch$ $modes$ $press:$ $'a'$, $'w'$ and $'d'$", color="dimgray"
        )
        ax.axis("off")
        color = "red"
        (point,) = ax.plot([], [], ".", color=color)
        (p1,) = ax.plot([], [], ".", color=color)
        (p2,) = ax.plot([], [], ".", color=color)
        img = ax.imshow(data, vmin=vmin, vmax=vmax, origin="lower", cmap=cmap)

        # initialize vmin slider
        ax_vmin = fig.add_axes([0.2, 0.07, 0.7, 0.03])

        vmin_slider = Slider(
            ax=ax_vmin,
            label="shadows",
            valmin=np.min(data),
            valmax=np.max(data),
            valinit=vmin,
            valstep=abs(vmin) / 10,
        )
        vmin_slider.on_changed(self._update)

        # initialize vmax slider
        ax_vmax = fig.add_axes([0.2, 0.03, 0.7, 0.03])
        vmax_slider = Slider(
            ax=ax_vmax,
            label="highlights",
            valmin=np.min(data),
            valmax=np.max(data),
            valinit=vmax,
            valstep=abs(vmax) / 10,
        )
        vmax_slider.on_changed(self._update)

        rect = ax.add_patch(
            Rectangle(
                [0, 0],
                width=0,
                height=0,
                fill=False,
                color=color,
                linewidth=0.8,
            )
        )

        # initialize width slider
        ax_width = fig.add_axes([0.03, 0.2, 0.03, 0.7])
        width_slider = Slider(
            ax=ax_width,
            label="width",
            valmin=0,
            valmax=data.shape[1] / 3,
            valinit=0,
            valstep=1,
            orientation="vertical",
        )
        width_slider.on_changed(self._update)

        # initialize height slider
        ax_height = fig.add_axes([0.13, 0.2, 0.03, 0.7])
        height_slider = Slider(
            ax=ax_height,
            label="height",
            valmin=0,
            valmax=data.shape[0] / 3,
            valinit=0,
            valstep=1,
            orientation="vertical",
        )
        height_slider.on_changed(self._update)

        # create level reset button
        ax_reset_levels = fig.add_axes([0.82, 0.4, 0.15, 0.04])
        level_button = Button(
            ax=ax_reset_levels,
            label="reset cuts",
            color="darkgray",
            hovercolor="lightgray",
        )
        level_button.on_clicked(self._reset_levels)

        # create level reset button
        ax_reset_rect = fig.add_axes([0.82, 0.7, 0.15, 0.04])
        rect_button = Button(
            ax=ax_reset_rect,
            label="reset box",
            color="darkgray",
            hovercolor="lightgray",
        )
        rect_button.on_clicked(self._reset_rect)

        # set attributes
        self.width = 0
        self.height = 0
        self._rect = rect
        self._vmin_slider = vmin_slider
        self._vmax_slider = vmax_slider
        self._width_slider = width_slider
        self._height_slider = height_slider
        self._fig = fig
        self._img = img
        self._point = point
        self._p1 = p1
        self._p2 = p2
        self.tail = None
        self.head = None
        self._pointmode = True
        self._p1mode = False
        self._p2mode = False

        self._cid = point.figure.canvas.mpl_connect(
            "button_press_event", self._mouse_click
        )
        self._fig.suptitle(r"set init point and box", color="k")

        self._fig.canvas.mpl_connect("key_press_event", self._key_press_event)

        plt.show()

    def _key_press_event(self, event):
        sys.stdout.flush()
        if event.key == "a":
            print("tail mode")
            self.tail = Point()
            self._p1mode = True
            self._p2mode = False
            self._pointmode = False
            self._fig.suptitle(r"set first end point (tail)", color="k")
            self._fig.figure.canvas.draw()
        if event.key == "d":
            self.head = Point()
            print("head mode")
            self._p1mode = False
            self._p2mode = True
            self._pointmode = False
            self._fig.suptitle(r"set second end point (head)", color="k")
            self._fig.figure.canvas.draw()
        if event.key == "w":
            print("center mode")
            self._p1mode = False
            self._p2mode = False
            self._pointmode = True
            self._fig.suptitle(r"set init point and box", color="k")
            self._fig.figure.canvas.draw()

    def _mouse_click(self, event):
        """
        Handles mouse clicking events in the matplotlib API.
        This is a private method and should not be used outside this class.
        """

        if event.inaxes != self._point.axes:
            return
        x = int(np.round(event.xdata, 0))
        y = int(np.round(event.ydata, 0))
        if self._p1mode:
            self.tail.x = x
            self.tail.y = y
            self._fig.suptitle(f"{x = }, {y = }")
            self._p1.set_data(x, y)

        if self._p2mode:
            self.head.x = x
            self.head.y = y
            self._fig.suptitle(f"{x = }, {y = }")
            self._p2.set_data(x, y)

        if self._pointmode:
            self.x = x
            self.y = y
            self._fig.suptitle(f"{x = }, {y = }")
            self._point.set_data(x, y)
            self._rect.set_xy([x - self.width, y - self.height])

        self._fig.figure.canvas.draw()

    def _update(self, val):
        """
        Updates the figure. This is a private method and should not be used outside this class.
        """
        self._img.set_clim([self._vmin_slider.val, self._vmax_slider.val])
        self._rect.set_width(2 * self._width_slider.val)
        self._rect.set_height(2 * self._height_slider.val)
        self.width = self._width_slider.val
        self.height = self._height_slider.val
        self._rect.set_xy([self.x - self.width, self.y - self.height])
        self._fig.canvas.draw()

    def _reset_levels(self, event):
        """
        Resets the cut levels. This is a private method and should not be used outside this class.
        """
        self._vmin_slider.reset()
        self._vmax_slider.reset()

    def _reset_rect(self, event):
        """
        Resets the box dimensions. This is a private method and should not be used outside this class.
        """
        self._width_slider.reset()
        self._height_slider.reset()
