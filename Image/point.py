import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider


class Point:
    """
    Class to define the starting point of the stream fitting algorithm. Additionally the initial
    box dimensions are set with it.

    This class can be used with an already existing plot, created outside of this class.
    The start point (x,y) can be set with a left mouse click (handled in the ``_mouse_click()`` method.

    """

    def __init__(
        self,
        data: np.ndarray,
        cmap: str = "YlOrBr",
        color: str = "red",
        xy: list = [0, 0],
        wh: list = [0, 0],
    ):
        """
        Initializes the figure.

        Parameters
        ----------
        data : np.ndarray
            Image array.

        cmap : str, optional
            (``matplotlib``) Colormap of the image. Default is "YlOrBr".

        color : color, optional
            (``matplotlib``) Color of the point. Default is "red".

        xy : list, optional
            List of the x and y coordinates of the point. Default is [0,0].

        wh : list, optional
            List of the width and height of the initial box. Default is [0,0].
        """
        data = np.nan_to_num(data, nan=0)

        # calculate cut levels
        vmin = np.percentile(data, 5)
        vmax = np.percentile(data, 95)

        # initialize plot
        fig, ax = plt.subplots()
        ax.set_title(r"$click$ $to$ $set$ $point$", color="dimgray")
        ax.axis("off")
        (point,) = ax.plot([], [], ".", color=color)
        img = ax.imshow(data, vmin=vmin, vmax=vmax, origin="lower", cmap=cmap)

        # initialize vmin slider
        ax_vmin = fig.add_axes([0.2, 0.07, 0.7, 0.03])
        vmin_slider = Slider(
            ax=ax_vmin,
            label="shadows",
            valmin=np.percentile(data, 1),
            valmax=np.percentile(data, 99),
            valinit=vmin,
            valstep=vmin / 10,
        )
        vmin_slider.on_changed(self._update)

        # initialize vmax slider
        ax_vmax = fig.add_axes([0.2, 0.03, 0.7, 0.03])
        vmax_slider = Slider(
            ax=ax_vmax,
            label="highlights",
            valmin=np.percentile(data, 1),
            valmax=np.percentile(data, 99),
            valinit=vmax,
            valstep=vmax / 10,
        )
        vmax_slider.on_changed(self._update)

        # initialize rectangle
        width, height = wh

        rect = ax.add_patch(
            Rectangle(
                [0, 0],
                width=width,
                height=height,
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
            valinit=width,
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
            valinit=height,
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
        self.x, self.y = xy

        self.width = width
        self.height = height
        self._rect = rect
        self._vmin_slider = vmin_slider
        self._vmax_slider = vmax_slider
        self._width_slider = width_slider
        self._height_slider = height_slider
        self._fig = fig
        self._img = img
        self._point = point
        self._cid = point.figure.canvas.mpl_connect(
            "button_press_event", self._mouse_click
        )

        plt.show()

    def _mouse_click(self, event):
        """
        Handles mouse clicking events in the matplotlib API.
        This is a private method and should not be used outside this class.
        """

        if event.inaxes != self._point.axes:
            return

        x = int(np.round(event.xdata, 0))
        y = int(np.round(event.ydata, 0))
        self.xy = (x, y)
        self.x = x
        self.y = y
        self._fig.suptitle(f"{x = }, {y = }")
        self._point.set_data(x, y)
        self._rect.set_xy([x - self.width, y - self.height])
        self._point.figure.canvas.draw()

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
