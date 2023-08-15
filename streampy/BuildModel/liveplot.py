from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np


class LivePlot:
    '''
    Creates a figure to display the live creation of a stream model.
    '''
    def __init__(self, real_data : np.ndarray, box_width : int, box_height : int):
        '''
        Initializes the figure.

        Parameters
        ----------
        real_data : np.ndarray
            Image array of the original image.

        box_width : int
            Width of the current box.

        box_height : int
            Height of the current box.
        '''
        plt.ion()

        self.data = np.zeros(real_data.shape)
        vmin = np.percentile(real_data,2)
        vmax = np.percentile(real_data,98)

        fig, ax = plt.subplots(1,2)

        ax[0].axis("off")
        ax[1].axis("off")
        ax[0].imshow(real_data, origin="lower", vmin=vmin, vmax=vmax,cmap="YlOrBr")

        canvas = ax[1].imshow(self.data, origin="lower", vmin=vmin, vmax=vmax,cmap="YlOrBr")
        self.width=box_width
        self.height=box_height
        
        rect = ax[0].add_patch(Rectangle((0,0),2*box_width+1,2*box_height+1,fill=False,color="red",linewidth=0.5))
        self.fig=fig
        self.ax=ax
        self.rect=rect
        self.canvas=canvas

    def plot(self, data : np.ndarray, center : list, width : int, height : int):
        '''
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
        '''
        self.canvas.set_data(data)
        self.rect.set_xy([center[0]-width,center[1]-height])
        self.rect.set_width(2*width+1)
        self.rect.set_height(2*height+1)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def close(self):
        '''
        Closes the figure.
        '''
        plt.close()
        plt.ioff()