import numpy as np
from astropy.io import fits

from . import utilities as util


class Stream:
    """
    Streampy Stream class. Used to prepare image data.
    """

    def __init__(
        self,
        filename: str,
        masks: list = [],
        interpolation_masks: list = [],
        angle: float = 0,
    ):
        """
        Creates a Stream object containing the image data.

        Parameters
        ----------
        filename : str, path-like
            Filename of a FITS file. If the file is a multiple extension FITS, the desired
            data-array needs to be the first entry in the HDUL.

        masks : list, optional
            List of filenames of one or multiple masks. If any file is a multiple extension FITS, the desired
            data-array need to be the first entry in the HDUL. The masks are used to mask other light sources in the image.

        interpolation_masks : list, optional
            List of filenames of one or multiple interpolation masks. If any file is a multiple extension FITS, the desired
            data-array need to be the first entry in the HDUL. The masks are used to mask light sources which lie in or overlap
            with the stream. Masked pixels from the interpolation masks are later interpolated.

        save_mask : bool, optional
            If ``True`` the masks of the parsed mask-lists are combined to one and saved as a fits file respectively.
            By default this parameter is set to ``True``.

        angle : float, optional
            The approximate angle of the stream. This is used to increase the quality of the interpolation process, by
            interpolating along the stream with respect to the given angle.
        """

        filename = filename.split(".\\")[-1]  # for windows
        self.filename = filename
        self.data, self.header = fits.getdata(filename, header=True)
        self.original_data = self.data.copy()

        self._angle = angle
        self._masks = masks
        self._interpolation_masks = interpolation_masks

    def apply_masks(self):
        """
        Applies all masks onto ``data``, either as source masks or to interpolate zero pixels.
        """

        if len(self._interpolation_masks) != 0:
            print("mask and interpolate sources ...")

            if not isinstance(self._interpolation_masks, list):
                raise TypeError(
                    f"'interpolation_masks' of non-list type {type(self._interpolation_masks)}"
                )

            self.interpolation_mask = np.ones(self.data.shape)

            for mask in self._interpolation_masks:
                if isinstance(mask, str):
                    try:
                        if mask.startswith(".\\"):  # for windows
                            mask = mask[2:]
                        mask = fits.getdata(mask)
                    except:
                        print(f"[WARNING] No path or file called {mask} found.")
                        continue

                if not self.data.shape == mask.shape:
                    raise ValueError(
                        f"Mask of wrong shape (expected {self.data.shape}, got {mask.shape})"
                    )

                self.data = util.multiply_mask(data=self.data, mask=mask)
                self.interpolation_mask = util.multiply_mask(
                    data=self.interpolation_mask, mask=mask
                )

            self.data = util.interpolate_zero_pixels(data=self.data, theta=self._angle)

        if len(self._masks) != 0:
            print("mask sources ...")
            if not isinstance(self._masks, list):
                raise TypeError(f"masks of non-list type {type(self._masks)}")

            self.mask = np.ones(self.data.shape)

            for mask in self._masks:
                if isinstance(mask, str):
                    try:
                        if mask.startswith(".\\"):  # for windows
                            mask = mask[2:]
                        mask = fits.getdata(mask)
                    except:
                        print(f"[WARNING] No path or file called {mask} found.")
                        continue

                if not self.data.shape == mask.shape:
                    raise ValueError(
                        f"Mask of wrong shape (expected {self.data.shape}, got {mask.shape})"
                    )

                self.data = util.multiply_mask(data=self.data, mask=mask)
                self.mask = util.multiply_mask(data=self.mask, mask=mask)

        else:
            self.mask_from_data()

    def mask_from_data(self):
        """
        A method used to extract a mask from an image file by setting every pixel which is not zero to one.
        """

        mask = self.data.copy()
        mask[mask != 0] = 1
        return mask
