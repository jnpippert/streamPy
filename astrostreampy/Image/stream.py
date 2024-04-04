"""
Module containing the Stream class.

>>> from stream import Stream
"""

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
        masks: list = None,
        interpolation_masks: list = None,
        angle: float = 0,
    ):
        """
        Creates an object containing the image and mask data of the stream.

        Parameters
        ----------
        filename : str, path-like
            Filename of a FITS file. If the file is a multiple extension FITS, the desired
            data-array needs to be the first entry in the HDUL.

        masks : list, optional
            List of filenames of one or multiple masks. If any file is a multiple extension FITS, the desired
            data-array need to be the first entry in the HDUL.
            The masks are used to mask other light sources in the image.

        interpolation_masks : list, optional
            List of filenames of one or multiple interpolation masks. If any file is a multiple extension FITS,
            the desired data-array need to be the first entry in the HDUL.
            The masks are used to mask light sources which lie in or overlap with the stream.
            Masked pixels from the interpolation masks are later interpolated.

        save_mask : bool, optional
            If ``True`` the masks of the parsed mask-lists are combined to one and saved as a fits file respectively.
            By default this parameter is set to ``True``.

        angle : float, optional
            The approximate angle of the stream. This is used to increase the quality of the interpolation process,
            by interpolating along the stream with respect to the given angle.
        """

        if masks is None:
            masks = []

        if interpolation_masks is None:
            interpolation_masks = []

        filename = filename.split(".\\")[-1]  # for windows
        self.filename = filename
        self.data, self.header = fits.getdata(filename, header=True)
        self.original_data = self.data.copy()

        self._angle = angle
        self._mask_lists = [interpolation_masks, masks]

        self.interpolation_mask = np.ones(self.data.shape)
        self.mask = np.ones(self.data.shape)

    def _masking(self, mask_list: list) -> np.ndarray:
        """
        Iterates through a list of filenames, creates a cummulative mask
        and multiplies this mask onto the image data.

        This is a private method and should not be used outside this class.

        Parameters
        ----------
        mask_list : list
            List of filenames.

        Returns
        -------
        super_mask : np.ndarray
            All masks from the list combined to one.
        """
        super_mask = np.ones(self.data.shape)

        for mask in mask_list:
            if not isinstance(mask, str):
                print(f"WARNING: Mask skipped. Mask name of non-str type {type(mask)}.")
                continue

            mask = mask.removeprefix(".\\")

            try:
                mask_data = fits.getdata(mask)
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"No file named {mask}.") from exc

            if not self.data.shape == mask_data.shape:
                raise ValueError(
                    f"Mask of wrong shape (expected {self.data.shape}, got {mask.shape})."
                )
            super_mask *= mask_data
        self.data *= mask_data
        return super_mask

    def mask_from_data(self):
        """
        A method used to extract a mask from an image file by setting every pixel which is not zero to one.
        """

        mask = self.data.copy()
        mask[mask != 0] = 1
        self.mask = mask

    def apply_masks(self):
        """
        Applies all masks onto ``data``, either as source masks or to interpolate zero pixels.
        """

        for i, item in enumerate(self._mask_lists):
            if not isinstance(item, list):
                raise TypeError(f"cannot iterate through non-list type {type(item)}")

            if len(item) > 0:
                tmp_mask = self._masking(item)

                if i == 0:
                    self.interpolation_mask = tmp_mask
                    self.data = util.interpolate_zero_pixels(
                        data=self.data, theta=self._angle
                    )
                    continue

                self.mask = tmp_mask

        if np.prod(self.mask) == 1:
            self.mask_from_data()
