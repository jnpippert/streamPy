"""
Provides two methods to create stream apertures.

Example
-------
>>> from aperture import fwhm_mask_from_paramtab
>>> from aperture import effective_mask_from_paramtab
"""

import numpy as np
from astropy.io import fits

from .utilities import get_distances, get_effective_distances


def fwhm_mask_from_paramtab(
    parameter_file: str,
    multifits_file: str,
    k: int = 1,
    smoothing: int = 10,
    verbose: int = 0,
) -> np.ndarray:
    """
    Creates the aperture mask from the fit parameter table.
    It iterates over every row of the table, i.e. slice of the stream,
    and sets every pixel farther away from the mean than `k*sigma` to zero.
    The `sigma` is caclulated a a running mean at
    each iteration step.

    Parameters
    ----------
    param_file : str
        A ``_paramtab.fits`` file created by the ``streampy.BuildModel.autobuild.build`` method.

    multifits_file : str
        A ``_multifits.fits`` file created by the ``streampy.BuildModel.autobuild.build`` method.

    k : int
        Kappa value.

    out : str, optional
        The name of the output files. If not None, then the mask is saved as a FITS file.

    verbose : int, optional
        Toggles some console outputs.

    Returns
    -------
    mask : `np.ndarray`
    """

    if verbose == 1:
        print(f"creating FHWM mask from parameter table '{parameter_file}' ...")

    table = fits.getdata(parameter_file, ext=1)

    model = fits.getdata(multifits_file, ext=4)
    mask = np.zeros(model.shape)
    border_mask = np.zeros(model.shape)
    center_mask = np.zeros(model.shape)
    res = get_distances(parameter_file, multifits_file)
    center_ids, left_dists, right_dists = res
    for i, row in enumerate(table):
        (x, y, w, h, *_) = row

        if w > h:
            model_slice = model.copy()[y, x - w : x + w + 1]
        else:
            model_slice = model.copy()[y - h : y + h + 1, x]

        if smoothing == 0:
            left_dist = left_dists[i]
            right_dist = right_dists[i]
        else:
            if i < smoothing:
                left_dist = np.mean(left_dists[: i + smoothing + 1])
                right_dist = np.mean(right_dists[: i + smoothing + 1])
            else:
                left_dist = np.mean(left_dists[i - smoothing : i + smoothing + 1])
                right_dist = np.mean(right_dists[i - smoothing : i + smoothing + 1])

        model_x = np.arange(0, len(model_slice), 1)
        left = np.argmin(np.abs(model_x - (center_ids[i] - k * (left_dist))))
        right = np.argmin(np.abs(model_x - (center_ids[i] + k * (right_dist))))

        model_slice[:left] = 0
        model_slice[right + 1 :] = 0
        model_slice[left : right + 1] = 1

        if w > h:
            mask[y, x - w : x + w + 1] += model_slice
        else:
            mask[y - h : y + h + 1, x] += model_slice

        # border
        mask_border_ids = np.where(model_slice == 1)[0]
        model_slice[mask_border_ids[0] + 1 : mask_border_ids[-1]] = 0
        if w > h:
            border_mask[y, x - w : x + w + 1] += model_slice

        else:
            border_mask[y - h : y + h + 1, x] += model_slice

        # center
        model_slice *= 0
        model_slice[center_ids[i] - 1 : center_ids[i] + 2] = 1
        if w > h:
            center_mask[y, x - w : x + w + 1] += model_slice

        else:
            center_mask[y - h : y + h + 1, x] += model_slice
    mask[mask != 1] = 0
    return mask, border_mask, center_mask


def effective_mask_from_paramtab(
    parameter_file: str,
    multifits_file: str,
    smoothing: int = 10,
    verbose: int = 0,
) -> np.ndarray:
    if verbose == 1:
        print(f"creating EFF mask from parameter table '{parameter_file}' ...")

    table = fits.getdata(parameter_file, ext=1)

    model = fits.getdata(multifits_file, ext=4)
    mask = np.zeros(model.shape)
    border_mask = np.zeros(model.shape)
    res = get_effective_distances(parameter_file, multifits_file)
    center_ids, dists = res

    for i, row in enumerate(table):
        (x, y, w, h, *_) = row

        if w > h:
            model_slice = model.copy()[y, x - w : x + w + 1]
        else:
            model_slice = model.copy()[y - h : y + h + 1, x]

        if smoothing == 0:
            dist = dists[i]
        else:
            if i < smoothing:
                dist = np.mean(dists[: i + smoothing + 1])
            else:
                dist = np.mean(dists[i - smoothing : i + smoothing + 1])
        center_id = center_ids[i]

        model_x = np.arange(0, len(model_slice), 1)
        left = np.argmin(np.abs(model_x - (center_id - dist)))
        right = np.argmin(np.abs(model_x - (center_id + dist)))

        model_slice[:left] = 0
        model_slice[right + 1 :] = 0
        model_slice[left : right + 1] = 1

        if w > h:
            mask[y, x - w : x + w + 1] += model_slice

        else:
            mask[y - h : y + h + 1, x] += model_slice

        # border
        mask_border_ids = np.where(model_slice == 1)[0]
        model_slice[mask_border_ids[0] + 1 : mask_border_ids[-1]] = 0
        if w > h:
            border_mask[y, x - w : x + w + 1] += model_slice

        else:
            border_mask[y - h : y + h + 1, x] += model_slice

    mask[mask != 1] = 0
    border_mask[border_mask != 1] = 0
    return mask, border_mask
