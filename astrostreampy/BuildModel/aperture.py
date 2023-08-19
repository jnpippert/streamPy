from math import ceil

import numpy as np
from astropy.io import fits

from .. import utilities


def fwhm_mask_from_paramtab():
    raise NotImplementedError


def std_mask_from_paramtab(
    parameter_file: str, multifits_file: str, k: int = 3, smoothing: int = 10
) -> np.ndarray:
    """
    Creates an aperture mask from the parameter FITS table.

    Parameters
    ----------
    parameter_file : str or path-like
        Name of the parameter FITS table.

    multifits_file : str or path-like
        Name of the mulit FITS file.

    k : int, optional
        Kappa value for mask creation. How much is set to zero k * sigma from the center.
        Default is 3.

    smoothing : int, optional
        Smoothing factor of the standard deviations. Default is 25.

    Raises
    ------
    KeyErrors if some of the keys are not found in the FITS header extension.

    Returns
    -------
    mask : np.ndarray
        Data array of the aperture mask.

    """
    table = fits.getdata(parameter_file, ext=1)
    data, header = fits.getdata(multifits_file, ext=0, header=True)
    mask = np.zeros(data.shape)

    header["PXSCALE"] = 0.2

    filter_band = header["FILTER"]  # Filter band the image was taken in.
    pxscale = header["PXSCALE"]  # Pixelscale of the image in arcseconds/pixel.
    psf = header["PSF"]  # Mean FWHM in arcseconds of all image PSF's.
    seeing = psf / pxscale / (2 * np.sqrt(2 * np.log(2)))

    print(f"using psf fhwm: {psf} [arcsec]")
    print(f"using pxscale: {pxscale} [arcsec / pixel]")
    print(f"using seeing: {seeing} [pixel]")

    sigmas = table[f"sigma_{filter_band}"]
    for orientation in ["horizontal", "vertical"]:
        tmp_mask = np.zeros(mask.shape)
        for i, row in enumerate(table):
            (
                x,
                y,
                w,
                h,
                angle,
                _,
                sigma,
                _,
                norm,
                _,
                offset,
                _,
                x0,
                _,
                y0,
                _,
                h2,
                _,
                skew,
                _,
                h4,
                _,
            ) = row

            model = utilities.fit_func_2D(
                None,
                angle,
                sigma,
                norm,
                offset,
                x0,
                y0,
                h2,
                skew,
                h4,
                w,
                h,
                seeing=seeing,
            )
            clean_model = utilities.fit_func_2D(
                None, angle, sigma, norm, offset, x0, y0, 0, 0, 0, w, h, seeing=seeing
            )

            model = model.reshape((2 * h + 1, 2 * w + 1))
            clean_model = clean_model.reshape((2 * h + 1, 2 * w + 1))

            if orientation == "vertical":
                if w <= h:
                    model_slice = model[:, w]
                    clean_model_slice = clean_model[:, w]
                else:
                    continue

            if orientation == "horizontal":
                if w > h:
                    model_slice = model[h]
                    clean_model_slice = clean_model[h]
                else:
                    continue

            center_id = np.argmax(clean_model_slice)

            if i < smoothing:
                sigma = np.mean(sigmas[: i + smoothing + 1])
            else:
                sigma = np.mean(sigmas[i - smoothing : i + smoothing + 1])

            right = ceil(center_id - k * sigma)
            left = int(center_id + k * sigma)

            model_slice[:right] = 0
            model_slice[left + 1 :] = 0
            model_slice[right : left + 1] = 1

            if w > h:
                tmp_mask[y, x - w : x + w + 1] += model_slice
            else:
                tmp_mask[y - h : y + h + 1, x] += model_slice

        mask = np.max([mask, tmp_mask], axis=0)

    mask[mask > 1] = 1
    mask[mask != 1] = 0
    return mask
