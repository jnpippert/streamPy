# streamPy

To create a stream model you need the following files:
  - the **IMAGE** file
  - a **MASK** of all neabry, and overlapping sources
  - if other stars or galaxies contaminate the stream's central region, create an additional **INTERPOLATION MASK** for sources to interpolate over

All have to be the same size, otherwise an error is raised. RGB *.TIF*, *.jpg* or *.png* does not work, they need to be in gray scale *.fits* format. 
The **IMAGE** header must contain the following keys:
  - 'FILTER'  : char, the band in which the image was taken.
  - 'PSF'     : float, the mean FWHM of the sources across the image in arc seconds. If there is no interest in the true intrinsic shape parameters of the stream set it to **1**.
  - 'PXSCALE' : float, the pixel scale of the image in arc seconds/pixel.
  - 'ZP'      : float, the photometric zero point.

Start by applying the masks using the ```Stream``` class.
```
from astrostreampy.Image.stream import Stream

image = "image.fits"
mask = "mask.fits"
intmask = "interpolationmask.fits"

stream = Stream(image,[mask],[intmask])

```

