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

# Walkthrough
First import all necessary classes and methods and define the files as variables.
```
from astrostreampy.Image.stream import Stream
from astrostreampy.Image.point import Point
from astrostreampy.BuilModel.autobuild import Model
from astrostreampy.BuildModel.modify import Modifier
from astrostreampy.BuildModel.aperture import fwhm_mask_from_paramtab

image = "image.fits"
mask = "mask.fits"
intmask = "interpolationmask.fits"
```
Start by applying the masks using the ```Stream``` class. Note that the masks are parsed as a list. This allows for multiple masks of the same type to apply simultaneously.
```
stream = Stream(image,[mask],[intmask])
stream.apply_masks()
```
Then the initial box position and dimensions can be set with the ```Point``` class. It opens a figure where the point can be set with left mouse click and the box dimensions are chosen with the sliders on the left. 
When satisfied close the plot by closing the window. ```stream.data()``` is the masked image.
```
init_box = Point(stream.data)
```
The modeling is setup and started with the ```Model``` class. The example presents its shortest and simplest form.
```
model = Model(stream.original_data, stream.data, stream.header, 
                  init_box.x, init_box.y, init_box.width, init_box.height, output="model")
model.build() # for further access get full model with .data
model.show() # for quality checks
```
If ```model.show()``` reveals that the algorithm went beyond the stream call the ````Modifier``` class to cut those regions off. A window opens displaying the image, model and residual. Type in the terminal the lower and upper indices sperated by "," and press *ENTER*. The model and residual changes based on the input. Repeat it as often as desired. When finished leave the line empty and press *ENTER* again. It saves the modified files with prefix "mod_".
```
Modifier("model_multifits.fits","model_paramtab.fits")
```
If you are interested in photometric measurements use
``` 
aperture = fwhm_mask_from_paramtab("mod_model_multifits.fits","mod_model_paramtab.fits"
```
to create an aperture mask, which is a ```numpy.ndarray```.

