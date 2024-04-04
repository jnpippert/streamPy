# streamPy

To create a stream model you need the following files:
  - the **IMAGE** file
  - a **MASK** of all nearby, and overlapping sources
  - if other stars or galaxies contaminate the stream's central region, create an additional **INTERPOLATION MASK** for sources to interpolate over

All have to be the same size, otherwise an error is raised. RGB *.TIF*, *.jpg* or *.png* does not work, they need to be in gray scale *.fits* format. 
The **IMAGE** header must contain the following keys:
  - 'FILTER'  : char, the band in which the image was taken.
  - 'PSF'     : float, the mean FWHM of the sources across the image in arc seconds. If there is no interest in the true intrinsic shape parameters of the stream set it to **1**.
  - 'PXSCALE' : float, the pixel scale of the image in arc seconds/pixel.
  - 'ZP'      : float, the photometric zero point.
  - 'ZSTREAM' : float, the redshift of the Stream

# Walkthrough
First import all necessary classes and methods and define the files as variables.
```
from astrostreampy.Image.stream import Stream
from astrostreampy.Image.point import Point
from astrostreampy.BuilModel.autobuild import Model
from astrostreampy.BuildModel.modify import Modifier
from astrostreampy.BuildModel.aperture import fwhm_mask_from_paramtab
from astrostreampy.Image.measure import StreamProperties

image = "image.fits"
mask = "mask.fits"
intmask = "interpolationmask.fits"
```
Start by applying the masks using the ```Stream``` class. Note that the masks are parsed as a list. This allows for multiple masks of the same type to be applied simultaneously.
```
stream = Stream(image,[mask],[intmask])
stream.apply_masks()
```
Then the initial box position and dimensions can be set with the ```Point``` class. It opens a figure where the point can be set with a left mouse click and the box dimensions are chosen with the sliders on the left. 
Optional, one can set both endpoints of the stream by switching the clicking mode from the center (w) to head (d) to tail(a) mode.
When satisfied close the plot by closing the window. ```stream.data()``` is the masked image.
```
init_box = InitBox(data=stream.data)
```
The modeling is set up and started with the ```Model``` class. It will output two files with the prefix parsed by the "output" argument and suffixes of ```_multifits.fits``` and ```_paramtab.fits```.
```
model = Model(
            original_data=stream.original_data,
            masked_data=stream.data,
            header=stream.header,
            sourcemask=stream.mask,
            init_x=init_box.x,
            init_y=init_box.y,
            init_width=init_box.width,
            init_height=init_box.height,
            init_angle=0,
            tail=init_box.tail,
            head=init_box.head,
            h2=False,
            skew=False,
            h4=False,
            output="streampy",
        )
        model.show()
```
If ```model.show()``` reveals that the algorithm went beyond the stream call the ```Modifier``` class to cut those regions off. A window opens displaying the image, model, and residual. Type in the terminal the lower and upper indices separated by "," and press *ENTER*. The model and residual changes are based on the input. Repeat it as often as desired. When finished leave the line empty and press *ENTER* again. It saves the modified files with the prefix ```mod_```. With given ```upper``` and ```lower``` arguments the modification is done automatically.
```
Modifier("streampy_multifits.fits", "streampy_paramtab.fits", upper=0, lower=0)
```
If you are interested in photometric measurements use
``` 
aperture_mask, _, _ = fwhm_mask_from_paramtab("mod_streampy_paramtab.fits", "mod_streampy_multifits.fits")
fits.append("mod_streampy_multifits.fits", aperture_mask)
```
to create an aperture mask, which is a ```numpy.ndarray``` and append it to the multifits. This is also needed when inspecting the model quality with ```Slice```.
The other two returns ```_``` are a central and outline 1D mask.

Lastly, to measure photometric properties use the ```StreamProperties``` class.
```
s = StreamProperties(
        "mod_conv2dtest_multifits.fits",
        "mod_conv2dtest_paramtab.fits",
        maskfiles=[mask_name, inter_name],
        redshift=0.0329069,
        zeropoint=30.1151,
        pixelscale=0.2,
    )
s.measure(errorfile="e.TS0120+1930_g.fits") # If no error file is given a global error of 0 is assumed
print(s)
s.writeto("streampy.txt", overwrite=True) # Per default adds an '_measurements' suffix.
# For comparison measure again on the model
s.measure(measure_model=True, bg=0)
print(s)
s.writeto("streampy_model.txt", overwrite=True)
```
