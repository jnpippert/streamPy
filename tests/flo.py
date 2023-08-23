from astrostreampy.BuildModel.autobuild import Model
from astrostreampy.Image.point import Point
from astrostreampy.Image.stream import Stream

file_name = "TS0120+1930.fits"
mask_name = "sm.TS0120+1930.fits"
inter_name = "im.TS0120+1930.fits"

# Create a stream object
stream = Stream(file_name, [mask_name], [inter_name])

stream()
stream.header["PXSCALE"] = 0.2

point = Point(data=stream.data)

# Create a model object
model = Model(
    original_data=stream.original_data,
    masked_data=stream.data,
    header=stream.header,
    sourcemask=stream.mask,
    init_x=point.x,
    init_y=point.y,
    init_width=point.width,
    init_height=point.height,
)

model.build()
model.show()
