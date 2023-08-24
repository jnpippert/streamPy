from astrostreampy.BuildModel.autobuild import Model
from astrostreampy.Image.point import Point
from astrostreampy.Image.stream import Stream

file_name = "stream.fits"
mask_name = "source_mask.fits"
inter_name = "interpolation_mask.fits"

stream = Stream(file_name, [mask_name], [inter_name])
stream()

point = Point(data=stream.data)

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
