import argparse
import sys

from streampy import argutils
from streampy.BuildModel.autobuild import Model
from streampy.Image.point import Point
from streampy.Image.stream import Stream

parser = argparse.ArgumentParser(
    description="wrapper for the streampy stellar stream modelling package. J.N. Pippert 2023"
)
parser.add_argument(
    "infile", nargs="?", type=argparse.FileType("r"), help="filename of the image"
)
parser.add_argument(
    "-mf", "--maskfiles", help="single or multiple masks seperated by comma", type=str
)
parser.add_argument(
    "-imf",
    "--intpolmaskfiles",
    help="single or multiple interpolation masks seperated by comma",
    type=str,
)
parser.add_argument(
    "-s", "--steps", help="number of steps seperated by comma", default="9999,9999"
)
parser.add_argument("-lp", "--liveplot", default=False, action="store_true")
parser.add_argument("-h2", "--h2param", default=False, action="store_true")
parser.add_argument("-skew", "--skewparam", default=False, action="store_true")
parser.add_argument("-h4", "--h4param", default=False, action="store_true")
parser.add_argument(
    "-sn",
    "--signalnoise",
    help="defines the signal to noise thershold at which the algortihm stops",
    default=5.0,
    type=float,
)
parser.add_argument(
    "-ia", "--initangle", help="initial angle guess", default=0, type=float
)
parser.add_argument("-o", "--output", help="output name", type=str, default="streampy")
parser.add_argument("-bg", "--fixbackground", type=float, default=None)
parser.add_argument("-hw", "--varyhw", default=False, action="store_true")
parser.parse_args(args=None if sys.argv[1:] else ["--help"])
args = parser.parse_args()


# handle args
infile = argutils.handle_infile(arg=args.infile)
s1, s2 = argutils.handle_steps(arg=args.steps)
mask_list = argutils.handle_fileargs(arg=args.maskfiles)
intmask_list = argutils.handle_fileargs(arg=args.intpolmaskfiles)

# intialize the data
stream = Stream(
    filename=infile,
    masks=mask_list,
    interpolation_masks=intmask_list,
    angle=args.initangle,
)
init_point = Point(stream.data)


stream_model = Model(
    original_data=stream.original_data,
    masked_data=stream.data,
    header=stream.header,
    sourcemask=stream.mask,
    init_x=init_point.x,
    init_y=init_point.y,
    init_width=init_point.width,
    init_height=init_point.height,
    init_angle=args.initangle,
    h2=args.h2param,
    skew=args.skewparam,
    h4=args.h4param,
    sn_threshold=args.signalnoise,
    fix_bg=args.fixbackground,
    vary_box_dim=args.varyhw,
    output=args.output,
)

stream_model.build(steps=(s1, s2), liveplot=args.liveplot)
stream_model.show(output=args.output)
