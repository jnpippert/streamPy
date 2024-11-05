"""
Example Wrapper to model stellar stream from command line.
"""

import argparse
import sys

from astropy.io import fits

from astrostreampy import argutils
from astrostreampy.BuildModel.aperture import fwhm_mask_from_paramtab
from astrostreampy.BuildModel.autobuild import Model
from astrostreampy.BuildModel.modify import Modifier
from astrostreampy.Image.inspect import Slice
from astrostreampy.Image.measure import StreamProperties
from astrostreampy.Image.point import InitBox
from astrostreampy.Image.stream import Stream

parser = argparse.ArgumentParser(
    description="Wrapper for the streampy stellar stream modelling package. J.N. Pippert 2023"
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
parser.add_argument("-ef", "--errorfile", type=str)
parser.add_argument(
    "-s", "--steps", help="number of steps seperated by comma", default="9999,9999"
)
parser.add_argument("-z", "--redshift", type=float, default=0)
parser.add_argument("-h2", "--h2param", default=False, action="store_true")
parser.add_argument("-skew", "--skewparam", default=False, action="store_true")
parser.add_argument("-h4", "--h4param", default=False, action="store_true")
parser.add_argument(
    "-sn",
    "--signaltonoise",
    help="defines the signal to noise thershold at which the algortihm stops",
    default=5.0,
    type=float,
)
parser.add_argument(
    "-ia", "--initangle", help="initial angle guess", default=0, type=float
)
parser.add_argument("-o", "--output", help="output name", type=str, default="streampy")
parser.add_argument("-bg", "--fixbackground", type=float, default=None)
parser.add_argument("--varyhw", default=False, action="store_true")
parser.parse_args(args=None if sys.argv[1:] else ["--help"])
args = parser.parse_args()


def model():
    """
    TODO
    """
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
    stream.apply_masks()
    init_point = InitBox(data=stream.data)

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
        tail=init_point.tail,
        head=init_point.head,
        h2=args.h2param,
        skew=args.skewparam,
        h4=args.h4param,
        sn_threshold=args.signaltonoise,
        output=args.output,
    )

    stream_model.build(steps=(s1, s2))
    stream_model.show(output=args.output)

    Modifier(
        multifits_file=f"{args.output}_multifits.fits",
        param_file=f"{args.output}_paramtab.fits",
    )

    # the '_' are the border mask and a 1d center mask (the peaks of the Gaussians)
    aperture_mask, _, _ = fwhm_mask_from_paramtab(
        f"mod_{args.output}_paramtab.fits",
        f"mod_{args.output}_multifits.fits",
        verbose=1,
    )

    fits.append(f"mod_{args.output}_multifits.fits", aperture_mask)
    Slice(f"mod_{args.output}_multifits.fits", f"mod_{args.output}_paramtab.fits")

    # Measure the Stream
    s = StreamProperties(
        f"mod_{args.output}_multifits.fits",
        f"mod_{args.output}_paramtab.fits",
        maskfiles=mask_list + intmask_list,
        redshift=args.redshift,
    )
    s.measure(errorfile=args.errorfile)
    s.writeto(f"{args.output}", overwrite=True)


if __name__ == "__main__":
    model()
