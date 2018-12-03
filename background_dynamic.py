# -*- coding: utf-8 -*-
#
# Subtract a dynamic (gaussian blurred) background from electron backscatter
# patterns. The background can either be subtracted or divided by. Relative
# intensities are lost since a unique pattern is subtracted from each pattern.
#
# Created by Håkon W. Ånes (hakon.w.anes@ntnu.no)
# 2018-11-07
#

import hyperspy.api as hs
import argparse
import os
import time
import skimage as sk
import scipy.ndimage as scn


hs.preferences.General.nb_progressbar = False  # Use tqdm progressbar
hs.preferences.General.parallel = True

# Parse input parameters
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('file', help='Full path of original file')
parser.add_argument('--lazy', dest='lazy', default=False, action='store_true',
                    help='Whether to read/write lazy or not')
parser.add_argument('--sigma', dest='sigma', default=8,
                    help='Full path of original file')
parser.add_argument('--divide', dest='divide', default=False,
                    action='store_true', help='Divide by static background')
args = parser.parse_args()

file = args.file
lazy = args.lazy
sigma = int(args.sigma)

# Set data directory, filename and file extension
datadir, fname = os.path.split(file)
fname, ext = os.path.splitext(fname)

# Read data
print('* Read data from file')
s = hs.load(file, lazy=lazy)

# Create dynamic background
print('* Create dynamic background')
s_blur = s.map(scn.gaussian_filter, inplace=False, ragged=False, sigma=sigma)

# Change data types (increase bit depth) to avoid negative intensites when
# subtracting background patterns
print('* Change data types to 16-bit')
s.change_dtype('int16')
s_blur.change_dtype('int16')

# Subtract by background pattern
print('* Subtract dynamic pattern')
if args.divide:
    s = s / s_blur
else:
    s = s - s_blur

# We don't care about relative intensities anymore since the subtracted
# pattern is different for all patterns. We therefore rescale intensities to
# full uint8 range [0, 255].
print('* Rescale patterns (timing)')
start = time.time()
s.map(sk.exposure.rescale_intensity, ragged=False, out_range=(0, 255))
s.change_dtype('uint8')
print('* Time: %.2f s' % (time.time() - start))

# Write data to file
print('* Write data to file')
s.save(os.path.join(datadir, fname + '_dyn' + str(sigma) + ext))
