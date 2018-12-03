# -*- coding: utf-8 -*-
#
# Subtract dynamic (gaussian blurred) background from patterns
#
# Created by Håkon W. Ånes (hakon.w.anes@ntnu.no)
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
arguments = parser.parse_args()

file = arguments.file
lazy = arguments.lazy
sigma = int(arguments.sigma)

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
s = s - s_blur

# Don't care about relative intensities since subtracted pattern is different
# for all patterns. Therefore rescale intensities to full uint8 range [0, 255].
print('* Rescale patterns (timing)')
start = time.time()
s.map(sk.exposure.rescale_intensity, ragged=False, out_range=(0, 255))
s.change_dtype('uint8')
print('* Time: %.2f s' % (time.time() - start))

# Write patterns to file
print('* Write data to file')
s.save(os.path.join(datadir, fname + '_dyn' + str(sigma) + ext))