# -*- coding: utf-8 -*-
#
# Subtract two types of backgrounds from each pattern:
#   1) an average static background image 
#   2) a dynamical Gaussian blur image calculated for each pattern with a certain sigma
#Save to a new hdf5-file.

import hyperspy.api as hs
import argparse
import os
import time
import skimage as sk
import scipy.ndimage as scn

start_tot = time.time()

hs.preferences.General.nb_progressbar = False
hs.preferences.General.parallel = True

# Parse input parameters
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('file', help='Full path of original file')
parser.add_argument('--lazy', dest='lazy', default=False, action='store_true',
                    help='Whether to read/write lazy or not. Do not recommend lazy, since it takes a long time!')
parser.add_argument('--bg_img', help='Background image to subtract.')
parser.add_argument('--sigma', dest='sigma', default=8/160*240,
                    help='Sigma of the Gaussian blur for the dynamical background.')
# Parse
arguments = parser.parse_args()
file = arguments.file
lazy = arguments.lazy
bg_img = arguments.bg_img
sigma = arguments.sigma

# Set data directory, filename and file extension
datadir, fname = os.path.split(file)
fname, ext = os.path.splitext(fname)

# Read data
print('* Read data from file')
s = hs.load(file, lazy=lazy)

# Read or create static background pattern by averaging the signal over the whole dataset
print('* Read background image from file')
if bg_img is None:
    try: bg_img = hs.load(os.path.join(datadir,'Background acquisition pattern.bmp'),lazy=lazy)
    except FileNotFoundError and ValueError:
        bg_img = s.sum()/s.axes_manager.navigation_size
else:
    bg_img = hs.load(bg_img, lazy=lazy)
    
# Change data types (increase bit depth) to avoid negative intensites when
# subtracting background patterns
bg_img.change_dtype('int16')
s.change_dtype('int16')

def rescale_pattern(pattern, smin, smax):
    return sk.exposure.rescale_intensity(pattern, out_range=(smin, smax))

# Subtract by background pattern
print('* Subtract background pattern')
s = s - bg_img

# Create new minimum and maximum intensities, keeping the ratios
print('* Create new minimum and maximum intensities before scaling')
# First, get new maximums and minimums after background subtraction
smin, smax = s.min(s.axes_manager.signal_axes), s.max(s.axes_manager.signal_axes)
# Set lowest intensity to zero
int_min = smin.data.min()
smin, smax = smin - int_min, smax - int_min
# Get scaling factor and scale intensities
scale = 255/smax.data.max()
smin, smax = smin * scale, smax * scale
if lazy:
    smin.compute()
    smax.compute()

print('* Rescale patterns (timing)')
start = time.time()
s.map(rescale_pattern, parallel=True, ragged=False, smin=smin, smax=smax)
print('* Time: %.4f min',(time.time() - start)/60)

# Create dynamic background
print('* Create dynamic background')
s_blur = s.map(scn.gaussian_filter, inplace=False, ragged=False, sigma=sigma)

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
s.save(os.path.join(datadir, fname + '_bkg_dyn' + str(int(sigma)) + ext))
print('* Total Time: %.2f min' % ((time.time() - start_tot)/60))