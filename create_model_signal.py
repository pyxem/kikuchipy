# -*- coding: utf-8 -*-
#
# Create a stack of denoised EBSPs using a limited set of decomposition
# components to make a model of the initial data, omitting components that
# ideally contain noise.
#
# Created by Håkon W. Ånes (hakon.w.anes@ntnu.no)
# 2018-11-25
#

import hyperspy.api as hs
import os
import argparse
import skimage as sk
import time


hs.preferences.General.nb_progressbar = False  # Use tqdm and not Jupyter
hs.preferences.General.parallel = True  # Use all the CPUS!

# Parse input parameters
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('file', help='Full path of input file')
parser.add_argument('components', help='Number of components to use')
parser.add_argument('--subfix', dest='subfix', default=None,
                    help='Subfix of .npz file with decomposition results')
parser.add_argument('--lazy', dest='lazy', default=False, action='store_true',
                    help='Read/write lazily')
parser.add_argument('--compute', dest='compute', default=False,
                    action='store_true',
                    help='Write signal to memory before rescaling intensities')
args = parser.parse_args()

file = args.file
subfix = args.subfix
components = int(args.components)
lazy = args.lazy

# Set data directory, filename and file extension
datadir, fname = os.path.split(file)
fname, ext = os.path.splitext(fname)

# Read data from file
print('* Read data from file (or not if lazy)')
s = hs.load(file, lazy=lazy)

# Read learning results if they are not read with the signal
if s.learning_results.decomposition_algorithm is None:
    print('* Read learning results from file')
    s.learning_results.load(os.path.join(datadir,
                                         fname + str(subfix) + '.npz'))

# Create new signal from decomposition components
print('* Create new signal from decomposition components')
sc = s.get_decomposition_model(components)
sc.change_dtype('float16')  # Change if higher precision is needed

# Rescale intensities if decomposition algorithm is PCA (IncrementalPCA). If
# the algorithm is SVD or NMF, HyperSpy fixes the intensities when calling
# s.get_decomposition_model().
if s.learning_results.decomposition_algorithm == 'PCA':
    # Create new minimum and maximum intensities, keeping the ratios
    print('* Create new minimum and maximum intensities before scaling')
    # First, get new maximums and minimums after background subtraction
    scmin = sc.min(sc.axes_manager.signal_axes)
    scmax = sc.max(sc.axes_manager.signal_axes)

    # Set lowest intensity to zero
    int_min = scmin.data.min()
    scmin = scmin - int_min
    scmax = scmax - int_min

    # Get scaling factor and scale intensities
    scale = 255 / scmax.data.max()
    scmin = scmin * scale
    scmax = scmax * scale

    # Convert to uint8 and write to memory if lazy
    scmin.change_dtype('uint8')
    scmax.change_dtype('uint8')
    if lazy:
        print('* Write minimum and maximum intensity signals to memory')
        scmin.compute()
        scmax.compute()

    def rescale_pattern(pattern, scmin, scmax):
        return sk.exposure.rescale_intensity(pattern, out_range=(scmin, scmax))

    if args.compute:
        print('* Writing signal to memory before rescaling intensities')
        sc.compute()

    # Rescale intensities
    print('* Rescale patterns (timing)')
    start = time.time()
    sc.map(rescale_pattern, parallel=True, ragged=False, scmin=scmin,
           scmax=scmax)
    print('* Time: %.2f s' % (time.time() - start))

# Revert data type
print('* Revert data type')
sc.change_dtype('uint8')

# Write data to file
print('* Write data to file')
sc.save(os.path.join(datadir, fname + '_model' + str(components) + ext))
