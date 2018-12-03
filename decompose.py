# -*- coding: utf-8 -*-
#
# Perform statistical decomposition of 4D dataset.
#
# Created by Håkon W. Ånes (hakon.w.anes@ntnu.no)
# 2018-11-23
#

import hyperspy.api as hs
import argparse
import os
import time


hs.preferences.General.nb_progressbar = False
hs.preferences.General.parallel = True

# Parse input parameters
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('file', help='Full path of original file')
parser.add_argument('--lazy', dest='lazy', default=False, action='store_true',
                    help='Whether to read/write lazy or not')
parser.add_argument('--algorithm', dest='algorithm', default='svd',
                    help='Algorithm to use')
parser.add_argument('--components', dest='components', default=None,
                    help='Number of components (output dimensions) to use')
parser.add_argument('--save-to-npz', dest='save_to_npz', default=False,
                    action='store_true',
                    help='Write learning results to a separate .npz file')
args = parser.parse_args()

# Set data directory, filename and file extension
datadir, fname = os.path.split(args.file)
fname, ext = os.path.splitext(fname)

algorithm = str(args.algorithm)
components = args.components

if components is not None:
    components = int(components)

# Read data
print('* Read data from file')
s = hs.load(args.file, lazy=args.lazy)

# Change datatype to 16-bit float precision. Increase to 32- or 64-bit
# precision if 16-bit is not enough.
s.change_dtype('float16')

print('* Perform decomposition (timing)')
start = time.time()
s.decomposition(algorithm=algorithm, output_dimension=components)
print('* Time: %.2f s' % (time.time() - start))

# Write data to file
if args.save_to_npz:
    print('* Write learning results to external file (.npz)')
    s.learning_results.save(os.path.join(datadir, fname + '_' + algorithm))
else:
    print('* Write learning results to a new signal file (.hspy)')
    s.save(os.path.join(datadir, fname), overwrite=True)
