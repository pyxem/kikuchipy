# -*- coding: utf-8 -*-
# Copyright 2019-2020 The kikuchipy developers
#
# This file is part of kikuchipy.
#
# kikuchipy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# kikuchipy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with kikuchipy. If not, see <http://www.gnu.org/licenses/>.

import os

import matplotlib

matplotlib.rcParams["backend"] = "qt5agg"
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fftshift

import kikuchipy as kp

data = "/home/hakon/kode/kikuchipy/kikuchipy/data/kikuchipy/patterns.h5"
data2 = "/home/hakon/phd/data/ni/2020/1/nordif/Pattern.dat"
outdir = "/home/hakon/kode/kikuchipy/doc/_static/image"
featdir = os.path.join(outdir, "feature_maps")

datadir, fname = os.path.split(data)
fname, ext = os.path.splitext(fname)
s = kp.load(data, lazy=False)
s2 = kp.load(data2, lazy=False)

# Image quality
s3 = s2.deepcopy()
s3.remove_static_background()
s3.remove_dynamic_background()
# Image quality map
iq = s3.get_image_quality()
x, y = 157, 80
iq_perc = np.percentile(iq, q=(0, 99.8))
plt.figure()
plt.imshow(iq, vmin=iq_perc[0], vmax=iq_perc[1], cmap="gray")
plt.colorbar(label="Image quality")
plt.savefig(
    os.path.join(featdir, "iq.png"), bbox_inches="tight", pad_inches=0,
)
# Pattern
p = s3.inav[x, y].data
plt.figure()
plt.imshow(p, cmap="gray")
plt.colorbar()
plt.savefig(
    os.path.join(featdir, "image_quality_pattern.png"),
    bbox_inches="tight",
    pad_inches=0,
)
# Pattern FFT
p_fft = kp.pattern.fft(p, shift=True)
p_spec = kp.pattern.fft_spectrum(p_fft)
plt.figure()
plt.imshow(np.log(p_spec), cmap="gray")
plt.colorbar()
plt.savefig(
    os.path.join(featdir, "fft_spectrum.png"),
    bbox_inches="tight",
    pad_inches=0,
)
# Frequency vectors
q = kp.pattern.fft_frequency_vectors(shape=p.shape)
plt.figure()
plt.imshow(fftshift(q), cmap="gray")
plt.colorbar()
plt.savefig(
    os.path.join(featdir, "fft_frequency_vectors.png"),
    bbox_inches="tight",
    pad_inches=0,
)
