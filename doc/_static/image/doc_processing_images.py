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
from scipy.ndimage import correlate

import kikuchipy as kp

data = "/home/hakon/kode/kikuchipy/kikuchipy/data/kikuchipy/patterns.h5"
data2 = "/home/hakon/phd/data/ni/2020/6/nordif/Pattern.dat"
outdir = "/home/hakon/kode/kikuchipy/doc/_static/image"
procdir = os.path.join(outdir, "pattern_processing")

datadir, fname = os.path.split(data)
fname, ext = os.path.splitext(fname)
s = kp.load(data, lazy=False)

s.metadata.General.title = "patterns"

# Get dynamic background
bg = s.get_dynamic_background()
bg.plot()
bg._plot.signal_plot.figure.savefig(
    os.path.join(procdir, "dynamic_background.png")
)

# Remove static background
s.plot()
s._plot.signal_plot.figure.savefig(os.path.join(procdir, "pattern_raw.png"))
s.remove_static_background(operation="subtract")
s.plot()
s._plot.signal_plot.figure.savefig(os.path.join(procdir, "pattern_static.png"))

# Remove dynamic background
s.remove_dynamic_background(operation="subtract", std=8)
s.plot()
s._plot.signal_plot.figure.savefig(os.path.join(procdir, "pattern_dynamic.png"))

# Change scan and pattern size
s2 = s.isig[10:50, 10:50]
s2.plot()
s2._plot.signal_plot.figure.savefig(
    os.path.join(outdir, "change_scan_pattern_size/pattern_cropped.png")
)

# Average neighbour patterns
s3 = kp.load(data2, lazy=True)
s4 = s3.inav[150:160, 75:85]
s4.remove_static_background()
s5 = s4.deepcopy()
k = kp.filters.Window(window="gaussian", shape=(3, 3), std=1)
fig, _, _ = k.plot(cmap="inferno")
fig.savefig(
    os.path.join(procdir, "window_gaussian_std1.png"),
    bbox_inches="tight",
    pad_inches=0,
    transparent=True,
)
s5.average_neighbour_patterns(k)
x, y = (7, 1)
s4.compute()
s5.compute()
s4.axes_manager.indices = (x, y)
s5.axes_manager.indices = (x, y)
s4.plot()
s4._plot.signal_plot.figure.savefig(
    os.path.join(procdir, "pattern_scan_static.png")
)
s5.plot()
s5._plot.signal_plot.figure.savefig(
    os.path.join(procdir, "pattern_scan_averaged.png")
)

# Circular (5,4) kernel
k = kp.filters.Window("circular", (5, 4))
fig, _, _ = k.plot()
fig.savefig(
    os.path.join(procdir, "window_circular_54.png"),
    bbox_inches="tight",
    pad_inches=0,
    transparent=True,
)

# Close everything
plt.close("all")

# Rescale intensities
s6 = s.deepcopy()
s6.change_dtype(np.uint16)
s6.plot(vmin=0, vmax=1000)
s6._plot.signal_plot.figure.savefig(
    os.path.join(procdir, "rescale_intensities_before.png")
)
s6.rescale_intensity(relative=True)
s6.plot(vmin=0, vmax=65535)
s6._plot.signal_plot.figure.savefig(
    os.path.join(procdir, "rescale_intensities_after.png")
)

# Contrast stretching
s7 = s.deepcopy()
y, x = (0, 0)
s7.axes_manager.indices = (x, y)
s7.plot(vmin=0, vmax=255)
s7._plot.signal_plot.figure.savefig(
    os.path.join(procdir, "contrast_stretching_before.png")
)
s7.rescale_intensity(percentiles=(1.5, 98.5))
s7.plot(vmin=0, vmax=255)
s7._plot.signal_plot.figure.savefig(
    os.path.join(procdir, "contrast_stretching_after.png")
)

# Normalize intensity
s8 = s.deepcopy()
plt.figure()
plt.hist(s8.data.ravel(), bins=255)
plt.savefig(os.path.join(procdir, "normalize_intensity_before.png"))
s8.normalize_intensity(dtype_out=np.float32)
plt.figure()
plt.hist(s8.data.ravel(), bins=255)
plt.savefig(os.path.join(procdir, "normalize_intensity_after.png"))

# FFT filtering
s9 = s.deepcopy()
pattern_shape = s9.axes_manager.signal_shape[::-1]
# Gaussian high/lowpass
w_low = kp.filters.Window(
    "lowpass", shape=pattern_shape, cutoff=22, cutoff_width=10,
)
w_high = kp.filters.Window(
    "highpass", shape=pattern_shape, cutoff=3, cutoff_width=2,
)
w = w_low * w_high
plt.figure()
plt.imshow(w, cmap="gray")
plt.colorbar()
plt.savefig(
    os.path.join(procdir, "fft_filter_highlowpass2d.png"),
    bbox_inches="tight",
    pad_inches=0,
)
plt.figure()
plt.plot(w[pattern_shape[0] // 2, :])
plt.savefig(
    os.path.join(procdir, "fft_filter_highlowpass1d.png"),
    bbox_inches="tight",
    pad_inches=0,
)

s9.fft_filter(transfer_function=w, function_domain="frequency", shift=True)
s9.plot()
s9._plot.signal_plot.figure.savefig(
    os.path.join(procdir, "fft_filter_highlowpass_after.png")
)
# Laplacian spatial kernel
s10 = s.deepcopy()
w_laplacian = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

p = s10.inav[0, 0].deepcopy().data.astype(np.float32)
p2 = correlate(p, w_laplacian)
p2 = kp.pattern.rescale_intensity(p2, dtype_out=np.uint8)
plt.figure()
plt.imshow(p2, cmap="gray")
plt.colorbar()
plt.savefig(
    os.path.join(procdir, "fft_filter_laplacian_correlate.png"),
    bbox_inches="tight",
    pad_inches=0,
)

s10.fft_filter(transfer_function=w_laplacian, function_domain="spatial")
s10.plot()
s10._plot.signal_plot.figure.savefig(
    os.path.join(procdir, "fft_filter_laplacian_spatial.png")
)

# Close everything
plt.close("all")
