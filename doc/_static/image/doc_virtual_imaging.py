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

import kikuchipy as kp

data = "/home/hakon/phd/data/ni/2020/1/nordif/Pattern.dat"
outdir = "/home/hakon/kode/kikuchipy/doc/_static/image"
vbsedir = os.path.join(outdir, "virtual_backscatter_electron_imaging")

datadir, fname = os.path.split(data)
fname, ext = os.path.splitext(fname)
s = kp.load(data, lazy=False)

vbse_gen = kp.generators.VirtualBSEGenerator(s)

# Plot grid
vbse_gen.grid_shape = (10, 10)
red = [(7, 1), (8, 1), (8, 2), (9, 1), (9, 2)]
green = [(8, 4), (8, 5), (9, 4), (9, 5)]
blue = [(7, 8), (8, 7), (8, 8), (9, 7), (9, 8)]
p = vbse_gen.plot_grid(
    pattern_idx=(100, 87),
    rgb_channels=[red, green, blue],
    visible_indices=True,
)
p._plot.signal_plot.figure.savefig(os.path.join(vbsedir, "plot_grid.jpg"),)

# Get an RGB image
vbse_rgb = vbse_gen.get_rgb_image(r=red, g=green, b=blue)
vbse_rgb.plot()
vbse_rgb._plot.signal_plot.figure.savefig(
    os.path.join(vbsedir, "rgb_image.jpg"), bbox_inches="tight",
)

# Get one image per grid tile
vbse_gen.grid_shape = (5, 5)
vbse_imgs = vbse_gen.get_images_from_grid()
pattern_idx = (2, 4)
vbse_imgs.axes_manager[0].index = pattern_idx[0]
vbse_imgs.axes_manager[1].index = pattern_idx[1]
vbse_imgs.plot()
vbse_imgs._plot.navigator_plot.figure.savefig(
    os.path.join(vbsedir, "images_nav.jpg"), bbox_inches="tight", pad_inches=0,
)
vbse_imgs._plot.signal_plot.figure.savefig(
    os.path.join(vbsedir, "images_sig.jpg"),
)

# Get an RGB image with alpha channel
s2 = s.deepcopy()
s2.remove_static_background()
s2.remove_dynamic_background()
s2.average_neighbour_patterns()
iq = s2.get_image_quality()
vbse_gen.grid_shape = (10, 10)
vbse_rgba = vbse_gen.get_rgb_image(r=red, g=green, b=blue, alpha=iq)
vbse_rgba.plot()
vbse_rgba._plot.signal_plot.figure.savefig(
    os.path.join(vbsedir, "rgba_image.jpg"),
    bbox_inches="tight",
    pad_inches=0,
    dpi=300,
)

plt.close("all")
