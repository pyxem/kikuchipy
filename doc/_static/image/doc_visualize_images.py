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
import logging

import matplotlib as mpl

mpl.rcParams["backend"] = "qt5agg"
import matplotlib.pyplot as plt
import hyperspy.api as hs
import numpy as np
import skimage.exposure as ske
import skimage.color as skc
import skimage.transform as skt

import kikuchipy as kp

data = "/home/hakon/phd/data/ni/190506_ebsd/1/nordif/Pattern.dat"
data_mp = "/home/hakon/kode/emsoft/emdata/crystal_data/ni/ni_mc_mp_20kv.h5"
imgdir = "/home/hakon/kode/kikuchipy/doc/_static/image"
visdir = os.path.join(imgdir, "visualizing_patterns")

datadir, fname = os.path.split(data)
fname, ext = os.path.splitext(fname)
s = kp.load(data, lazy=False)

# VBSE
roi = hs.roi.RectangularROI(left=18, top=20, right=23, bottom=25)
s.plot_virtual_bse_intensity(roi)
s._plot.navigator_plot.figure.savefig(
    os.path.join(visdir, "standard_navigator.jpg")
)
s._plot.signal_plot.figure.savefig(os.path.join(visdir, "pattern_roi.jpg"))

# Virtual image
vbse_gen = kp.generators.VirtualBSEGenerator(s)
vbse_gen.grid_shape = (10, 10)
red = [(7, 1), (8, 1), (8, 2), (9, 1), (9, 2)]
green = [(8, 4), (8, 5), (9, 4), (9, 5)]
blue = [(7, 8), (8, 7), (8, 8), (9, 7), (9, 8)]
vbse_rgb = vbse_gen.get_rgb_image(r=red, g=green, b=blue)
s.axes_manager.navigation_axes[0].index = 155
s.axes_manager.navigation_axes[1].index = 77
s.plot(navigator=vbse_rgb)
s._plot.navigator_plot.figure.savefig(
    os.path.join(visdir, "vbse_navigator.png"), bbox_inches="tight",
)
s._plot.signal_plot.figure.savefig(os.path.join(visdir, "vbse_signal.png"),)

# Quality metric map
osm = plt.imread(os.path.join(visdir, "orientation_similarity_map.png"))
osm = skc.rgb2gray(skc.rgba2rgb(osm))
s_osm = hs.signals.Signal2D(osm)
s.axes_manager.navigation_axes[0].index = 155
s.axes_manager.navigation_axes[1].index = 77
s.plot(navigator=s_osm)
s._plot.navigator_plot.figure.savefig(
    os.path.join(visdir, "orientation_similarity_map_navigator.jpg")
)

# Orientation map
om = plt.imread(os.path.join(visdir, "orientation_map.jpg"))
om_rescaled = skt.resize(
    om, s.axes_manager.navigation_shape[::-1], anti_aliasing=False
)
om_scaled = ske.rescale_intensity(om_rescaled, out_range=np.uint8)
s_om = hs.signals.Signal2D(om_scaled)
s_om = s_om.transpose(signal_axes=1)
s_om.change_dtype("rgb8")
s.axes_manager.navigation_axes[0].index = 155
s.axes_manager.navigation_axes[1].index = 77
s.plot(navigator=s_om)
s._plot.navigator_plot.figure.savefig(
    os.path.join(visdir, "orientation_map_navigator.jpg")
)

# Plot simulated and experimental side-by-side
s.remove_static_background()
s.remove_dynamic_background()
s_sim = kp.load(os.path.join(datadir, "../emsoft/orig/ni_emebsd2.h5"))
s.plot()
s._plot.signal_plot.figure.savefig(os.path.join(visdir, "pattern.jpg"))
s_sim.plot()
s_sim._plot.signal_plot.figure.savefig(
    os.path.join(visdir, "simulated_pattern.jpg")
)

plt.close("all")
