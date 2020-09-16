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

import matplotlib.pyplot as plt

from kikuchipy.detectors import EBSDDetector
from kikuchipy import load


data = "/home/hakon/kode/kikuchipy/kikuchipy/data/kikuchipy/patterns.h5"
outdir = "/home/hakon/kode/kikuchipy/doc/_static/image"
refframe_dir = os.path.join(outdir, "reference_frames")

datadir, fname = os.path.split(data)
fname, ext = os.path.splitext(fname)
s = load(data)

s.remove_static_background()
s.remove_dynamic_background()
s.rescale_intensity(percentiles=(0.5, 99.5))

det = EBSDDetector(shape=(60, 60), pc=[0.4210, 0.2206, 0.5049],)
fig, ax = det.plot(
    coordinates="detector",
    pattern=s.inav[0, 0].data,
    show_pc=True,
    return_fig_ax=True,
)
arrow_dict1 = {
    "x": 0,
    "y": 0,
    "width": det.nrows * 0.01,
    "head_width": 3,
    "head_length": 4,
    "zorder": 10,
    "clip_on": False,
}
arrow_length = det.ncols * 0.2
x_color = "r"
y_color = "b"  # green (0, 0.78, 0)
ax.set_xlabel(ax.get_xlabel(), color=x_color)
ax.set_ylabel(ax.get_ylabel(), color=y_color)
ax.arrow(
    dx=arrow_length, dy=0, fc=x_color, ec=x_color, **arrow_dict1,
)
ax.arrow(
    dx=0, dy=arrow_length, fc=y_color, ec=y_color, **arrow_dict1,
)
fig.savefig(
    os.path.join(refframe_dir, "detector_coordinates.png"),
    bbox_inches="tight",
    pad_inches=0,
    transparent=True,
    dpi=150,
)

fig2, ax2 = det.plot(
    coordinates="gnomonic",
    pattern=s.inav[0, 0].data,
    show_pc=True,
    pc_kwargs={"zorder": 500},
    draw_gnomonic_circles=True,
    gnomonic_circles_kwargs={"alpha": 0.4},
    return_fig_ax=True,
)
gn_scale = det.x_scale.squeeze()
arrow_dict = {
    "width": det.nrows * 0.01 * gn_scale,
    "head_width": 3 * gn_scale,
    "head_length": 4 * gn_scale,
    "clip_on": False,
    "zorder": 10,
}
arrow_length = det.ncols * 0.2 * gn_scale
ax2.set_xlabel(ax2.get_xlabel(), color=x_color)
ax2.set_ylabel(ax2.get_ylabel(), color=y_color)
ax2.arrow(x=0, y=0, dx=arrow_length, dy=0, fc=x_color, ec=x_color, **arrow_dict)
ax2.arrow(x=0, y=0, dx=0, dy=arrow_length, fc=y_color, ec=y_color, **arrow_dict)
fig2.savefig(
    os.path.join(refframe_dir, "gnomonic_coordinates.png"),
    bbox_inches="tight",
    pad_inches=0,
    transparent=True,
    dpi=150,
)

plt.close("all")
