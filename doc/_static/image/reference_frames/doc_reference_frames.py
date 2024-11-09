# Copyright 2019-2024 The kikuchipy developers
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

from pathlib import Path

import matplotlib.pyplot as plt

import kikuchipy as kp

doc_path = Path(__file__).parent.parent.parent.parent
data = doc_path / "../src/kikuchipy/data/kikuchipy_h5ebsd/patterns.h5"
ref_frame_path = doc_path / "_static/image/reference_frames"

s = kp.load(data)

s.remove_static_background()
s.remove_dynamic_background()
s.rescale_intensity(percentiles=(0.5, 99.5))

det = kp.detectors.EBSDDetector(shape=(60, 60), pc=[0.4210, 0.2206, 0.5049])
fig = det.plot(
    coordinates="detector", pattern=s.inav[0, 0].data, show_pc=True, return_figure=True
)
ax = fig.axes[0]
arrow_dict1 = {
    "x": 0,
    "y": 0,
    "width": det.nrows * 0.01,
    "head_width": 3,
    "head_length": 4,
    "clip_on": False,
    "zorder": 10,
}
arrow_length = det.ncols * 0.2
x_color = "r"
y_color = (0, 0.78, 0)  # green (0, 0.78, 0)
ax.set_xlabel(ax.get_xlabel(), color=x_color)
ax.set_ylabel(ax.get_ylabel(), color=y_color)
ax.arrow(dx=arrow_length, dy=0, fc=x_color, ec=x_color, **arrow_dict1)
ax.arrow(dx=0, dy=arrow_length, fc=y_color, ec=y_color, **arrow_dict1)
fig.savefig(
    ref_frame_path / "detector_coordinates.png",
    bbox_inches="tight",
    pad_inches=0,
    transparent=True,
    dpi=100,
)

fig2 = det.plot(
    coordinates="gnomonic",
    pattern=s.inav[0, 0].data,
    show_pc=True,
    pc_kwargs={"zorder": 500},
    draw_gnomonic_circles=True,
    gnomonic_circles_kwargs={"alpha": 0.4},
    return_figure=True,
)
ax2 = fig2.axes[0]
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
    ref_frame_path / "gnomonic_coordinates.png",
    bbox_inches="tight",
    pad_inches=0,
    transparent=True,
    dpi=100,
)

plt.close("all")
