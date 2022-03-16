# Copyright 2019-2022 The kikuchipy developers
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

from typing import Optional

from diffsims.crystallography import ReciprocalLatticeVector
import matplotlib.pyplot as plt
import numpy as np
from orix.crystal_map import Phase
from orix.plot._util import Arrow3D
from orix.vector import Vector3d


class KikuchiPatternSimulator:
    def __init__(self, phase: Phase, reflectors: ReciprocalLatticeVector):
        self.phase = phase
        self.reflectors = reflectors

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def plot_reflectors(
        self,
        projection: str = "stereographic",
        feature: str = "lines",
        scale_intensity: Optional[str] = "structure_factor",
        return_figure: bool = False,
        backend: str = "matplotlib",
    ):
        ref = self.reflectors

        allowed_features = ["lines", "bands"]
        if feature not in allowed_features:
            raise ValueError(f"`feature` must be either of {allowed_features}")
        if feature == "bands" and ref.theta[0] is None:
            raise ValueError(
                "Plotting of bands requires knowledge of the Bragg angle. Calculate "
                "with `self.reflectors.calculate_theta()`."
            )

        # Scale colors by structure factor if available
        structure_factor = ref.structure_factor
        if structure_factor[0] is not None and scale_intensity == "structure_factor":
            color = structure_factor / structure_factor.max()
            color = abs(color - color.min() - color.max())
            ref_colors = np.full((ref.size, 3), color[:, np.newaxis])
        else:
            ref_colors = np.zeros((ref.size, 3))

        if projection == "stereographic":
            if feature == "lines":
                fig = ref.draw_circle(return_figure=True, color=ref_colors)
            else:  # bands
                v = Vector3d(ref)
                theta = ref.theta
                fig = v.draw_circle(
                    opening_angle=np.pi / 2 - theta,
                    color=ref_colors,
                    return_figure=True,
                )
                v.draw_circle(
                    opening_angle=np.pi / 2 + theta, figure=fig, color=ref_colors
                )
        elif projection == "spherical":
            fig = _plot_spherical(feature, ref, ref_colors, backend)
        else:
            raise ValueError

        if return_figure:
            return fig


def _plot_spherical(feature, ref, ref_colors, backend):
    v = Vector3d(ref).unit

    if feature == "lines":
        circles = v.get_circle().data
    else:  # bands
        theta = ref.theta
        circles = (
            v.get_circle(opening_angle=np.pi / 2 - theta).data,
            v.get_circle(opening_angle=np.pi / 2 + theta).data,
        )

    colors = ["r", "g", "b"]
    labels = ["Xc", "Yc", "Zc"]

    if backend == "matplotlib":
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        if feature == "lines":
            for i, ci in enumerate(circles):
                ax.plot3D(*ci.T, c=ref_colors[i])
        else:  # bands
            for i, (c1i, c2i) in enumerate(zip(*circles)):
                ax.plot3D(*c1i.T, c=ref_colors[i])
                ax.plot3D(*c2i.T, c=ref_colors[i])
        ax.axis("off")
        ax.set_xlim((-1, 1))
        ax.set_ylim((-1, 1))
        ax.set_zlim((-1, 1))
        ax.set_box_aspect((1, 1, 1))
        ax.azim = -90
        ax.elev = 90
        arrow_kwargs = dict(mutation_scale=20, arrowstyle="-|>", linewidth=1)
        ha = ["left", "center", "center"]
        va = ["center", "bottom", "bottom"]
        for i in range(3):
            data = np.zeros((3, 2))
            data[i, 0] = 1
            data[i, 1] = 1.5
            arrow = Arrow3D(
                *data, color=colors[i], label=labels[i] + " axis", **arrow_kwargs
            )
            ax.add_artist(arrow)
            ax.text3D(
                *data[:, 1],
                s=labels[i],
                color=colors[i],
                label=labels[i] + " label",
                ha=ha[i],
                va=va[i],
            )
    else:  # pyvista
        import pyvista

        pl = pyvista.Plotter()
        pl.add_axes(color="k", xlabel=labels[0], ylabel=labels[1], zlabel=labels[2])
        sphere = pyvista.Sphere(radius=1, theta_resolution=360, phi_resolution=360)
        pl.add_mesh(sphere, color="w", lighting=False)
        if feature == "lines":
            for i, ci in enumerate(circles):
                pl.add_lines(lines=ci, color=tuple(ref_colors[i]), width=2)
        else:  # bands
            for i, (c1i, c2i) in enumerate(zip(*circles)):
                pl.add_lines(lines=c1i, color=tuple(ref_colors[i]), width=2)
                pl.add_lines(lines=c2i, color=tuple(ref_colors[i]), width=2)
        pl.disable_shadows()
        pl.set_background("w")
        pl.show()
        fig = pl

    return fig
