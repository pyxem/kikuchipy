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

from typing import Optional, Tuple

from diffsims.crystallography import ReciprocalLatticeVector
import matplotlib.pyplot as plt
import numpy as np
from orix.crystal_map import Phase
from orix.plot._util import Arrow3D
from orix.quaternion import Rotation
from orix.vector import Miller, Vector3d

from kikuchipy.detectors import EBSDDetector
from kikuchipy.simulations._kikuchi_pattern_simulation import (
    GeometricalKikuchiPatternSimulation,
)
from kikuchipy.simulations._kikuchi_pattern_features import (
    KikuchiPatternLine,
    KikuchiPatternZoneAxis,
)


class KikuchiPatternSimulator:
    def __init__(self, reflectors: ReciprocalLatticeVector):
        self.reflectors = reflectors

    @property
    def phase(self) -> Phase:
        return self.reflectors.phase

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    def on_detector(
        self,
        detector: EBSDDetector,
        rotations: Rotation,
    ) -> GeometricalKikuchiPatternSimulation:
        lattice = self.phase.structure.lattice
        ref = self.reflectors
        hkl = ref.hkl.round(0).astype(int)

        # Transformation from detector reference frame CSd to sample
        # reference frame CSs
        total_tilt = np.deg2rad(detector.sample_tilt - 90 - detector.tilt)
        u_s_bruker = Rotation.from_axes_angles((-1, 0, 0), total_tilt)
        u_s = Rotation.from_axes_angles((0, 0, -1), -np.pi / 2) * u_s_bruker
        u_s = u_s.to_matrix().squeeze()
        # Transformation from CSs to cartesian crystal reference frame
        # CSc
        u_o = rotations.to_matrix()
        # Transform rotations once
        u_os = np.matmul(u_o, u_s)
        # Transformation from CSc to reciprocal crystal reference frame
        # CSk*
        u_astar = lattice.recbase.T

        # Combine transformations
        u_kstar = np.matmul(u_astar, u_os)

        print(1)
        # Transform {hkl} from CSk* to CSd
        hkl_d = np.matmul(hkl, u_kstar)

        print(2)
        # Get vectors that are in some pattern
        nav_axes = (0, 1)[: rotations.ndim]
        hkl_is_upper, hkl_in_a_pattern = _get_coordinates_in_upper_hemisphere(
            z_coordinates=hkl_d[..., 2], navigation_axes=nav_axes
        )
        hkl_in_pattern = hkl_is_upper[..., hkl_in_a_pattern]
        hkl_d = hkl_d[..., hkl_in_a_pattern, :]

        print(3)
        # Visible reflectors
        reflectors_visible = self.reflectors[hkl_in_a_pattern]
        hkl = hkl[hkl_in_a_pattern]

        # Max. gnomonic radius to consider
        max_r_gnomonic = detector.r_max[0]

        print(4)
        lines = KikuchiPatternLine(
            hkl=Miller(hkl=hkl, phase=self.phase),
            hkl_detector=Vector3d(hkl_d),
            in_pattern=hkl_in_pattern,
            max_r_gnomonic=max_r_gnomonic,
        )

        print(5)
        # Zone axes <uvw> from {hkl}
        uvw = np.cross(hkl[:, None, :], hkl)
        uvw = uvw[~np.isclose(uvw, 0).all(axis=-1)]
        uvw = np.unique(uvw, axis=0)

        print(6)
        # Reduce an index triplet to smallest integer
        uvw = Miller(uvw=uvw, phase=self.phase)
        uvw = uvw.round().unique(use_symmetry=False)
        uvw = uvw.coordinates

        # Transformation from CSc to direct crystal reference frame CSk
        u_a = lattice.base

        # Combine transformations
        u_k = np.matmul(u_a, u_os)

        print(7)
        # Transform direct lattice vectors from CSk to CSd
        uvw_d = np.matmul(uvw, u_k)

        print(8)
        # Get vectors that are in some pattern
        uvw_is_upper, uvw_in_a_pattern = _get_coordinates_in_upper_hemisphere(
            z_coordinates=uvw_d[..., 2], navigation_axes=nav_axes
        )
        uvw_in_pattern = uvw_is_upper[..., uvw_in_a_pattern]
        uvw_d = uvw_d[..., uvw_in_a_pattern, :]

        print(9)
        # Visible zone axes
        uvw = uvw[uvw_in_a_pattern]

        print(10)
        zone_axes = KikuchiPatternZoneAxis(
            uvw=Miller(uvw=uvw, phase=self.phase),
            uvw_detector=Vector3d(uvw_d),
            in_pattern=uvw_in_pattern,
            max_r_gnomonic=max_r_gnomonic,
        )

        print(11)
        simulation = GeometricalKikuchiPatternSimulation(
            detector=detector,
            rotations=rotations,
            reflectors=reflectors_visible,
            lines=lines,
            zone_axes=zone_axes,
        )

        return simulation

    def plot_reflectors(
        self,
        projection: str = "stereographic",
        mode: str = "lines",
        figure: plt.Figure = None,
        return_figure: bool = False,
        backend: str = "matplotlib",
    ):
        ref = self.reflectors

        allowed_modes = ["lines", "bands"]
        if mode not in allowed_modes:
            raise ValueError(f"`mode` must be either of {allowed_modes}")
        if mode == "bands" and ref.theta[0] is None:
            raise ValueError(
                "Plotting of bands requires knowledge of the Bragg angle. Calculate "
                "with `self.reflectors.calculate_theta()`."
            )

        # Sort reflectors into groups with identical structure factors
        f_hkl = abs(ref.structure_factor)
        color = f_hkl / f_hkl.max()
        color = abs(color - color.min() - color.max())
        color = np.full((ref.size, 3), color[:, np.newaxis])

        if projection == "stereographic":
            kwargs = dict(color=color, linewidth=1)
            if mode == "lines":
                if figure is not None:
                    ref.draw_circle(figure=figure, **kwargs)
                else:
                    figure = ref.draw_circle(return_figure=True, **kwargs)
            else:  # bands
                v = Vector3d(ref)
                theta = ref.theta
                if figure is not None:
                    v.draw_circle(
                        opening_angle=np.pi / 2 - theta, figure=figure, **kwargs
                    )
                else:
                    figure = v.draw_circle(
                        opening_angle=np.pi / 2 - theta, return_figure=True, **kwargs
                    )
                v.draw_circle(opening_angle=np.pi / 2 + theta, figure=figure, **kwargs)
        elif projection == "spherical":
            figure = _plot_reflectors_spherical(mode, ref, color, backend, figure)
        else:
            raise ValueError

        if return_figure:
            return figure


def _get_coordinates_in_upper_hemisphere(
    z_coordinates: np.ndarray, navigation_axes: tuple
) -> Tuple[np.ndarray, np.ndarray]:
    """Return two boolean arrays with True if a coordinate is in the
    upper hemisphere and if it is in the upper hemisphere in some
    pattern, respectively.
    """
    upper_hemisphere = np.atleast_2d(z_coordinates) > 0
    in_a_pattern = np.sum(upper_hemisphere, axis=navigation_axes) != 0
    return upper_hemisphere, in_a_pattern


def _plot_reflectors_spherical(
    mode: str, ref: ReciprocalLatticeVector, color: np.ndarray, backend: str, figure
):
    v = Vector3d(ref).unit

    steps = 101
    if mode == "lines":
        circles = v.get_circle(steps=steps).data
    else:  # bands
        theta = ref.theta
        circles = (
            v.get_circle(opening_angle=np.pi / 2 - theta, steps=steps).data,
            v.get_circle(opening_angle=np.pi / 2 + theta, steps=steps).data,
        )

    colors = ["r", "g", "b"]
    labels = ["e1", "e2", "e3"]

    if backend == "matplotlib":
        if figure is not None:
            ax = figure.axes[0]
        else:
            figure, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        if mode == "lines":
            for i, ci in enumerate(circles):
                ax.plot3D(*ci.T, c=color[i])
        else:  # bands
            for i, (c1i, c2i) in enumerate(zip(*circles)):
                ax.plot3D(*c1i.T, c=color[i])
                ax.plot3D(*c2i.T, c=color[i])
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
        import pyvista as pv

        if figure is None:
            show = True
            figure = pv.Plotter()
            figure.add_axes(
                color="k", xlabel=labels[0], ylabel=labels[1], zlabel=labels[2]
            )
            sphere = pv.Sphere(radius=0.99)
            figure.add_mesh(sphere, color="w", lighting=False)
            figure.disable_shadows()
            figure.set_background("w")
            figure.set_viewup((0, 1, 0))
        else:
            show = False
            # Assume that the existing plot has a sphere with a radius of 1
            v = Vector3d(circles)
            circles = Vector3d.from_polar(v.azimuth, v.polar, radial=1.01).data

        if mode == "lines":
            circles_shape = circles.shape[:-1]
            circles = circles.reshape((-1, 3))
            lines = np.arange(circles.shape[0]).reshape(circles_shape)
            color = color[:, 0]
        else:  # bands
            circles = np.vstack(circles)
            circles_shape = circles.shape[:-1]
            circles = circles.reshape((-1, 3))
            lines = np.arange(circles.shape[0]).reshape(circles_shape)
            color = np.tile(color[:, 0], 2)

        # Create mesh from vertices (3D coordinates) and line
        # connectivity arrays
        lines = np.insert(lines, 0, steps, axis=1)
        lines = lines.ravel()
        mesh = pv.PolyData(circles, lines=lines)
        figure.add_mesh(
            mesh,
            scalars=color,
            cmap="gray",
            scalar_bar_args=dict(title=r"$F_{hkl}$"),
        )

        if show:
            figure.show()

    return figure
