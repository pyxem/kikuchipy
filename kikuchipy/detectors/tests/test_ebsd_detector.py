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

from copy import deepcopy

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest

from kikuchipy.detectors import EBSDDetector


class TestEBSDDetector:
    def test_init(self, pc1):
        """Initialization works."""
        shape = (1, 2)
        px_size = 3
        binning = 4
        tilt = 5
        det = EBSDDetector(
            shape=shape, px_size=px_size, binning=binning, tilt=tilt, pc=pc1,
        )
        assert det.shape == shape
        assert det.aspect_ratio == 0.5

    @pytest.mark.parametrize(
        "nav_shape, desired_nav_shape, desired_nav_dim",
        [
            ((), (1,), 1),
            ((1,), (1,), 1),
            ((10, 1), (10,), 1),
            ((10, 10, 1), (10, 10), 2),
        ],
    )
    def test_nav_shape_dim(
        self, pc1, nav_shape, desired_nav_shape, desired_nav_dim
    ):
        """Navigation shape and dimension is derived correctly from PC shape."""
        det = EBSDDetector(pc=np.tile(pc1, nav_shape))
        assert det.navigation_shape == desired_nav_shape
        assert det.navigation_dimension == desired_nav_dim

    @pytest.mark.parametrize("pc_type", [list, tuple, np.asarray])
    def test_pc_initialization(self, pc1, pc_type):
        """Initialize PC of valid types."""
        det = EBSDDetector(pc=pc_type(pc1))
        assert isinstance(det.pc, np.ndarray)

    @pytest.mark.parametrize(
        (
            "shape, px_size, binning, pc, ssd, width, height, size, "
            "shape_unbinned, px_size_binned"
        ),
        [
            # fmt: off
            (
                (60, 60), 70, 8, [1, 1, 0.5], 16800, 33600, 33600, 3600,
                (480, 480), 560,
            ),
            (
                (60, 60), 70, 8, [1, 1, 0.7], 23520, 33600, 33600, 3600,
                (480, 480), 560,
            ),
            (
                (480, 460), 70, 0.5, [1, 1, 0.7], 11760, 16100, 16800, 220800,
                (240, 230), 35,
            ),
            (
                (340, 680), 40, 2, [1, 1, 0.7], 19040, 54400, 27200, 231200,
                (680, 1360), 80,
            ),
            # fmt: on
        ],
    )
    def test_detector_dimensions(
        self,
        shape,
        px_size,
        binning,
        pc,
        ssd,
        width,
        height,
        size,
        shape_unbinned,
        px_size_binned,
    ):
        """Initialization yields expected derived values."""
        detector = EBSDDetector(
            shape=shape, px_size=px_size, binning=binning, pc=pc
        )
        assert detector.specimen_scintillator_distance == ssd
        assert detector.width == width
        assert detector.height == height
        assert detector.size == size
        assert detector.unbinned_shape == shape_unbinned
        assert detector.px_size_binned == px_size_binned

    def test_repr(self, pc1):
        """Expected string representation."""
        assert repr(
            EBSDDetector(shape=(1, 2), px_size=3, binning=4, tilt=5, pc=pc1)
        ) == (
            "EBSDDetector (1, 2), px_size 3 um, binning 4, tilt 5, pc "
            "(0.421, 0.779, 0.505)"
        )

    def test_deepcopy(self, pc1):
        """Yields the expected parameters and an actual deep copy."""
        detector1 = EBSDDetector(pc=pc1)
        detector2 = detector1.deepcopy()
        detector1.pcx += 0.1
        assert np.allclose(detector1.pcx, 0.521)
        assert np.allclose(detector2.pcx, 0.421)

    def test_set_pc_coordinates(self, pc1):
        """Returns desired arrays with desired shapes."""
        ny, nx = (2, 3)
        nav_shape = (ny, nx)
        n = ny * nx
        detector = EBSDDetector(pc=np.tile(pc1, nav_shape + (1,)))
        assert detector.navigation_shape == nav_shape

        new_pc = np.zeros(nav_shape + (3,))
        new_pc[..., 0] = pc1[0] * 0.01 * np.arange(n).reshape(nav_shape)
        new_pc[..., 1] = pc1[1] * 0.01 * np.arange(n).reshape(nav_shape)
        new_pc[..., 2] = pc1[2] * 0.01 * np.arange(n).reshape(nav_shape)
        detector.pcx = new_pc[..., 0]
        detector.pcy = new_pc[..., 1]
        detector.pcz = new_pc[..., 2]
        assert np.allclose(detector.pc, new_pc)

    @pytest.mark.parametrize(
        "pc, desired_pc_average",
        [
            ([0.1234, 0.1235, 0.1234], [0.1230, 0.1240, 0.1230]),
            (np.arange(30).reshape((2, 5, 3)), [13.5, 14.5, 15.5]),
            (np.arange(30).reshape((10, 3)), [13.5, 14.5, 15.5]),
        ],
    )
    def test_pc_average(self, pc, desired_pc_average):
        """Calculation of PC average."""
        assert np.allclose(EBSDDetector(pc=pc).pc_average, desired_pc_average)

    @pytest.mark.parametrize(
        "pc, desired_nav_shape, desired_nav_ndim",
        [
            (np.arange(30).reshape((2, 5, 3)), (5, 2), 2),
            (np.arange(30).reshape((5, 2, 3)), (10, 1), 2),
            (np.arange(30).reshape((2, 5, 3)), (10,), 1),
        ],
    )
    def test_set_navigation_shape(
        self, pc, desired_nav_shape, desired_nav_ndim
    ):
        """Change shape of PC array."""
        detector = EBSDDetector(pc=pc)
        detector.navigation_shape = desired_nav_shape
        assert detector.navigation_shape == desired_nav_shape
        assert detector.navigation_dimension == desired_nav_ndim
        assert detector.pc.shape == desired_nav_shape + (3,)

    def test_set_navigation_shape_raises(self, pc1):
        """Desired error message."""
        detector = EBSDDetector(pc=pc1)
        with pytest.raises(ValueError, match="A maximum dimension of 2"):
            detector.navigation_shape = (1, 2, 3)

    @pytest.mark.parametrize(
        "shape, desired_x_range, desired_y_range",
        [
            ((60, 60), [-0.833828, 1.1467617], [-0.4369182, 1.54367201]),
            ((510, 510), [-0.833828, 1.1467617], [-0.4369182, 1.54367201]),
            ((1, 1), [-0.833828, 1.1467617], [-0.4369182, 1.54367201]),
            ((480, 640), [-0.6253713, 0.860071], [-0.4369182, 1.54367201]),
        ],
    )
    def test_gnomonic_range(self, pc1, shape, desired_x_range, desired_y_range):
        """Gnomonic x/y range, x depends on aspect ratio."""
        detector = EBSDDetector(shape=shape, pc=pc1)
        assert np.allclose(detector.x_range, desired_x_range)
        assert np.allclose(detector.y_range, desired_y_range)

    @pytest.mark.parametrize(
        "shape, desired_x_scale, desired_y_scale",
        [
            ((60, 60), 0.033569, 0.033569),
            ((510, 510), 0.003891, 0.003891),
            ((1, 1), 1.980590, 1.980590),
            ((480, 640), 0.002324, 0.004134),
        ],
    )
    def test_gnomonic_scale(self, pc1, shape, desired_x_scale, desired_y_scale):
        """Gnomonic x/y scale."""
        detector = EBSDDetector(shape=shape, pc=pc1)
        assert np.allclose(detector.x_scale, desired_x_scale, atol=1e-6)
        assert np.allclose(detector.y_scale, desired_y_scale, atol=1e-6)

    @pytest.mark.parametrize(
        "shape, pc, px_size, binning, version, desired_pc",
        [
            (
                (60, 60),
                [3.4848, 114.2016, 15767.7],
                59.2,
                8,
                4,
                [0.50726, 0.26208, 0.55849],
            ),
            (
                (60, 60),
                [-3.4848, 114.2016, 15767.7],
                59.2,
                8,
                5,
                [0.50726, 0.26208, 0.55849],
            ),
            (
                (61, 61),
                [-10.632, 145.5187, 19918.9],
                59.2,
                8,
                4,
                [0.47821, 0.20181, 0.68948],
            ),
            (
                (61, 61),
                [10.632, 145.5187, 19918.9],
                59.2,
                8,
                5,
                [0.47821, 0.20181, 0.68948],
            ),
            (
                (80, 60),
                [-0.55, -13.00, 16075.2],
                50,
                6,
                4,
                [0.4991, 0.5271, 0.6698],
            ),
            (
                (80, 60),
                [0.55, -13.00, 16075.2],
                50,
                6,
                5,
                [0.4991, 0.5271, 0.6698],
            ),
            ((480, 640), [0, 0, 15000], 50, 1, 5, [0.5, 0.5, 0.625]),
        ],
    )
    def test_set_pc_from_emsoft(
        self, shape, pc, px_size, binning, version, desired_pc
    ):
        """PC EMsoft -> Bruker -> EMsoft, also checking to_tsl() and
        to_bruker().
        """
        det = EBSDDetector(
            shape=shape,
            pc=pc,
            px_size=px_size,
            binning=binning,
            convention=f"emsoft{version}",
        )

        assert np.allclose(det.pc, desired_pc, atol=1e-2)
        assert np.allclose(det.pc_emsoft(version=version), pc, atol=1e-2)
        assert np.allclose(det.pc_bruker(), desired_pc, atol=1e-2)

        pc_tsl = deepcopy(det.pc)
        pc_tsl[..., 1] = 1 - pc_tsl[..., 1]
        assert np.allclose(det.pc_tsl(), pc_tsl, atol=1e-2)
        assert np.allclose(det.pc_oxford(), pc_tsl, atol=1e-2)

    def test_set_pc_from_emsoft_no_version(self):
        """PC EMsoft -> Bruker, no EMsoft version specified gives v5."""
        assert np.allclose(
            EBSDDetector(
                shape=(60, 60),
                pc=[-3.4848, 114.2016, 15767.7],
                px_size=59.2,
                binning=8,
                convention="emsoft",
            ).pc,
            [0.50726, 0.26208, 0.55849],
            atol=1e-2,
        )

    @pytest.mark.parametrize(
        "pc, convention, desired_pc",
        [
            ([0.35, 1, 0.65], "tsl", [0.35, 0, 0.65]),
            ([0.25, 0, 0.75], "oxford", [0.25, 1, 0.75]),
            ([0.1, 0.2, 0.3], "amatek", [0.1, 0.8, 0.3]),
            ([0.6, 0.6, 0.6], "edax", [0.6, 0.4, 0.6]),
        ],
    )
    def test_set_pc_from_tsl_oxford(self, pc, convention, desired_pc):
        """PC TSL -> Bruker -> TSL."""
        det = EBSDDetector(pc=pc, convention=convention)
        assert np.allclose(det.pc, desired_pc)
        assert np.allclose(det.pc_tsl(), pc)
        assert np.allclose(
            EBSDDetector(pc=det.pc_tsl(), convention="tsl").pc_tsl(), pc
        )

    @pytest.mark.parametrize(
        "pc, convention",
        [
            ([0.35, 1, 0.65], None),
            ([0.25, 0, 0.75], None),
            ([0.1, 0.2, 0.3], "Bruker"),
            ([0.6, 0.6, 0.6], "bruker"),
        ],
    )
    def test_set_pc_from_bruker(self, pc, convention):
        """PC Bruker returns Bruker PC, which is the default."""
        det = EBSDDetector(pc=pc, convention=convention)
        assert np.allclose(det.pc, pc)

    def test_set_pc_convention_raises(self, pc1):
        """Wrong convention raises."""
        with pytest.raises(ValueError, match="Projection center convention "):
            _ = EBSDDetector(pc=pc1, convention="nordif")

    @pytest.mark.parametrize(
        "coordinates, show_pc, pattern, zoom, desired_labels",
        [
            (
                None,
                False,
                None,
                1,
                ["$x_{\mathrm{detector}}$", "$y_{\mathrm{detector}}$"],
            ),
            (
                "detector",
                True,
                np.ones((60, 60)),
                1,
                ["$x_{\mathrm{detector}}$", "$y_{\mathrm{detector}}$"],
            ),
            (
                "gnomonic",
                True,
                np.ones((60, 60)),
                2,
                ["$x_{\mathrm{gnomonic}}$", "$y_{\mathrm{gnomonic}}$"],
            ),
        ],
    )
    def test_plot_detector(
        self, detector, coordinates, show_pc, pattern, zoom, desired_labels
    ):
        """Plotting detector works, *not* checking whether Matplotlib
        displays the pattern correctly.
        """
        _, ax = detector.plot(
            coordinates=coordinates,
            show_pc=show_pc,
            pattern=pattern,
            zoom=zoom,
            return_fig_ax=True,
        )
        assert ax.get_xlabel() == desired_labels[0]
        assert ax.get_ylabel() == desired_labels[1]
        if isinstance(pattern, np.ndarray):
            assert np.allclose(ax.get_images()[0].get_array(), pattern)
        plt.close("all")

    @pytest.mark.parametrize(
        "gnomonic_angles, gnomonic_circles_kwargs",
        [
            ([10, 20], {"edgecolor": "b"}),
            (np.arange(1, 3) * 10, {"edgecolor": "r"}),
            (None, None),
        ],
    )
    def test_plot_detector_gnomonic_circles(
        self, detector, gnomonic_angles, gnomonic_circles_kwargs
    ):
        """Draw gnomonic circles."""
        _, ax = detector.plot(
            coordinates="gnomonic",
            draw_gnomonic_circles=True,
            gnomonic_angles=gnomonic_angles,
            gnomonic_circles_kwargs=gnomonic_circles_kwargs,
            return_fig_ax=True,
        )
        if gnomonic_angles is None:
            n_angles = 8
        else:
            n_angles = len(gnomonic_angles)
        assert len(ax.artists) == n_angles
        if gnomonic_circles_kwargs is None:
            edgecolor = "k"
        else:
            edgecolor = gnomonic_circles_kwargs["edgecolor"]
        assert np.allclose(
            ax.artists[0]._edgecolor[:3], mcolors.to_rgb(edgecolor)
        )
        plt.close("all")

    @pytest.mark.parametrize("pattern", [np.ones((61, 61)), np.ones((59, 60))])
    def test_plot_detector_pattern_raises(self, detector, pattern):
        """Pattern shape unequal to detector shape raises ValueError."""
        with pytest.raises(ValueError, match=f"Pattern shape {pattern.shape}*"):
            detector.plot(pattern=pattern)
        plt.close("all")

    @pytest.mark.parametrize(
        "pattern_kwargs", [None, {"cmap": "inferno"}, {"cmap": "plasma"}]
    )
    def test_plot_pattern_kwargs(self, detector, pattern_kwargs):
        """Pass pattern kwargs to imshow()."""
        _, ax = detector.plot(
            pattern=np.ones((60, 60)),
            pattern_kwargs=pattern_kwargs,
            return_fig_ax=True,
        )
        if pattern_kwargs is None:
            pattern_kwargs = {"cmap": "gray"}
        assert ax.images[0].cmap.name == pattern_kwargs["cmap"]
        plt.close("all")

    @pytest.mark.parametrize(
        "pc_kwargs", [None, {"facecolor": "r"}, {"facecolor": "b"}]
    )
    def test_plot_pc_kwargs(self, detector, pc_kwargs):
        """Pass PC kwargs to scatter()."""
        _, ax = detector.plot(
            show_pc=True, pc_kwargs=pc_kwargs, return_fig_ax=True
        )
        if pc_kwargs is None:
            pc_kwargs = {"facecolor": "gold"}
        assert np.allclose(
            ax.collections[0].get_facecolor().squeeze()[:3],
            mcolors.to_rgb(pc_kwargs["facecolor"]),
        )
        plt.close("all")

    @pytest.mark.parametrize("coordinates", ["detector", "gnomonic"])
    def test_plot_extent(self, detector, coordinates):
        """Correct detector extent."""
        _, ax = detector.plot(
            coordinates=coordinates,
            pattern=np.ones(detector.shape),
            return_fig_ax=True,
        )
        if coordinates == "gnomonic":
            desired_data_lim = np.concatenate(
                [
                    detector._average_gnomonic_bounds[::2],
                    np.diff(detector._average_gnomonic_bounds)[::2],
                ]
            )
        else:
            desired_data_lim = np.sort(detector.bounds)
        assert np.allclose(ax.dataLim.bounds, desired_data_lim)
        plt.close("all")

    @pytest.mark.parametrize(
        "shape, desired_shapes",
        [
            (
                (1,),  # PC
                [
                    (4,),  # extent
                    (1,),  # x_min
                    (1,),  # y_min
                    (1, 2),  # x_range
                    (1, 2),  # y_range
                    (1,),  # x_scale
                    (1,),  # y_scale
                    (1, 4),  # extent_gnomonic
                ],
            ),
            (
                (10,),
                [(4,), (10,), (10,), (10, 2), (10, 2), (10,), (10,), (10, 4),],
            ),
            (
                (10, 10),
                [
                    (4,),
                    (10, 10),
                    (10, 10),
                    (10, 10, 2),
                    (10, 10, 2),
                    (10, 10),
                    (10, 10),
                    (10, 10, 4),
                ],
            ),
            (
                (1, 10),
                [
                    (4,),
                    (1, 10),
                    (1, 10),
                    (1, 10, 2),
                    (1, 10, 2),
                    (1, 10),
                    (1, 10),
                    (1, 10, 4),
                ],
            ),
            (
                (10, 1),
                [
                    (4,),
                    (10, 1),
                    (10, 1),
                    (10, 1, 2),
                    (10, 1, 2),
                    (10, 1),
                    (10, 1),
                    (10, 1, 4),
                ],
            ),
        ],
    )
    def test_property_shapes(self, shape, desired_shapes):
        """Expected property shapes when varying navigation shape."""
        det = EBSDDetector(pc=np.ones(shape + (3,)))
        assert det.bounds.shape == desired_shapes[0]
        assert det.x_min.shape == desired_shapes[1]
        assert det.y_min.shape == desired_shapes[2]
        assert det.x_range.shape == desired_shapes[3]
        assert det.y_range.shape == desired_shapes[4]
        assert det.x_scale.shape == desired_shapes[5]
        assert det.y_scale.shape == desired_shapes[6]
        assert det.gnomonic_bounds.shape == desired_shapes[7]
