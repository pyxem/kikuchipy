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

import numpy as np
import pytest

from kikuchipy.simulations.features import KikuchiBand


class TestKikuchiBand:
    @pytest.mark.parametrize(
        "hkl, hkl_detector, within_gnomonic_radius, hesse_distance, "
        "hesse_alpha,nav_shape, nav_dim, size",
        [
            (
                np.array([[-1, 1, 1], [2, 0, 0]]),
                np.array([[0.26, 0.32, 0.26], [0.21, -0.45, -0.27]]),
                np.array([True, False]),
                np.array([0.63059263, -0.54370988]),
                np.array([1.5076952, np.nan]),
                (),
                0,
                2,
            ),
            (
                np.array([[-1, 1, 1], [2, 0, 0]]),
                np.tile(
                    np.array([[0.26, 0.32, 0.26], [-0.21, 0.45, -0.27]]),
                    (2, 2, 1, 1),  # shape (2, 2, 2, 3)
                ),
                np.array(
                    [
                        [[True, False], [True, False]],
                        [[True, False], [True, False]],
                    ]
                ),
                np.ones((2, 2, 2)) * np.array([0.63059263, -0.54370988]),
                np.ones((2, 2, 2)) * np.array([1.5076952, np.nan]),
                (2, 2),
                2,
                2,
            ),
        ],
    )
    def test_init(
        self,
        nickel_phase,
        hkl,
        hkl_detector,
        within_gnomonic_radius,
        hesse_distance,
        hesse_alpha,
        nav_shape,
        nav_dim,
        size,
    ):
        """Desired initialization behaviour."""
        in_pattern = np.ones(hkl_detector.shape[:-1], dtype=bool)
        gnomonic_radius = 10
        bands = KikuchiBand(
            phase=nickel_phase,
            hkl=hkl,
            hkl_detector=hkl_detector,
            in_pattern=in_pattern,
            gnomonic_radius=gnomonic_radius,
        )

        # Correct number of bands and navigation points and their shape
        assert bands.navigation_shape == nav_shape
        assert bands.navigation_dimension == nav_dim
        assert bands.size == size

        # No rounding or anything
        assert np.allclose(bands.hkl.data, hkl)
        assert np.allclose(bands.hkl_detector.data, hkl_detector)

        # Correct detector and gnomonic coordinates
        assert np.allclose(bands.x_detector, hkl_detector[..., 0])
        assert np.allclose(bands.y_detector, hkl_detector[..., 1])
        assert np.allclose(bands.z_detector, hkl_detector[..., 2])
        assert np.allclose(
            bands.x_gnomonic, hkl_detector[..., 0] / hkl_detector[..., 2]
        )
        assert np.allclose(
            bands.y_gnomonic, hkl_detector[..., 1] / hkl_detector[..., 2]
        )

        # Whether bands are in points are treated correctly
        assert bands.in_pattern.shape == in_pattern.shape
        assert np.allclose(bands.in_pattern, in_pattern)

        # Broadcasting of gnomonic radius is correct
        gnomonic_radius2 = np.atleast_1d(gnomonic_radius * np.ones(nav_shape))
        assert bands.gnomonic_radius.shape == gnomonic_radius2.shape
        assert np.allclose(bands.gnomonic_radius, gnomonic_radius2)

        # Whether a band should be plotted or not
        assert (
            bands.within_gnomonic_radius.shape == within_gnomonic_radius.shape
        )
        assert np.allclose(bands.within_gnomonic_radius, within_gnomonic_radius)

        # Hesse distance and alpha
        assert bands.hesse_distance.shape == hesse_distance.shape
        assert np.allclose(bands.hesse_distance, hesse_distance)
        assert bands.hesse_alpha.shape == hesse_alpha.shape
        assert np.allclose(bands.hesse_alpha, hesse_alpha, equal_nan=True)

    @pytest.mark.parametrize(
        "hkl, hkl_detector, gnomonic_radius, plane_trace_coordinates",
        [
            (
                np.array([-1, 1, 1]),
                np.array([0.26, 0.32, 0.26]),
                10,
                np.array([[7.3480, -8.1433, -6.7827, 5.8039]]),
            ),
            (
                np.array([[-1, 1, 1], [2, 0, 0]]),
                np.tile(
                    np.array([[0.26, 0.32, 0.26], [-0.21, 0.45, -0.27]]),
                    (2, 2, 1, 1),  # shape (2, 2, 2, 3)
                ),
                10,
                np.tile(
                    np.array(
                        [[7.3480, -8.1433, -6.7827, 5.8039], [np.nan] * 4]
                    ),
                    (2, 2, 1, 1),
                ),
            ),
            (
                np.array([[-1, 1, 1], [2, 0, 0]]),
                np.tile(
                    np.array([[0.26, 0.32, 0.26], [-0.21, 0.45, -0.27]]),
                    (5, 1, 1),  # shape (5, 2, 3)
                ),
                10,
                np.tile(
                    np.array(
                        [[7.3480, -8.1433, -6.7827, 5.8039], [np.nan] * 4]
                    ),
                    (5, 1, 1),
                ),
            ),
            (
                np.array([[-1, 1, 1], [2, 0, 0]]),
                np.tile(
                    np.array([[0.26, 0.32, 0.26], [-0.21, 0.45, -0.27]]),
                    (5, 1, 1, 1),  # shape (5, 1, 2, 3)
                ),
                10,
                np.tile(
                    np.array(
                        [[7.3480, -8.1433, -6.7827, 5.8039], [np.nan] * 4]
                    ),
                    (5, 1, 1, 1),  # shape (5, 1, 2, 3)
                ),
            ),
            (
                np.array([[-1, 1, 1], [2, 0, 0]]),
                np.tile(
                    np.array([[0.26, 0.32, 0.26], [-0.21, 0.45, -0.27]]),
                    (1, 5, 1, 1),  # shape (1, 5, 2, 3)
                ),
                10,
                np.tile(
                    np.array(
                        [[7.3480, -8.1433, -6.7827, 5.8039], [np.nan] * 4]
                    ),
                    (1, 5, 1, 1),  # shape (1, 5, 2, 3)
                ),
            ),
        ],
    )
    def test_plane_trace_coordinates(
        self,
        nickel_phase,
        hkl,
        hkl_detector,
        gnomonic_radius,
        plane_trace_coordinates,
    ):
        """Desired gnomonic (P1, P2) coordinates."""
        bands = KikuchiBand(
            phase=nickel_phase,
            hkl=hkl,
            hkl_detector=hkl_detector,
            in_pattern=np.ones(hkl_detector.shape[:-1], dtype=bool),
            gnomonic_radius=gnomonic_radius,
        )

        assert (
            bands.plane_trace_coordinates.shape == plane_trace_coordinates.shape
        )
        assert np.allclose(
            bands.plane_trace_coordinates,
            plane_trace_coordinates,
            atol=1e-4,
            equal_nan=True,
        )

    @pytest.mark.parametrize(
        "hkl, hkl_detector, dim_str, band_str",
        [
            (
                np.array([-1, 1, 1]),
                np.array([0.26, 0.32, 0.26]),
                "(|1)",
                "[[-1  1  1]]",
            ),
            (
                np.array([[-1, 1, 1], [2, 0, 0]]),
                np.tile(
                    np.array([[0.26, 0.32, 0.26], [-0.21, 0.45, -0.27]]),
                    (2, 2, 1, 1),  # shape (2, 2, 2, 3)
                ),
                "(2, 2|2)",
                "[[-1  1  1]\n [ 2  0  0]]",
            ),
        ],
    )
    def test_repr(self, nickel_phase, hkl, hkl_detector, dim_str, band_str):
        """Desired representation."""
        bands = KikuchiBand(
            phase=nickel_phase,
            hkl=hkl,
            hkl_detector=hkl_detector,
            in_pattern=np.ones(hkl_detector.shape[:-1], dtype=bool),
        )
        assert repr(bands) == (
            f"KikuchiBand {dim_str}\n"
            f"Phase: {nickel_phase.name} ({nickel_phase.point_group.name})\n"
            f"{band_str}"
        )

    @pytest.mark.parametrize(
        "nav_shape, n_bands, get_key, desired_data_shape_before, "
        "desired_data_shape_after",
        [
            ((2, 3), 9, (), (2, 3, 9), (2, 3, 9)),
            ((2, 3), 18, slice(None), (2, 3, 18), (2, 3, 18)),
            ((2, 3), 9, 0, (2, 3, 9), (3, 9)),
            ((2, 3), 9, (slice(0, 2), 0), (2, 3, 9), (2, 9)),
            ((2, 3), 9, (1, slice(None)), (2, 3, 9), (3, 9)),
            ((2, 3), 5, (slice(None), 2, slice(0, 5)), (2, 3, 5), (2, 5)),
        ],
    )
    def test_getitem(
        self,
        nickel_phase,
        nav_shape,
        n_bands,
        get_key,
        desired_data_shape_before,
        desired_data_shape_after,
    ):
        """Slicing works as expected."""
        hkl = np.repeat(np.arange(1, n_bands + 1), 3).reshape((n_bands, 3))
        hkl_detector = np.tile(
            np.repeat(np.arange(1, n_bands + 1), 3).reshape((n_bands, 3)),
            nav_shape + (1, 1),
        )  # shape nav_shape + (size, 3)
        bands = KikuchiBand(
            phase=nickel_phase,
            hkl=hkl,
            hkl_detector=hkl_detector,
            in_pattern=np.ones(nav_shape + (n_bands,), dtype=bool),
        )

        assert bands._data_shape == desired_data_shape_before

        new_bands = bands[get_key]
        assert new_bands._data_shape == desired_data_shape_after

    def test_hesse_lines(self, nickel_phase):
        """Desired Hesse lines."""
        full_shape = (2, 2, 2)
        bands = KikuchiBand(
            phase=nickel_phase,
            hkl=np.array([[-1, 1, 1], [0, -2, 0]]),
            hkl_detector=np.tile(
                np.array([[0.26, 0.32, 0.26], [-0.21, 0.45, -0.27]]),
                full_shape + (1, 1),
            ),
            in_pattern=np.ones(full_shape, dtype=bool),
        )

        assert np.allclose(
            bands.hesse_line_x,
            np.array(
                [
                    [[-0.39764706, -0.22992701], [-0.39764706, -0.22992701]],
                    [[-0.39764706, -0.22992701], [-0.39764706, -0.22992701]],
                ]
            ),
        )

        assert np.allclose(
            bands.hesse_line_y,
            np.array(
                [
                    [[-0.48941176, 0.49270073], [-0.48941176, 0.49270073]],
                    [[-0.48941176, 0.49270073], [-0.48941176, 0.49270073]],
                ]
            ),
        )

    def test_hesse_alpha(self):
        """Expected behaviour with varying shapes."""
        pass
