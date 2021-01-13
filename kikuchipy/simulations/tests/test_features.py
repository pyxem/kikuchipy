# -*- coding: utf-8 -*-
# Copyright 2019-2021 The kikuchipy developers
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

from kikuchipy.crystallography._computations import _get_uvw_from_hkl
from kikuchipy.generators.ebsd_simulation_generator import (
    _get_coordinates_in_upper_hemisphere,
)
from kikuchipy.projections.ebsd_projections import detector2direct_lattice
from kikuchipy.simulations.features import KikuchiBand, ZoneAxis


class TestKikuchiBand:
    @pytest.mark.parametrize(
        "hkl, hkl_detector, within_gnomonic_radius, hesse_distance, "
        "hesse_alpha, nav_shape, nav_dim, size",
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

    def test_get_item(self, nickel_kikuchi_band):
        """KikuchiBand.__getitem__() works as desired."""
        bands = nickel_kikuchi_band
        nav_shape = bands.navigation_shape

        # Zero getitem keys
        assert np.allclose(bands[:].hkl_detector.data, bands.hkl_detector.data)
        assert bands[:].navigation_shape == nav_shape
        assert np.allclose(bands[()].hkl_detector.data, bands.hkl_detector.data)
        assert bands[()].navigation_shape == nav_shape
        assert np.allclose(
            bands[slice(None)].hkl_detector.data, bands.hkl_detector.data
        )
        assert bands[slice(None)].navigation_shape == nav_shape

        # One getitem key
        # All bands visible in first pattern
        assert np.allclose(
            bands[0].hkl_detector.data, bands.hkl_detector.data[0]
        )
        assert bands[0].navigation_shape == (5,)

        # Two getitem keys
        with pytest.raises(IndexError, match="Not enough axes to slice"):
            # Slicing shape () with two keys
            _ = bands[0, 0][0, 0]
        # Slicing ndim == 1 with two keys
        assert np.allclose(bands[0][0, :2].hkl.data, bands.hkl.data[:2])
        assert bands[0][0, :2].navigation_shape == ()

        # Three getitem keys
        with pytest.raises(IndexError, match="Not enough axes to slice"):
            # Slicing ndim == 1 with three keys
            _ = bands[0][0, :2, :3]
        this_slice = (0, slice(0, 2), slice(0, 3))
        bands2 = bands[this_slice]
        assert np.allclose(
            bands2.hkl_detector.data, bands.hkl_detector.data[this_slice]
        )
        assert bands2.size == 3
        assert bands[this_slice].navigation_shape == (2,)

    def test_get_item_structure_factor_theta(self, nickel_kikuchi_band):
        """Calculated structure factors and theta angles carry over."""
        bands = nickel_kikuchi_band
        v = 20e3
        bands.calculate_structure_factor(voltage=v)
        bands.calculate_theta(voltage=v)

        # All bands
        assert np.allclose(
            bands[2, 2:4].structure_factor, bands.structure_factor
        )
        # Some bands
        new_bands = bands[2, 2:4, 3:15]
        assert np.allclose(
            new_bands.structure_factor, bands.structure_factor[3:15]
        )
        assert np.allclose(new_bands.theta, bands.theta[3:15])

    def test_unique(self, nickel_kikuchi_band):
        with pytest.raises(NotImplementedError):
            _ = nickel_kikuchi_band.unique()

    def test_symmetrise(self, nickel_kikuchi_band):
        with pytest.raises(NotImplementedError):
            _ = nickel_kikuchi_band.symmetrise()

    def test_from_min_dspacing(self, nickel_kikuchi_band):
        with pytest.raises(NotImplementedError):
            _ = nickel_kikuchi_band.from_min_dspacing()

    def test_from_highest_hkl(self, nickel_kikuchi_band):
        with pytest.raises(NotImplementedError):
            _ = nickel_kikuchi_band.from_highest_hkl()


class TestZoneAxis:
    @pytest.mark.parametrize(
        "hkl_slices, desired_nav_shape, desired_nav_dims, desired_data_shape",
        [
            ((slice(0, 2), slice(0, 2), slice(None)), (2, 2), 2, (2, 2, 27)),
            ((slice(None), slice(None), slice(None)), (5, 5), 2, (5, 5, 35)),
            ((0, slice(0, 1), slice(None)), (1,), 1, (1, 25)),
            ((slice(0, 1), slice(1, 2), slice(None)), (1, 1), 2, (1, 1, 25)),
            ((0, 0, slice(None)), (), 0, (25,)),
        ],
    )
    def test_init(
        self,
        nickel_kikuchi_band,
        detector,
        nickel_rotations,
        hkl_slices,
        desired_nav_shape,
        desired_nav_dims,
        desired_data_shape,
    ):
        bands = nickel_kikuchi_band[hkl_slices]
        phase = bands.phase

        n_nav_dims = bands.navigation_dimension
        navigation_axes = (1, 2)[:n_nav_dims]

        rotations = nickel_rotations.reshape(5, 5)[hkl_slices[:2]]

        uvw = _get_uvw_from_hkl(bands.hkl.data)
        det2direct = detector2direct_lattice(
            sample_tilt=detector.sample_tilt,
            detector_tilt=detector.tilt,
            lattice=phase.structure.lattice,
            rotation=rotations,
        )
        uvw_detector = np.tensordot(uvw, det2direct, axes=(1, 0))
        if n_nav_dims == 0:
            uvw_detector = uvw_detector.squeeze()
        uvw_is_upper, uvw_in_a_pattern = _get_coordinates_in_upper_hemisphere(
            z_coordinates=uvw_detector[..., 2], navigation_axes=navigation_axes
        )
        uvw = uvw[uvw_in_a_pattern, ...]
        uvw_in_pattern = uvw_is_upper[uvw_in_a_pattern, ...].T
        uvw_detector = np.moveaxis(
            uvw_detector[uvw_in_a_pattern], source=0, destination=n_nav_dims
        )
        za = ZoneAxis(
            phase=phase,
            uvw=uvw,
            uvw_detector=uvw_detector,
            in_pattern=uvw_in_pattern,
            gnomonic_radius=detector.r_max,
        )

        assert za.navigation_shape == desired_nav_shape
        assert za.navigation_dimension == desired_nav_dims
        assert za._data_shape == desired_data_shape

        x_detector = uvw_detector[..., 0]
        y_detector = uvw_detector[..., 1]
        z_detector = uvw_detector[..., 2]
        assert np.allclose(za.x_detector, x_detector)
        assert np.allclose(za.y_detector, y_detector)
        assert np.allclose(za.z_detector, z_detector)

        with np.errstate(divide="ignore"):
            desired_x_gnomonic = x_detector / z_detector
            desired_y_gnomonic = y_detector / z_detector
        assert np.allclose(za.x_gnomonic, desired_x_gnomonic)
        assert np.allclose(za.y_gnomonic, desired_y_gnomonic)
        assert np.allclose(
            za.r_gnomonic,
            np.sqrt(desired_x_gnomonic ** 2 + desired_y_gnomonic ** 2),
        )

    @pytest.mark.parametrize(
        "uvw, uvw_detector, dim_str, za_str",
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
    def test_repr(self, nickel_phase, uvw, uvw_detector, dim_str, za_str):
        """Desired representation."""
        za = ZoneAxis(
            phase=nickel_phase,
            uvw=uvw,
            uvw_detector=uvw_detector,
            in_pattern=np.ones(uvw_detector.shape[:-1], dtype=bool),
        )
        assert repr(za) == (
            f"ZoneAxis {dim_str}\n"
            f"Phase: {nickel_phase.name} ({nickel_phase.point_group.name})\n"
            f"{za_str}"
        )

    @pytest.mark.parametrize(
        (
            "uvw_detector, uvw_in_pattern, gr, desired_data_shape, "
            "desired_within_gr"
        ),
        [
            # 2D, (ny, nx, n, xyz): (2, 2, 2, 3)
            (
                [
                    [[[1, 5, 1], [-1, -3, 4]], [[2, 4, -1], [0, -3, 4]]],
                    [[[1, 5, 1], [0, -3, 4]], [[1, 5, 1], [-1, -3, 4]]],
                ],
                [[[True, True], [True, True]], [[False, True], [True, True]]],
                1.92,
                (2, 2, 2),
                [
                    [[False, True], [False, True]],
                    [[False, True], [False, True]],
                ],
            ),
            # 1D, (nx, n, xyz): (1, 2, 3)
            (
                [[[-14, 11, 8], [-10, 9, 14]]],
                [[True, True]],
                1.92,
                (1, 2),
                [[False, True]],
            ),
            # 0D, (n, xyz): (2, 3)
            (
                [[-14, 11, 8], [-10, 9, 14]],
                [True, True],
                1.92,
                (2,),
                [False, True],
            ),
        ],
    )
    def test_within_gnomonic_radius(
        self,
        nickel_phase,
        uvw_detector,
        uvw_in_pattern,
        gr,
        desired_data_shape,
        desired_within_gr,
    ):
        """Gnomonic radius behaves for 2D, 1D and 0D data sets."""
        za = ZoneAxis(
            phase=nickel_phase,
            uvw=[[-1, 1, 0], [0, -1, 1]],
            uvw_detector=uvw_detector,
            in_pattern=uvw_in_pattern,
            gnomonic_radius=gr,
        )

        assert za._data_shape == desired_data_shape

        uvw_detector = np.asarray(uvw_detector, dtype=np.float32)
        desired_within_gr = np.asarray(desired_within_gr)

        assert np.allclose(za.within_gnomonic_radius, desired_within_gr)

        desired_xy = np.ones((za._data_shape + (2,))) * np.nan
        desired_xy[..., 0] = uvw_detector[..., 0] / uvw_detector[..., 2]
        desired_xy[..., 1] = uvw_detector[..., 1] / uvw_detector[..., 2]
        desired_xy[~desired_within_gr] = np.nan
        assert np.allclose(
            za._xy_within_gnomonic_radius, desired_xy, equal_nan=True
        )

    def test_get_item(self, nickel_zone_axes):
        """ZoneAxis.__getitem__() works as desired."""
        za = nickel_zone_axes
        nav_shape = za.navigation_shape

        # Zero getitem keys
        assert np.allclose(za[:].uvw_detector.data, za.uvw_detector.data)
        assert za[:].navigation_shape == nav_shape
        assert np.allclose(za[()].uvw_detector.data, za.uvw_detector.data)
        assert za[()].navigation_shape == nav_shape
        assert np.allclose(
            za[slice(None)].uvw_detector.data, za.uvw_detector.data
        )
        assert za[slice(None)].navigation_shape == nav_shape

        # One getitem key
        # All bands visible in first pattern
        assert np.allclose(za[0].uvw_detector.data, za.uvw_detector.data[0])
        assert za[0].navigation_shape == (5,)

        # Two getitem keys
        with pytest.raises(IndexError, match="Not enough axes to slice"):
            # Slicing shape () with two keys
            _ = za[0, 0][0, 0]
        # Slicing ndim == 1 with two keys
        assert np.allclose(za[0][0, :2].hkl.data, za.hkl.data[:2])
        assert za[0][0, :2].navigation_shape == ()

        # Three getitem keys
        with pytest.raises(IndexError, match="Not enough axes to slice"):
            # Slicing ndim == 1 with three keys
            _ = za[0][0, :2, :3]
        this_slice = (0, slice(0, 2), slice(0, 3))
        bands2 = za[this_slice]
        assert np.allclose(
            bands2.uvw_detector.data, za.uvw_detector.data[this_slice]
        )
        assert bands2.size == 3
        assert za[this_slice].navigation_shape == (2,)

    def test_get_item_structure_factor_theta(self, nickel_zone_axes):
        """Calculated structure factors and theta angles carry over."""
        za = nickel_zone_axes
        v = 20e3
        za.calculate_structure_factor(voltage=v)
        za.calculate_theta(voltage=v)

        # All bands
        assert np.allclose(za[2, 2:4].structure_factor, za.structure_factor)
        # Some bands
        new_bands = za[2, 2:4, 3:15]
        assert np.allclose(
            new_bands.structure_factor, za.structure_factor[3:15]
        )
        assert np.allclose(new_bands.theta, za.theta[3:15])

    def test_unique(self, nickel_zone_axes):
        with pytest.raises(NotImplementedError):
            _ = nickel_zone_axes.unique()

    def test_symmetrise(self, nickel_zone_axes):
        with pytest.raises(NotImplementedError):
            _ = nickel_zone_axes.symmetrise()

    def test_from_min_dspacing(self, nickel_zone_axes):
        with pytest.raises(NotImplementedError):
            _ = nickel_zone_axes.from_min_dspacing()

    def test_from_highest_hkl(self, nickel_zone_axes):
        with pytest.raises(NotImplementedError):
            _ = nickel_zone_axes.from_highest_hkl()
