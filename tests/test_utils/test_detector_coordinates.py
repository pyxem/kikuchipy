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

import numpy as np
import pytest

import kikuchipy as kp
from kikuchipy._utils._detector_coordinates import (
    _convert_coordinates,
    convert_coordinates,
    get_coordinate_conversions,
)


class TestDetectorCoordinates:
    s = kp.data.nickel_ebsd_small()
    det_1d = kp.detectors.EBSDDetector(shape=(60, 60), pc=s.detector.pc[0,])
    det_2d = s.detector.deepcopy()
    conv_1d = get_coordinate_conversions(det_1d.gnomonic_bounds, det_1d.bounds)
    conv_2d = get_coordinate_conversions(det_2d.gnomonic_bounds, det_2d.bounds)
    coords_5d = np.random.randint(0, 60, (3, 3, 17, 300, 2))
    coords_4d = coords_5d[:, :, 0]  # (3, 3, 300, 2)
    coords_3d = coords_5d[0, 0]  # (17, 300, 2)
    coords_2d = coords_5d[0, 0, 0]  # (300, 2)

    def test_get_conversion_factors(self):
        conv_1d = get_coordinate_conversions(
            self.det_1d.gnomonic_bounds, self.det_1d.bounds
        )

        exp_res_1d = {
            "pix_to_gn": {
                "m_x": np.array([0.03319923, 0.03326385, 0.03330547]),
                "c_x": np.array([-0.83957734, -0.84652344, -0.85204404]),
                "m_y": np.array([-0.03319923, -0.03326385, -0.03330547]),
                "c_y": np.array([0.42827701, 0.41940433, 0.42255835]),
            },
            "gn_to_pix": {
                "m_x": np.array([30.12118421, 30.06266362, 30.02509794]),
                "c_x": np.array([25.28906376, 25.4487495, 25.58270568]),
                "m_y": np.array([-30.12118421, -30.06266362, -30.02509794]),
                "c_y": np.array([12.90021062, 12.60841133, 12.6873559]),
            },
        }

        for i in ["pix_to_gn", "gn_to_pix"]:
            for j in ["m_x", "c_x", "m_y", "c_y"]:
                assert np.allclose(conv_1d[i][j], exp_res_1d[i][j])

    @pytest.mark.parametrize(
        "coords, detector_index, desired_coords",
        [
            (
                np.array([[36.2, 12.7]]),
                None,
                np.array(
                    [
                        [[[0.36223463, 0.00664684]], [[0.357628, -0.00304659]]],
                        [[[0.36432453, 0.00973462]], [[0.35219232, 0.00567801]]],
                    ]
                ),
            ),
            (
                np.array([[36.2, 12.7], [2.5, 43.7], [8.2, 27.7]]),
                (0, 1),
                np.array(
                    [
                        [0.35762801, -0.00304659],
                        [-0.76336381, -1.03422601],
                        [-0.57375985, -0.50200438],
                    ]
                ),
            ),
        ],
    )
    def test_coordinate_conversions_correct(
        self, coords, detector_index, desired_coords
    ):
        """Coordinate conversion factors have expected values."""
        pc = np.array(
            [
                [
                    [0.4214844, 0.21500351, 0.50201974],
                    [0.42414583, 0.21014019, 0.50104439],
                ],
                [
                    [0.42088203, 0.2165417, 0.50079336],
                    [0.42725023, 0.21450546, 0.49996293],
                ],
            ]
        )
        det = kp.detectors.EBSDDetector(shape=(60, 60), pc=pc)
        conv = get_coordinate_conversions(det.gnomonic_bounds, det.bounds)
        cds_out = convert_coordinates(coords, "pix_to_gn", conv, detector_index)
        cds_back = convert_coordinates(cds_out, "gn_to_pix", conv, detector_index)
        assert np.allclose(cds_out, desired_coords)
        assert np.allclose(cds_back[..., :], coords[..., :])

    @pytest.mark.parametrize(
        "coords, detector_index, desired_coords",
        [
            (
                np.array([[36.2, 12.7]]),
                None,
                np.array(
                    [
                        [[[0.36223463, 0.00664684]], [[0.357628, -0.00304659]]],
                        [[[0.36432453, 0.00973462]], [[0.35219232, 0.00567801]]],
                    ]
                ),
            ),
            (
                np.array([[36.2, 12.7], [2.5, 43.7], [8.2, 27.7]]),
                (0, 1),
                np.array(
                    [
                        [0.35762801, -0.00304659],
                        [-0.76336381, -1.03422601],
                        [-0.57375985, -0.50200438],
                    ]
                ),
            ),
        ],
    )
    def test_coordinate_conversions_correct_worker(
        self, coords, detector_index, desired_coords
    ):
        """Worker function coordinate conversion factors have expected values."""
        pc = np.array(
            [
                [
                    [0.4214844, 0.21500351, 0.50201974],
                    [0.42414583, 0.21014019, 0.50104439],
                ],
                [
                    [0.42088203, 0.2165417, 0.50079336],
                    [0.42725023, 0.21450546, 0.49996293],
                ],
            ]
        )
        det = kp.detectors.EBSDDetector(shape=(60, 60), pc=pc)
        conv = get_coordinate_conversions(det.gnomonic_bounds, det.bounds)
        nav_shape = conv["pix_to_gn"]["m_x"].shape
        nav_ndim = len(nav_shape)

        if isinstance(detector_index, type(None)):
            detector_index = ()
            if coords.ndim >= nav_ndim + 2 and coords.shape[:nav_ndim] == nav_shape:
                # one or more sets of coords, different for each image
                out_shape = coords.shape
            else:
                # one or more sets of coords, the same for each image
                out_shape = nav_shape + coords.shape

            extra_axes = list(range(nav_ndim, len(out_shape) - 1))
            cds_out = _convert_coordinates(
                coords,
                out_shape,
                detector_index,
                np.expand_dims(conv["pix_to_gn"]["m_x"], extra_axes),
                np.expand_dims(conv["pix_to_gn"]["c_x"], extra_axes),
                np.expand_dims(conv["pix_to_gn"]["m_y"], extra_axes),
                np.expand_dims(conv["pix_to_gn"]["c_y"], extra_axes),
            )

            cds_back = _convert_coordinates(
                cds_out,
                out_shape,
                detector_index,
                np.expand_dims(conv["gn_to_pix"]["m_x"], extra_axes),
                np.expand_dims(conv["gn_to_pix"]["c_x"], extra_axes),
                np.expand_dims(conv["gn_to_pix"]["m_y"], extra_axes),
                np.expand_dims(conv["gn_to_pix"]["c_y"], extra_axes),
            )

        else:
            out_shape = coords.shape

            cds_out = _convert_coordinates(
                coords,
                out_shape,
                detector_index,
                conv["pix_to_gn"]["m_x"],
                conv["pix_to_gn"]["c_x"],
                conv["pix_to_gn"]["m_y"],
                conv["pix_to_gn"]["c_y"],
            )

            cds_back = _convert_coordinates(
                cds_out,
                out_shape,
                detector_index,
                conv["gn_to_pix"]["m_x"],
                conv["gn_to_pix"]["c_x"],
                conv["gn_to_pix"]["m_y"],
                conv["gn_to_pix"]["c_y"],
            )

        assert np.allclose(cds_out, desired_coords)
        assert np.allclose(cds_back[..., :], coords[..., :])

    def test_coordinate_conversions_indexing(self):
        """Converting from pixel to gnomonic coords and back."""
        # conversions
        cds_out5d_1 = convert_coordinates(
            self.coords_5d, "pix_to_gn", self.conv_2d, None
        )
        cds_out5d_2 = convert_coordinates(
            self.coords_5d, "pix_to_gn", self.conv_2d, (1, 2)
        )

        cds_out4d_1 = convert_coordinates(
            self.coords_4d, "pix_to_gn", self.conv_2d, None
        )
        cds_out4d_2 = convert_coordinates(
            self.coords_4d, "pix_to_gn", self.conv_2d, (1, 2)
        )

        cds_out3d_1 = convert_coordinates(
            self.coords_3d, "pix_to_gn", self.conv_2d, None
        )
        cds_out3d_2 = convert_coordinates(
            self.coords_3d, "pix_to_gn", self.conv_2d, (1, 2)
        )

        cds_out2d_1 = convert_coordinates(
            self.coords_2d, "pix_to_gn", self.conv_2d, None
        )
        cds_out2d_2 = convert_coordinates(
            self.coords_2d, "pix_to_gn", self.conv_2d, (1, 2)
        )

        cds_out5d_3 = convert_coordinates(
            self.coords_5d, "pix_to_gn", self.conv_1d, None
        )
        cds_out5d_4 = convert_coordinates(
            self.coords_5d, "pix_to_gn", self.conv_1d, (1)
        )
        cds_out5d_5 = convert_coordinates(self.coords_5d, "pix_to_gn", self.conv_1d, 1)

        cds_out4d_3 = convert_coordinates(
            self.coords_4d, "pix_to_gn", self.conv_1d, None
        )
        cds_out4d_4 = convert_coordinates(
            self.coords_4d, "pix_to_gn", self.conv_1d, (1,)
        )
        cds_out4d_5 = convert_coordinates(self.coords_4d, "pix_to_gn", self.conv_1d, 1)

        cds_out3d_3 = convert_coordinates(
            self.coords_3d, "pix_to_gn", self.conv_1d, None
        )
        cds_out3d_4 = convert_coordinates(
            self.coords_3d, "pix_to_gn", self.conv_1d, (1,)
        )
        cds_out3d_5 = convert_coordinates(self.coords_3d, "pix_to_gn", self.conv_1d, 1)

        cds_out2d_3 = convert_coordinates(
            self.coords_2d, "pix_to_gn", self.conv_1d, None
        )
        cds_out2d_4 = convert_coordinates(
            self.coords_2d, "pix_to_gn", self.conv_1d, (1,)
        )
        cds_out2d_5 = convert_coordinates(self.coords_2d, "pix_to_gn", self.conv_1d, 1)

        # convert back
        cds_back_5d_1 = convert_coordinates(
            cds_out5d_1, "gn_to_pix", self.conv_2d, None
        )
        cds_back_4d_1 = convert_coordinates(
            cds_out4d_1, "gn_to_pix", self.conv_2d, None
        )
        cds_back_3d_1 = convert_coordinates(
            cds_out3d_1, "gn_to_pix", self.conv_2d, None
        )
        cds_back_2d_1 = convert_coordinates(
            cds_out2d_1, "gn_to_pix", self.conv_2d, None
        )

        # indexing checks
        assert np.allclose(cds_out5d_2[1, 2], cds_out5d_1[1, 2])
        assert np.allclose(cds_out4d_2[1, 2], cds_out4d_1[1, 2])
        assert np.allclose(cds_out3d_2, cds_out3d_1[1, 2])
        assert np.allclose(cds_out2d_2, cds_out2d_1[1, 2])

        assert np.allclose(cds_out5d_4[1], cds_out5d_3[1])
        assert np.allclose(cds_out5d_5, cds_out5d_4)
        assert np.allclose(cds_out4d_4[1], cds_out4d_3[1])
        assert np.allclose(cds_out4d_5, cds_out4d_4)
        assert np.allclose(cds_out3d_4, cds_out3d_3[1])
        assert np.allclose(cds_out3d_5, cds_out3d_4)
        assert np.allclose(cds_out2d_4, cds_out2d_3[1])
        assert np.allclose(cds_out2d_5, cds_out2d_4)

        # back-conversion checks
        assert np.allclose(cds_back_5d_1, self.coords_5d)
        assert np.allclose(cds_back_4d_1, self.coords_4d)
        assert np.allclose(cds_back_3d_1, self.coords_3d)
        assert np.allclose(cds_back_2d_1, self.coords_2d)

        for i in range(3):
            for j in range(3):
                q = convert_coordinates(
                    self.coords_5d[i, j], "pix_to_gn", self.conv_2d, (i, j)
                )
                assert np.allclose(q, cds_out5d_1[i, j])
