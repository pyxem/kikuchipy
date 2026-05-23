#
# Copyright 2019-2026 the kikuchipy developers
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
#

import numpy as np

import kikuchipy as kp
from kikuchipy.detectors._convert_detector_coordinates import (
    convert_coordinates,
    get_coordinate_conversion,
)


class TestDetectorCoordinates:
    s = kp.data.nickel_ebsd_small()
    det_1d = kp.detectors.EBSDDetector(shape=(60, 60), pc=s.detector.pc[0,])
    det_2d = s.detector.deepcopy()
    conv_1d_pix_to_gn = get_coordinate_conversion(
        det_1d.gnomonic_bounds, det_1d.bounds, "pix_to_gn"
    )
    conv_1d_gn_to_pix = get_coordinate_conversion(
        det_1d.gnomonic_bounds, det_1d.bounds, "gn_to_pix"
    )
    conv_2d_pix_to_gn = get_coordinate_conversion(
        det_2d.gnomonic_bounds, det_2d.bounds, "pix_to_gn"
    )
    conv_2d_gn_to_pix = get_coordinate_conversion(
        det_2d.gnomonic_bounds, det_2d.bounds, "gn_to_pix"
    )
    coords_5d = np.random.randint(0, 60, (3, 3, 17, 300, 2))
    coords_4d = coords_5d[:, :, 0]  # (3, 3, 300, 2)
    coords_3d = coords_5d[0, 0]  # (17, 300, 2)
    coords_2d = coords_5d[0, 0, 0]  # (300, 2)

    def test_get_conversion_factors(self):
        conv_1d_pix_to_gn = get_coordinate_conversion(
            self.det_1d.gnomonic_bounds, self.det_1d.bounds, "pix_to_gn"
        )
        conv_1d_gn_to_pix = get_coordinate_conversion(
            self.det_1d.gnomonic_bounds, self.det_1d.bounds, "gn_to_pix"
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

        for j in ["m_x", "c_x", "m_y", "c_y"]:
            assert np.allclose(conv_1d_pix_to_gn[j], exp_res_1d["pix_to_gn"][j])
            assert np.allclose(conv_1d_gn_to_pix[j], exp_res_1d["gn_to_pix"][j])

    def test_coordinate_conversions_indexing(self):
        """Converting from pixel to gnomonic coords and back."""
        # conversions
        cds_out5d_1 = convert_coordinates(self.coords_5d, self.conv_2d_pix_to_gn)
        cds_out5d_2 = convert_coordinates(
            self.coords_5d, self.conv_2d_pix_to_gn, (1, 2)
        )

        cds_out4d_1 = convert_coordinates(self.coords_4d, self.conv_2d_pix_to_gn)
        cds_out4d_2 = convert_coordinates(
            self.coords_4d, self.conv_2d_pix_to_gn, (1, 2)
        )

        cds_out3d_1 = convert_coordinates(self.coords_3d, self.conv_2d_pix_to_gn)
        cds_out3d_2 = convert_coordinates(
            self.coords_3d, self.conv_2d_pix_to_gn, (1, 2)
        )

        cds_out2d_1 = convert_coordinates(self.coords_2d, self.conv_2d_pix_to_gn)
        cds_out2d_2 = convert_coordinates(
            self.coords_2d, self.conv_2d_pix_to_gn, (1, 2)
        )

        cds_out5d_3 = convert_coordinates(self.coords_5d, self.conv_1d_pix_to_gn)
        cds_out5d_4 = convert_coordinates(self.coords_5d, self.conv_1d_pix_to_gn, (1))
        cds_out5d_5 = convert_coordinates(self.coords_5d, self.conv_1d_pix_to_gn, 1)

        cds_out4d_3 = convert_coordinates(self.coords_4d, self.conv_1d_pix_to_gn)
        cds_out4d_4 = convert_coordinates(self.coords_4d, self.conv_1d_pix_to_gn, (1,))
        cds_out4d_5 = convert_coordinates(self.coords_4d, self.conv_1d_pix_to_gn, 1)

        cds_out3d_3 = convert_coordinates(self.coords_3d, self.conv_1d_pix_to_gn)
        cds_out3d_4 = convert_coordinates(self.coords_3d, self.conv_1d_pix_to_gn, (1,))
        cds_out3d_5 = convert_coordinates(self.coords_3d, self.conv_1d_pix_to_gn, 1)

        cds_out2d_3 = convert_coordinates(self.coords_2d, self.conv_1d_pix_to_gn)
        cds_out2d_4 = convert_coordinates(self.coords_2d, self.conv_1d_pix_to_gn, (1,))
        cds_out2d_5 = convert_coordinates(self.coords_2d, self.conv_1d_pix_to_gn, 1)

        # convert back
        cds_back_5d_1 = convert_coordinates(cds_out5d_1, self.conv_2d_gn_to_pix)
        cds_back_4d_1 = convert_coordinates(cds_out4d_1, self.conv_2d_gn_to_pix)
        cds_back_3d_1 = convert_coordinates(cds_out3d_1, self.conv_2d_gn_to_pix)
        cds_back_2d_1 = convert_coordinates(cds_out2d_1, self.conv_2d_gn_to_pix)

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
                    self.coords_5d[i, j], self.conv_2d_pix_to_gn, (i, j)
                )
                assert np.allclose(q, cds_out5d_1[i, j])
