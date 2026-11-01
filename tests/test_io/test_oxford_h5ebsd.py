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

import h5py
import numpy as np
from orix.quaternion import Rotation
import pytest

import kikuchipy as kp
from kikuchipy.detectors._detector_angles import (
    _DEFAULT_DETECTOR_MATRIX,
    _sample_to_detector_matrix,
)
from kikuchipy.io.plugins.oxford_h5ebsd._api import get_binning


class TestOxfordH5EBSD:
    @pytest.mark.parametrize(
        "oxford_h5ebsd_file", ["7.0", "6.0"], indirect=["oxford_h5ebsd_file"]
    )
    def test_load_oxford_h5ebsd(
        self, oxford_h5ebsd_file, ni_small_axes_manager, assert_dictionary_func
    ):
        s = kp.load(oxford_h5ebsd_file)
        assert s.data.shape == (3, 3, 60, 60)
        assert_dictionary_func(s.axes_manager.as_dictionary(), ni_small_axes_manager)
        assert s.metadata.Acquisition_instrument.SEM.beam_energy == 20

        s2 = kp.data.nickel_ebsd_small()
        s2.remove_static_background()
        assert np.allclose(s.data, s2.data)
        assert np.allclose(s.static_background, s2.static_background)

        # Detector
        det = s.detector
        assert det.pc.shape == (3, 3, 3)
        assert det.binning == 17.0
        assert np.isclose(det.sample_tilt, 69.9, atol=0.1)
        assert np.isclose(det.tilt, 1.5)
        assert np.isclose(det.azimuthal, 0)
        assert np.isclose(det.twist, 0)

    @pytest.mark.parametrize(
        "euler, desired_angles",
        [
            ([0.35, 94, -0.20], [4.0001, -0.3491, 0.2244]),
            ([0.128, 100.125, 0.825], [10.1250, -0.1260, -0.8025]),
        ],
    )
    def test_load_detector_orientation(self, oxford_h5ebsd_file, euler, desired_angles):
        """Detector tilt, azimuthal, and twist are derived from the
        detector orientation so that the projection reproduces it.
        """
        with h5py.File(oxford_h5ebsd_file, mode="r+") as f:
            header = f["1/EBSD/Header"]
            del header["Detector Orientation Euler"]
            header.create_dataset("Detector Orientation Euler", data=np.deg2rad(euler))

        det = kp.load(oxford_h5ebsd_file).detector
        assert np.allclose(
            [det.tilt, det.azimuthal, det.twist], desired_angles, atol=1e-4
        )

        sigma, theta, omega, gamma = np.deg2rad(
            [det.sample_tilt, det.tilt, det.azimuthal, det.twist]
        )
        m = _sample_to_detector_matrix(sigma, theta, omega, gamma)
        m0 = _sample_to_detector_matrix(sigma, 0.0, 0.0, 0.0)
        g_proj = m @ m0.T @ _DEFAULT_DETECTOR_MATRIX
        g_oxford = Rotation.from_euler(np.deg2rad(euler)).to_matrix().squeeze()
        assert np.allclose(g_proj, g_oxford, atol=1e-12)

    def test_load_detector_orientation_tilt_axis_not_zero(self, oxford_h5ebsd_file):
        with h5py.File(oxford_h5ebsd_file, mode="r+") as f:
            header = f["1/EBSD/Header"]
            del header["Detector Orientation Euler"]
            header.create_dataset(
                "Detector Orientation Euler", data=np.deg2rad([0.35, 94, -0.20])
            )
            header.create_dataset("Tilt Axis", data=np.pi / 2)

        with pytest.warns(UserWarning, match="Sample tilt axis 90.0 deg is not 0 deg"):
            det = kp.load(oxford_h5ebsd_file).detector
        assert np.isclose(det.tilt, 4)
        assert det.azimuthal == 0
        assert det.twist == 0

    def test_load_unprocessed_patterns(self, oxford_h5ebsd_file):
        s1 = kp.load(oxford_h5ebsd_file)
        s2 = kp.load(oxford_h5ebsd_file, processed=False)
        assert np.allclose(s1.data + 1, s2.data)

    @pytest.mark.parametrize(
        ["version", "header_group", "binning"],
        [
            ("5.0", {"Camera Binning Mode": "8x8 (168x128 px)"}, 8),
            ("6.0", {"Camera Binning Mode": "Resolution (1244x1024 px)"}, 1),
            ("6.0", {"Camera Binning Mode": "Speed 3 (156x88 px)"}, 8),
            ("7.0", {"Camera Mode": "Sensitivity (622x512 px)"}, 2),
        ],
    )
    def test_get_binning(self, version, header_group, binning):
        assert get_binning(header_group, version) == binning
