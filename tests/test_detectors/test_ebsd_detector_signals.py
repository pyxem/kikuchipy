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

import pytest

_ = pytest.importorskip("psygnal", reason="psygnal is not installed")

import numpy as np
import psygnal.testing as pt

import kikuchipy as kp


class TestEBSDDetectorSignals:
    def test_sample_tilt_signal_emitted(self):
        det = kp.detectors.EBSDDetector()
        assert np.isclose(det.sample_tilt, 70)
        with pt.assert_emitted(det._sample_tilt_changed):
            det.sample_tilt += 1
        assert np.isclose(det.sample_tilt, 71)

    def test_tilt_signal_emitted(self):
        det = kp.detectors.EBSDDetector()
        assert np.isclose(det.tilt, 0)
        with pt.assert_emitted(det._tilt_changed):
            det.tilt += 1
        assert np.isclose(det.tilt, 1)

    def test_azimuthal_signal_emitted(self):
        det = kp.detectors.EBSDDetector()
        assert np.isclose(det.azimuthal, 0)
        with pt.assert_emitted(det._azimuthal_changed):
            det.azimuthal += 1
        assert np.isclose(det.azimuthal, 1)

    def test_twist_signal_emitted(self):
        det = kp.detectors.EBSDDetector()
        assert np.isclose(det.twist, 0)
        with pt.assert_emitted(det._twist_changed):
            det.twist += 1
        assert np.isclose(det.twist, 1)

    def test_pc_signal_emitted(self):
        det1 = kp.detectors.EBSDDetector()
        assert np.allclose(det1.pc, 0.5)

        with pt.assert_emitted(det1._pc_changed):
            det1.pcx += 0.1
        assert np.allclose(det1.pc, [0.6, 0.5, 0.5])

        with pt.assert_emitted(det1._pc_changed):
            det1.pcy += 0.1
        assert np.allclose(det1.pc, [0.6, 0.6, 0.5])

        with pt.assert_emitted(det1._pc_changed):
            det1.pcz += 0.1
        assert np.allclose(det1.pc, [0.6, 0.6, 0.6])

        with pt.assert_emitted(det1._pc_changed):
            det1.pc += 0.1
        assert np.allclose(det1.pc, 0.7)

        det2 = kp.detectors.EBSDDetector(pc=np.full((2, 3, 3), 0.5))
        assert det2.navigation_shape == (2, 3)
        with pt.assert_emitted(det2._pc_changed):
            det2.navigation_shape = (3, 2)
        assert det2.navigation_shape == (3, 2)
