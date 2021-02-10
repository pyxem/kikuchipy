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

import matplotlib.pyplot as plt
import numpy as np
import pytest

from kikuchipy import data
from kikuchipy.detectors import PCCalibrationMovingScreen


POINTS_IN = [(109, 131), (390, 139), (246, 232), (129, 228), (364, 237)]
POINTS_OUT = [(77, 146), (424, 156), (246, 269), (104, 265), (392, 276)]
PX_SIZE = 46 / 508  # mm / px


@pytest.fixture
def moving_screen_cal_instance(request):
    return PCCalibrationMovingScreen(
        pattern_in=data.silicon_ebsd_moving_screen_in(allow_download=True),
        pattern_out=data.silicon_ebsd_moving_screen_out5mm(allow_download=True),
        points_in=POINTS_IN,
        points_out=POINTS_OUT,
        delta_z=5,  # mm
        px_size=PX_SIZE,
    )


class TestPCCalibrationMovingScreen:
    def test_points_lines(self, moving_screen_cal_instance):
        cal = moving_screen_cal_instance
        nrows, ncols = (480, 480)
        n_points = 5
        n_lines = n_points * 2

        assert cal.shape == (nrows, ncols)
        assert cal.nrows == nrows
        assert cal.ncols == ncols

        assert cal.n_points == n_points
        assert cal.points.shape == (2, n_points, 2)
        assert np.allclose(cal.points[0], POINTS_IN)
        assert np.allclose(cal.points[1], POINTS_OUT)

        assert cal.n_lines == n_lines
        assert cal.lines.shape == (2, n_lines, 4)
        assert cal.lines_out_in.shape == (n_points, 4)
        assert cal.lines_start.shape == (2, n_lines, 2)
        assert cal.lines_end.shape == (2, n_lines, 2)
        assert cal.lines_out_in.shape == (n_points, 4)
        assert np.allclose(cal.lines_out_in_start, POINTS_OUT)
        assert cal.lines_out_in_start.shape == (n_points, 2)
        assert np.allclose(cal.lines_out_in_end, POINTS_IN)
        assert cal.lines_out_in_end.shape == (n_points, 2)

    @pytest.mark.parametrize(
        "n_points, desired_pc",
        [
            (3, [0.5123, 0.8606, 0.4981]),
            (4, [0.5062, 0.8640, 0.5064]),
            (5, [0.5054, 0.8624, 0.5036]),
        ],
    )
    def test_pc(self, moving_screen_cal_instance, n_points, desired_pc):
        cal = moving_screen_cal_instance

        cal.points = cal.points[:, :n_points]
        cal.make_lines()

        assert cal.n_points == n_points
        assert np.allclose(cal.pc, desired_pc, atol=1e-4)

    def test_pc_convention(self, moving_screen_cal_instance):
        cal_tsl = moving_screen_cal_instance
        cal_bruker = PCCalibrationMovingScreen(
            pattern_in=cal_tsl.patterns[0],
            pattern_out=cal_tsl.patterns[1],
            points_in=cal_tsl.points[0],
            points_out=cal_tsl.points[1],
            delta_z=cal_tsl.delta_z,
            px_size=cal_tsl.px_size,
            convention="bruker",
        )

        assert np.allclose(cal_tsl.pc[1], 0.8624, atol=1e-4)
        assert np.allclose(cal_bruker.pc[1], 0.1376, atol=1e-4)

    def test_pc_no_px_size(self, moving_screen_cal_instance):
        """PCz in same unit as `delta_z`."""
        cal = moving_screen_cal_instance
        cal.px_size = None
        assert np.allclose(cal.pc[2], 21.8872, atol=1e-4)
        cal.px_size = PX_SIZE
        assert np.allclose(cal.pc[2], 0.5036, atol=1e-4)

    def test_plot(self, moving_screen_cal_instance):
        cal = moving_screen_cal_instance
        fig, ax = cal.plot(return_fig_ax=True)

        assert isinstance(fig, plt.Figure)
        assert len(ax) == 3
        assert isinstance(ax[0], plt.Subplot)

    def test_repr(self, moving_screen_cal_instance):
        assert repr(moving_screen_cal_instance) == (
            "PCCalibrationMovingScreen: (PCx, PCy, PCz) = "
            "(0.5054, 0.8624, 0.5036)\n"
            "5 points:\n"
            "[[[109 131]\n"
            "  [390 139]\n"
            "  [246 232]\n"
            "  [129 228]\n"
            "  [364 237]]\n"
            "\n"
            " [[ 77 146]\n"
            "  [424 156]\n"
            "  [246 269]\n"
            "  [104 265]\n"
            "  [392 276]]]"
        )
