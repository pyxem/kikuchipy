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

# Many of these tests are inspired by the tests written for the block_file
# reader/writer available in HyperSpy: https://github.com/hyperspy/hyperspy/
# blob/RELEASE_next_minor/hyperspy/tests/io/test_blockfile.py

import datetime
import gc
import os
import tempfile
import time

import dask.array as da
from matplotlib.pyplot import imread
import numpy as np
import pytest

from kikuchipy.io._io import load
from kikuchipy.io.plugins.nordif import get_settings_from_file, get_string
from kikuchipy.signals.ebsd import EBSD

DIR_PATH = os.path.dirname(__file__)
NORDIF_PATH = os.path.join(DIR_PATH, "../../../data/nordif")
PATTERN_FILE = os.path.join(NORDIF_PATH, "Pattern.dat")
SETTING_FILE = os.path.join(NORDIF_PATH, "Setting.txt")
BG_FILE = os.path.join(NORDIF_PATH, "Background acquisition pattern.bmp")

# Settings content
METADATA = {
    "Acquisition_instrument": {
        "SEM": {
            "microscope": "Hitachi SU-6600",
            "magnification": 200,
            "beam_energy": 20.0,
            "working_distance": 24.7,
            "Detector": {
                "EBSD": {
                    "azimuth_angle": 0.0,
                    "binning": 1,
                    "detector": "NORDIF UF1100",
                    "elevation_angle": 0.0,
                    "exposure_time": 0.0035,
                    "frame_number": -1,
                    "frame_rate": 202,
                    "gain": 0.0,
                    "grid_type": "square",
                    "sample_tilt": 70.0,
                    "scan_time": 148,
                    "static_background": -1,
                    "xpc": -1.0,
                    "ypc": -1.0,
                    "zpc": -1.0,
                    "version": "3.1.2",
                    "manufacturer": "NORDIF",
                }
            },
        }
    },
    "Sample": {
        "Phases": {
            "1": {
                "atom_coordinates": {
                    "1": {
                        "atom": "",
                        "coordinates": np.array([0.0, 0.0, 0.0]),
                        "site_occupation": 0.0,
                        "debye_waller_factor": 0.0,
                    }
                },
                "formula": "",
                "info": "",
                "lattice_constants": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                "laue_group": "",
                "material_name": "Ni",
                "point_group": "",
                "setting": 0,
                "source": "",
                "space_group": 0,
                "symmetry": 0,
            }
        }
    },
}
ORIGINAL_METADATA = {
    "nordif_header": [
        "[NORDIF]\t\t",
        "Software version\t3.1.2\t",
        "\t\t",
        "[Microscope]\t\t",
        "Manufacturer\tHitachi\t",
        "Model\tSU-6600\t",
        "Magnification\t200\t#",
        "Scan direction\tDirect\t",
        "Accelerating voltage\t20\tkV",
        "Working distance\t24.7\tmm",
        "Tilt angle\t70\t°",
        "\t\t",
        "[Signal voltages]\t\t",
        "Minimum\t0.0\tV",
        "Maximum\t1.0\tV",
        "\t\t",
        "[Deflection voltages]\t\t",
        "Minimum\t-5.5\tV",
        "Maximum\t5.5\tV",
        "\t\t",
        "[Electron image]\t\t",
        "Frame rate\t0.25\tfps",
        "Resolution\t1000x1000\tpx",
        "Rotation\t0\t°",
        "Flip x-axis\tFalse\t",
        "Flip y-axis\tFalse\t",
        "Calibration factor\t7273\tµm/V",
        "Tilt axis\tx-axis\t",
        "\t\t",
        "[Aspect ratio]\t\t",
        "X-axis\t1.000\t",
        "Y-axis\t1.000\t",
        "\t\t",
        "[EBSD detector]\t\t",
        "Model\tUF1100\t",
        "Port position\t90\t",
        "Jumbo frames\tFalse\t",
        "\t\t",
        "[Detector angles]\t\t",
        "Euler 1\t0\t°",
        "Euler 2\t0\t°",
        "Euler 3\t0\t°",
        "Azimuthal\t0\t°",
        "Elevation\t0\t°",
        "\t\t",
        "[Acquisition settings]\t\t",
        "Frame rate\t202\tfps",
        "Resolution\t60x60\tpx",
        "Exposure time\t3500\tµs",
        "Gain\t0\t",
        "\t\t",
        "[Calibration settings]\t\t",
        "Frame rate\t10\tfps",
        "Resolution\t480x480\tpx",
        "Exposure time\t99950\tµs",
        "Gain\t8\t",
        "\t\t",
        "[Specimen]\t\t",
        "Name\tNi\t",
        "Mounting\t1. ND||EB TD||TA\t",
        "\t\t",
        "[Phase 1]\t\t",
        "Name\t\t",
        "Pearson S.T.\t\t",
        "IT\t\t",
        "\t\t",
        "[Phase 2]\t\t",
        "Name\t\t",
        "Pearson S.T.\t\t",
        "IT\t\t",
        "\t\t",
        "[Region of interest]\t\t",
        "\t\t",
        "[Area]\t\t",
        "Top\t89.200 (223)\tµm (px)",
        "Left\t60.384 (152)\tµm (px)",
        "Width\t4.500 (11)\tµm (px)",
        "Height\t4.500 (11)\tµm (px)",
        "Step size\t1.500\tµm",
        "Number of samples\t3x3\t#",
        "Scan time\t00:02:28\t",
        "\t\t",
        "[Points of interest]\t\t",
        "\t\t",
        "[Acquisition patterns]\t\t",
        "Acquisition (507,500)\t507,500\tpx",
        "Acquisition (393,501)\t393,501\tpx",
        "Acquisition (440,448)\t440,448\tpx",
        "\t\t",
        "[Calibration patterns]\t\t",
        "Calibration (425,447)\t425,447\tpx",
        "Calibration (294,532)\t294,532\tpx",
        "Calibration (573,543)\t573,543\tpx",
        "Calibration (596,378)\t596,378\tpx",
        "Calibration (308,369)\t308,369\tpx",
        "Calibration (171,632)\t171,632\tpx",
        "Calibration (704,668)\t704,668\tpx",
        "Calibration (696,269)\t696,269\tpx",
        "Calibration (152,247)\t152,247\tpx",
    ]
}
SCAN_SIZE_FILE = {
    "nx": 3,
    "ny": 3,
    "sx": 60,
    "sy": 60,
    "step_x": 1.5,
    "step_y": 1.5,
}
AXES_MANAGER = {
    "axis-0": {
        "name": "y",
        "scale": 1.5,
        "offset": 0.0,
        "size": 3,
        "units": "um",
        "navigate": True,
    },
    "axis-1": {
        "name": "x",
        "scale": 1.5,
        "offset": 0.0,
        "size": 3,
        "units": "um",
        "navigate": True,
    },
    "axis-2": {
        "name": "dy",
        "scale": 1.0,
        "offset": 0.0,
        "size": 60,
        "units": "um",
        "navigate": False,
    },
    "axis-3": {
        "name": "dx",
        "scale": 1.0,
        "offset": 0.0,
        "size": 60,
        "units": "um",
        "navigate": False,
    },
}


@pytest.fixture()
def save_path_nordif():
    with tempfile.TemporaryDirectory() as tmp:
        file_path = os.path.join(tmp, "nordif", "save_temp.dat")
        yield file_path
        gc.collect()


class TestNORDIF:
    def test_get_settings_from_file(self):
        settings = get_settings_from_file(SETTING_FILE)

        answers = [METADATA, ORIGINAL_METADATA, SCAN_SIZE_FILE]
        assert len(settings) == len(answers)
        for setting_read, answer in zip(settings, answers):
            np.testing.assert_equal(setting_read.as_dictionary(), answer)

    @pytest.mark.parametrize("line_no, correct", [(10, True), (11, False)])
    def test_get_string(self, line_no, correct):
        f = open(SETTING_FILE, "r", encoding="latin-1")
        content = f.read().splitlines()
        exp = "Tilt angle\t(.*)\t"
        if correct:
            sample_tilt = get_string(
                content=content, expression=exp, line_no=line_no, file=f
            )
            assert sample_tilt == str(70)
        else:
            with pytest.warns(UserWarning):
                sample_tilt = get_string(
                    content=content, expression=exp, line_no=line_no, file=f
                )
            assert sample_tilt == 0

    @pytest.mark.parametrize("setting_file", (None, SETTING_FILE))
    def test_load(self, setting_file):
        s = load(PATTERN_FILE, setting_file=SETTING_FILE)

        assert s.data.shape == (3, 3, 60, 60)
        assert s.axes_manager.as_dictionary() == AXES_MANAGER

        static_bg = imread(BG_FILE)
        assert np.allclose(
            s.metadata.Acquisition_instrument.SEM.Detector.EBSD.static_background,
            static_bg,
        )

    @pytest.mark.parametrize(
        "nav_shape, sig_shape",
        [((3, 3), (60, 60)), ((3, 4), (60, 60)), (None, None)],
    )
    def test_load_parameters(self, nav_shape, sig_shape):
        if nav_shape is None and sig_shape is None:
            with pytest.raises(ValueError):
                _ = load(
                    PATTERN_FILE,
                    setting_file="Setting.txt",
                    scan_size=nav_shape,
                    pattern_size=sig_shape,
                )
        else:
            if sum(nav_shape + sig_shape) > 126:
                # Check if zero padding user warning is raised if sum of data
                # shape is bigger than file size
                with pytest.warns(UserWarning):
                    s = load(
                        PATTERN_FILE,
                        scan_size=nav_shape,
                        pattern_size=sig_shape,
                    )
            else:
                s = load(
                    PATTERN_FILE, scan_size=nav_shape, pattern_size=sig_shape
                )
            assert s.data.shape == nav_shape[::-1] + sig_shape

    def test_load_save_cycle(self, save_path_nordif):
        s = load(PATTERN_FILE)

        scan_time_string = s.original_metadata["nordif_header"][80][10:18]
        scan_time = time.strptime(scan_time_string, "%H:%M:%S")
        scan_time = datetime.timedelta(
            hours=scan_time.tm_hour,
            minutes=scan_time.tm_min,
            seconds=scan_time.tm_sec,
        ).total_seconds()
        assert (
            s.metadata.Acquisition_instrument.SEM.Detector.EBSD.scan_time
            == scan_time
        )
        assert s.metadata.General.title == "Pattern"

        s.save(save_path_nordif, overwrite=True)
        with pytest.warns(UserWarning):  # No background pattern in directory
            s_reload = load(save_path_nordif, setting_file=SETTING_FILE)

        assert np.allclose(s.data, s_reload.data)

        # Add static background and change filename to make metadata equal
        s_reload.metadata.Acquisition_instrument.SEM.Detector.EBSD.static_background = imread(
            BG_FILE
        )
        s_reload.metadata.General.original_filename = (
            s.metadata.General.original_filename
        )
        s_reload.metadata.General.title = s.metadata.General.title
        np.testing.assert_equal(
            s_reload.metadata.as_dictionary(), s.metadata.as_dictionary()
        )

        # Delete reference to close np.memmap file
        del s_reload

    def test_load_save_lazy(self, save_path_nordif):
        s = load(PATTERN_FILE, lazy=True)
        assert isinstance(s.data, da.Array)
        s.save(save_path_nordif, overwrite=True)
        with pytest.warns(UserWarning):  # No background pattern in directory
            s_reload = load(
                save_path_nordif, lazy=True, setting_file=SETTING_FILE
            )
        assert s.data.shape == s_reload.data.shape

    def test_load_to_memory(self):
        s = load(PATTERN_FILE, lazy=False)
        assert isinstance(s.data, np.ndarray)
        assert not isinstance(s.data, np.memmap)

    def test_load_readonly(self):
        s = load(PATTERN_FILE, lazy=True)
        k = next(
            filter(
                lambda x: isinstance(x, str) and x.startswith("array-original"),
                s.data.dask.keys(),
            )
        )
        mm = s.data.dask[k]
        assert isinstance(mm, np.memmap)
        assert not mm.flags["WRITEABLE"]
        with pytest.raises(NotImplementedError):
            s.data[:] = 23

    @pytest.mark.parametrize("lazy", [True, False])
    def test_load_inplace(self, lazy):
        if lazy:
            with pytest.raises(ValueError):
                _ = load(PATTERN_FILE, lazy=lazy, mmap_mode="r+")
        else:
            s = load(PATTERN_FILE, lazy=lazy, mmap_mode="r+")
            assert s.axes_manager.as_dictionary() == AXES_MANAGER

    def test_save_fresh(self, save_path_nordif):
        scan_size = (10, 3)
        pattern_size = (5, 5)
        data_shape = scan_size + pattern_size
        s = EBSD((255 * np.random.rand(*data_shape)).astype(np.uint8))
        s.save(save_path_nordif, overwrite=True)
        with pytest.warns(UserWarning):  # No background or setting files
            s_reload = load(
                save_path_nordif,
                scan_size=scan_size[::-1],
                pattern_size=pattern_size,
            )
        assert np.allclose(s.data, s_reload.data)

    def test_write_data_line(self, save_path_nordif):
        scan_size = 3
        pattern_size = (5, 5)
        data_shape = (scan_size,) + pattern_size
        s = EBSD((255 * np.random.rand(*data_shape)).astype(np.uint8))
        s.save(save_path_nordif, overwrite=True)
        with pytest.warns(UserWarning):  # No background or setting files
            s_reload = load(
                save_path_nordif,
                scan_size=scan_size,
                pattern_size=pattern_size,
            )
        assert np.allclose(s.data, s_reload.data)

    def test_write_data_single(self, save_path_nordif):
        pattern_size = (5, 5)
        s = EBSD((255 * np.random.rand(*pattern_size)).astype(np.uint8))
        s.save(save_path_nordif, overwrite=True)
        with pytest.warns(UserWarning):  # No background or setting files
            s_reload = load(
                save_path_nordif, scan_size=1, pattern_size=pattern_size
            )
        assert np.allclose(s.data, s_reload.data)

    def test_read_cutoff(self, save_path_nordif):
        scan_size = (10, 3)
        scan_size_reloaded = (10, 20)
        pattern_size = (5, 5)
        data_shape = scan_size + pattern_size
        s = EBSD((255 * np.random.rand(*data_shape)).astype(np.uint8))
        s.save(save_path_nordif, overwrite=True)

        # Reload data but with a scan_size bigger than available file bytes,
        # so that the data has to be padded
        with pytest.warns(UserWarning):  # No background or setting files
            s_reload = load(
                save_path_nordif,
                scan_size=scan_size_reloaded[::-1],
                pattern_size=pattern_size,
            )

        # To check if the data padding works as expected, the original data is
        # padded and compared to the reloaded data
        cut_data = s.data.flatten()
        pw = [
            (
                0,
                (scan_size_reloaded[1] - scan_size[1])
                * scan_size[0]
                * np.prod(pattern_size),
            )
        ]
        cut_data = np.pad(cut_data, pw, mode="constant")
        cut_data = cut_data.reshape(scan_size_reloaded + pattern_size)
        assert np.allclose(cut_data, s_reload.data)
