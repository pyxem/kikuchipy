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

import os
from numbers import Number

import dask.array as da
from hyperspy.utils.roi import RectangularROI
from hyperspy.misc.utils import DictionaryTreeBrowser
import matplotlib
from matplotlib.pyplot import close
import numpy as np
from orix.crystal_map import CrystalMap
from orix.quaternion import Rotation
import pytest
from scipy.ndimage import correlate
from skimage.exposure import rescale_intensity

from kikuchipy import load
from kikuchipy.filters.window import Window
from kikuchipy.pattern._pattern import fft_spectrum
from kikuchipy.signals.ebsd import EBSD, LazyEBSD
from kikuchipy.signals.util._metadata import ebsd_metadata, metadata_nodes

matplotlib.use("Agg")  # For plotting

DIR_PATH = os.path.dirname(__file__)
KIKUCHIPY_FILE = os.path.join(DIR_PATH, "../../data/kikuchipy/patterns.h5")
EMSOFT_FILE = os.path.join(DIR_PATH, "../../data/emsoft_ebsd/simulated_ebsd.h5")


def assert_dictionary(input_dict, output_dict):
    if isinstance(input_dict, DictionaryTreeBrowser):
        input_dict = input_dict.as_dictionary()
        output_dict = output_dict.as_dictionary()
    for key in output_dict.keys():
        if isinstance(output_dict[key], dict):
            assert_dictionary(input_dict[key], output_dict[key])
        else:
            if isinstance(output_dict[key], list) or isinstance(
                input_dict[key], list
            ):
                output_dict[key] = np.array(output_dict[key])
                input_dict[key] = np.array(input_dict[key])
            if isinstance(output_dict[key], (np.ndarray, Number)):
                assert np.allclose(input_dict[key], output_dict[key])
            else:
                assert input_dict[key] == output_dict[key]


class TestEBSD:
    def test_init(self):

        # Signal shape
        array0 = np.zeros(shape=(10, 10, 10, 10))
        s0 = EBSD(array0)
        assert array0.shape == s0.axes_manager.shape

        # Cannot initialise signal with one signal dimension
        with pytest.raises(ValueError):
            _ = EBSD(np.zeros(10))

        # Shape of one-image signal
        array1 = np.zeros(shape=(10, 10))
        s1 = EBSD(array1)
        assert array1.shape == s1.axes_manager.shape

        # SEM metadata
        kp_md = ebsd_metadata()
        sem_node = metadata_nodes("sem")
        assert_dictionary(
            kp_md.get_item(sem_node), s1.metadata.get_item(sem_node)
        )

        # Phases metadata
        assert s1.metadata.has_item("Sample.Phases")

    def test_as_lazy(self, dummy_signal):
        lazy_signal = dummy_signal.as_lazy()

        # Assert that lazy attribute and class changed, while metadata was not
        # changed
        assert lazy_signal._lazy is True
        assert lazy_signal.__class__ == LazyEBSD
        assert_dictionary(
            dummy_signal.metadata.as_dictionary(),
            lazy_signal.metadata.as_dictionary(),
        )

    def test_set_experimental_parameters(self, dummy_signal):
        p = {
            "detector": "NORDIF UF-1100",
            "azimuth_angle": 1.0,
            "elevation_angle": 1.0,
            "sample_tilt": 70.0,
            "working_distance": 23.2,
            "binning": 8,
            "exposure_time": 0.01,
            "grid_type": "square",
            "gain": 10,
            "frame_number": 4,
            "frame_rate": 100,
            "scan_time": 60.0,
            "beam_energy": 20.0,
            "xpc": 0.5,
            "ypc": 0.5,
            "zpc": 15000.0,
            "static_background": np.ones(shape=(10, 10)),
            "manufacturer": "NORDIF",
            "version": "3.1.2",
            "microscope": "Hitachi SU-6600",
            "magnification": 500,
        }
        dummy_signal.set_experimental_parameters(**p)
        ebsd_node = metadata_nodes("ebsd")
        md_dict = dummy_signal.metadata.get_item(ebsd_node).as_dictionary()
        assert_dictionary(p, md_dict)

    def test_set_phase_parameters(self, dummy_signal):
        p = {
            "number": 1,
            "atom_coordinates": {
                "1": {
                    "atom": "Ni",
                    "coordinates": [0, 0, 0],
                    "site_occupation": 1,
                    "debye_waller_factor": 0.0035,
                }
            },
            "formula": "Ni",
            "info": "Some sample info",
            "lattice_constants": [0.35236, 0.35236, 0.35236, 90, 90, 90],
            "laue_group": "m3m",
            "material_name": "Ni",
            "point_group": "432",
            "space_group": 225,
            "setting": 1,
            "source": "Peng",
            "symmetry": 43,
        }
        dummy_signal.set_phase_parameters(**p)
        md_dict = dummy_signal.metadata.get_item(
            "Sample.Phases.1"
        ).as_dictionary()
        p.pop("number")
        assert_dictionary(p, md_dict)

    def test_set_scan_calibration(self, dummy_signal):
        (new_step_x, new_step_y) = (2, 3)
        dummy_signal.set_scan_calibration(step_x=new_step_x, step_y=new_step_y)
        x, y = dummy_signal.axes_manager.navigation_axes
        assert (x.name, y.name) == ("x", "y")
        assert (x.scale, y.scale) == (new_step_x, new_step_y)
        assert x.units, y.units == "um"

    def test_set_detector_calibration(self, dummy_signal):
        delta = 70
        dummy_signal.set_detector_calibration(delta=delta)
        dx, dy = dummy_signal.axes_manager.signal_axes
        centre = np.array(dummy_signal.axes_manager.signal_shape) / 2 * delta
        assert dx.units, dy.units == "um"
        assert dx.scale, dy.scale == delta
        assert dx.offset, dy.offset == -centre


class TestRemoveStaticBackgroundEBSD:
    @pytest.mark.parametrize(
        "operation, relative, answer",
        [
            (
                "subtract",
                True,
                # fmt: off
                np.array(
                    [
                        115, 162, 115, 185, 185, 139, 162, 46, 46, 208, 185,
                        185, 185, 46, 208, 208, 185, 185, 0, 92, 69, 139, 92,
                        231, 92, 92, 255, 185, 46, 162, 162, 139, 208, 208, 69,
                        92, 92, 23, 208, 23, 69, 23, 69, 69, 162, 185, 162, 0,
                        115, 208, 185, 185, 162, 46, 92, 46, 139, 92, 139, 23,
                        46, 46, 46, 115, 231, 185, 115, 185, 23, 69, 231, 92,
                        208, 115, 69, 162, 162, 162, 69, 139, 255,
                    ]
                ),
                # fmt: on
            ),
            (
                "subtract",
                False,
                # fmt: off
                np.array(
                    [
                        127, 212, 127, 255, 255, 170, 212, 0, 0, 255, 218, 218,
                        218, 0, 255, 255, 218, 218, 0, 92, 69, 139, 92, 231, 92,
                        92, 255, 218, 0, 182, 182, 145, 255, 255, 36, 72, 95, 0,
                        255, 0, 63, 0, 63, 63, 191, 226, 198, 0, 141, 255, 226,
                        226, 198, 56, 153, 51, 255, 153, 255, 0, 51, 51, 51,
                        113, 255, 198, 113, 198, 0, 56, 255, 85, 191, 63, 0,
                        127, 127, 127, 0, 95, 255,
                    ]
                ),
                # fmt: on
            ),
            (
                "divide",
                True,
                # fmt: off
                np.array(
                    [
                        85, 127, 85, 148, 170, 106, 127, 21, 0, 152, 148, 136,
                        148, 0, 170, 170, 148, 170, 0, 63, 51, 106, 56, 191, 63,
                        63, 255, 136, 21, 119, 127, 113, 170, 170, 42, 56, 68,
                        0, 152, 0, 28, 0, 42, 42, 141, 136, 127, 0, 85, 198,
                        148, 148, 127, 0, 68, 21, 102, 63, 113, 0, 21, 21, 0,
                        85, 191, 136, 85, 170, 0, 42, 191, 56, 152, 85, 51, 127,
                        141, 127, 42, 106, 255,
                    ]
                )
                # fmt: on
            ),
            (
                "divide",
                False,
                # fmt: off
                np.array(
                    [
                        127, 191, 127, 223, 255, 159, 191, 31, 0, 229, 223, 204,
                        223, 0, 255, 255, 223, 255, 0, 63, 51, 106, 56, 191, 63,
                        63, 255, 196, 0, 167, 182, 157, 255, 255, 36, 60, 113,
                        0, 255, 0, 47, 0, 70, 70, 236, 174, 163, 0, 109, 255,
                        191, 191, 163, 0, 152, 47, 229, 143, 255, 0, 47, 47, 0,
                        113, 255, 181, 113, 226, 0, 56, 255, 75, 132, 51, 10,
                        102, 118, 102, 0, 76, 255,
                    ]
                )
                # fmt: on
            ),
        ],
    )
    def test_remove_static_background(
        self, dummy_signal, dummy_background, operation, relative, answer
    ):
        """This tests uses a hard-coded answer. If specifically
        improvements to the intensities produced by this correction is
        to be made, these hard-coded answers will have to be
        recalculated for the tests to pass.
        """

        dummy_signal.remove_static_background(
            operation=operation, relative=relative, static_bg=dummy_background
        )
        answer = answer.reshape((3, 3, 3, 3)).astype(np.uint8)
        assert np.allclose(dummy_signal.data, answer)

    @pytest.mark.parametrize(
        "static_bg, error, match",
        [
            (
                np.ones((3, 3), dtype=np.int8),
                ValueError,
                "The static background dtype_out",
            ),
            (None, OSError, "The static background is not a numpy or dask"),
            (np.ones((3, 2), dtype=np.uint8), OSError, "The pattern"),
        ],
    )
    def test_incorrect_static_background_pattern(
        self, dummy_signal, static_bg, error, match
    ):
        """Test for expected error messages when passing an incorrect
        static background pattern to `remove_static_background().`
        """

        ebsd_node = metadata_nodes("ebsd")
        dummy_signal.metadata.set_item(
            ebsd_node + ".static_background", static_bg
        )
        with pytest.raises(error, match=match):
            dummy_signal.remove_static_background()

    def test_lazy_remove_static_background(
        self, dummy_signal, dummy_background
    ):
        dummy_signal = dummy_signal.as_lazy()
        dummy_signal.remove_static_background(static_bg=dummy_background)
        assert isinstance(dummy_signal.data, da.Array)

    def test_remove_static_background_scalebg(
        self, dummy_signal, dummy_background
    ):
        dummy_signal2 = dummy_signal.deepcopy()

        dummy_signal.remove_static_background(
            scale_bg=True, relative=False, static_bg=dummy_background,
        )

        dummy_signal2.remove_static_background(
            scale_bg=False, relative=False, static_bg=dummy_background,
        )

        p1 = dummy_signal.inav[0, 0].data
        p2 = dummy_signal2.inav[0, 0].data

        assert not np.allclose(p1, p2, atol=0.1)
        assert np.allclose(
            p1, np.array([[15, 150, 15], [180, 255, 120], [150, 0, 75]])
        )

    def test_remove_static_background_relative_and_scalebg_raises(
        self, dummy_signal, dummy_background
    ):
        with pytest.raises(ValueError, match="'scale_bg' must be False"):
            dummy_signal.remove_static_background(
                relative=True, scale_bg=True, static_bg=dummy_background,
            )


class TestRemoveDynamicBackgroundEBSD:
    @pytest.mark.parametrize(
        "operation, std, answer",
        [
            (
                "subtract",
                2,
                # fmt: off
                np.array(
                    [
                        170, 215, 181, 255, 221, 188, 221, 32, 0, 255, 198, 228,
                        199, 0, 230, 229, 201, 174, 0, 84, 77, 147, 48, 255, 81,
                        74, 249, 246, 0, 216, 177, 109, 255, 250, 40, 44, 120,
                        2, 255, 8, 32, 0, 67, 63, 145, 255, 195, 0, 120, 229,
                        237, 222, 196, 1, 164, 34, 255, 128, 173, 0, 47, 49, 7,
                        133, 245, 218, 110, 166, 0, 59, 255, 60, 255, 71, 35,
                        145, 108, 144, 0, 108, 253,
                    ],
                ),
                # fmt: on
            ),
            (
                "subtract",
                3,
                # fmt: off
                np.array(
                    [
                        181, 218, 182, 255, 218, 182, 218, 36, 0, 255, 198, 226,
                        198, 0, 226, 226, 198, 170, 0, 84, 84, 142, 56, 255, 84,
                        84, 254, 254, 0, 218, 181, 109, 255, 254, 36, 36, 113,
                        0, 255, 0, 28, 0, 57, 57, 141, 255, 191, 0, 127, 223,
                        223, 223, 191, 0, 169, 42, 255, 127, 170, 0, 42, 42, 0,
                        141, 254, 226, 113, 169, 0, 56, 255, 56, 255, 72, 36,
                        145, 109, 145, 0, 109, 254,
                    ],
                ),
                # fmt: on
            ),
            (
                "divide",
                2,
                # fmt: off
                np.array(
                    [
                        176, 217, 186, 255, 225, 194, 225, 39, 0, 255, 199, 228,
                        199, 0, 231, 230, 202, 174, 0, 93, 88, 159, 60, 255, 91,
                        86, 245, 241, 0, 214, 174, 107, 255, 247, 37, 38, 127,
                        0, 255, 0, 30, 0, 67, 63, 150, 255, 199, 0, 128, 234,
                        244, 224, 201, 0, 166, 42, 255, 133, 180, 0, 47, 48, 0,
                        132, 238, 212, 109, 164, 0, 56, 255, 57, 255, 72, 36,
                        146, 109, 145, 0, 109, 252,
                    ],
                ),
                # fmt: on
            ),
            (
                "divide",
                3,
                # fmt: off
                np.array(
                    [
                        181, 218, 182, 255, 219, 182, 219, 36, 0, 255, 198, 226,
                        198, 0, 226, 226, 198, 170, 0, 85, 85, 142, 56, 255, 85,
                        85, 254, 254, 0, 218, 181, 109, 255, 254, 36, 36, 114,
                        0, 255, 0, 28, 0, 57, 57, 142, 255, 191, 0, 127, 223,
                        224, 223, 191, 0, 169, 42, 255, 127, 170, 0, 42, 42, 0,
                        141, 253, 225, 113, 169, 0, 56, 255, 56, 255, 72, 36,
                        145, 109, 145, 0, 109, 254,
                    ],
                ),
                # fmt: on
            ),
        ],
    )
    def test_remove_dynamic_background_spatial(
        self, dummy_signal, operation, std, answer
    ):
        """This tests uses a hard-coded answer. If specifically
        improvements to the intensities produced by this correction is
        to be made, these hard-coded answers will have to be
        recalculated for the tests to pass.
        """

        dummy_signal.remove_dynamic_background(
            operation=operation, std=std, filter_domain="spatial",
        )
        answer = answer.reshape((3,) * 4).astype(np.uint8)
        assert np.allclose(dummy_signal.data, answer)
        assert dummy_signal.data.dtype == answer.dtype

    def test_lazy_remove_dynamic_background(self, dummy_signal):
        dummy_signal = dummy_signal.as_lazy()
        dummy_signal.remove_dynamic_background(filter_domain="spatial")
        assert isinstance(dummy_signal.data, da.Array)

    @pytest.mark.parametrize(
        "operation, std, answer",
        [
            (
                "subtract",
                2,
                np.array(
                    [
                        [0.2518, 0.6835, 0.4054],
                        [1, 0.7815, 0.5793],
                        [0.6947, -0.8867, -1],
                    ],
                    dtype=np.float64,
                ),
            ),
            (
                "subtract",
                3,
                np.array(
                    [
                        [42133, 55527, 47066],
                        [65535, 58072, 50768],
                        [56305, 6059, 0],
                    ],
                    dtype=np.uint16,
                ),
            ),
            (
                "divide",
                2,
                np.array(
                    [
                        [0.4119, 0.7575, 0.5353],
                        [1, 0.8562, 0.7038],
                        [0.7683, -0.6622, -1],
                    ],
                    dtype=np.float32,
                ),
            ),
            (
                "divide",
                3,
                np.array(
                    [[177, 222, 195], [255, 234, 210], [226, 41, 0]],
                    dtype=np.uint8,
                ),
            ),
        ],
    )
    def test_remove_dynamic_background_frequency(
        self, dummy_signal, operation, std, answer
    ):
        dtype_out = answer.dtype
        dummy_signal.data = dummy_signal.data.astype(dtype_out)

        filter_domain = "frequency"
        dummy_signal.remove_dynamic_background(
            operation=operation, std=std, filter_domain=filter_domain,
        )

        assert dummy_signal.data.dtype == dtype_out
        assert np.allclose(dummy_signal.inav[0, 0].data, answer, atol=1e-4)

    def test_remove_dynamic_background_raises(self, dummy_signal):
        filter_domain = "wildmount"
        with pytest.raises(ValueError, match=f"{filter_domain} must be "):
            dummy_signal.remove_dynamic_background(filter_domain=filter_domain)


class TestRescaleIntensityEBSD:
    @pytest.mark.parametrize(
        "relative, dtype_out, answer",
        [
            (
                True,
                None,
                np.array(
                    [[141, 170, 141], [198, 170, 141], [170, 28, 0]],
                    dtype=np.uint8,
                ),
            ),
            (
                True,
                np.float32,
                np.array(
                    [
                        [0.1111, 0.3333, 0.1111],
                        [0.5555, 0.3333, 0.1111],
                        [0.3333, -0.7777, -1],
                    ],
                    dtype=np.float32,
                ),
            ),
            (
                False,
                None,
                np.array(
                    [[182, 218, 182], [255, 218, 182], [218, 36, 0]],
                    dtype=np.uint8,
                ),
            ),
            (
                False,
                np.float32,
                np.array(
                    [
                        [0.4285, 0.7142, 0.4285],
                        [1, 0.7142, 0.4285],
                        [0.7142, -0.7142, -1],
                    ],
                    dtype=np.float32,
                ),
            ),
        ],
    )
    def test_rescale_intensity(self, dummy_signal, relative, dtype_out, answer):
        """This tests uses a hard-coded answer. If specifically
        improvements to the intensities produced by this correction is
        to be made, these hard-coded answers will have to be
        recalculated for the tests to pass.
        """

        dummy_signal.rescale_intensity(relative=relative, dtype_out=dtype_out)

        assert dummy_signal.data.dtype == answer.dtype
        assert np.allclose(dummy_signal.inav[0, 0].data, answer, atol=1e-4)

    def test_lazy_rescale_intensity(self, dummy_signal):
        dummy_signal = dummy_signal.as_lazy()
        dummy_signal.rescale_intensity()
        assert isinstance(dummy_signal.data, da.Array)

    @pytest.mark.parametrize(
        "percentiles, answer",
        [
            (
                (10, 90),
                np.array([[198, 245, 198], [254, 245, 198], [245, 9, 0]]),
            ),
            (
                (1, 99),
                np.array([[183, 220, 183], [255, 220, 183], [220, 34, 0]]),
            ),
        ],
    )
    def test_rescale_intensity_percentiles(
        self, dummy_signal, percentiles, answer
    ):
        dummy_signal.data = dummy_signal.data.astype(np.float32)
        dtype_out = np.uint8
        dummy_signal.rescale_intensity(
            percentiles=percentiles, dtype_out=dtype_out
        )

        assert dummy_signal.data.dtype == dtype_out
        assert np.allclose(dummy_signal.inav[0, 0].data, answer)

    def test_rescale_intensity_in_range(self, dummy_signal):
        dummy_data = dummy_signal.deepcopy().data

        dummy_signal.rescale_intensity()

        assert dummy_signal.data.dtype == dummy_data.dtype
        assert not np.allclose(dummy_signal.data, dummy_data, atol=1)

    def test_rescale_intensity_raises_in_range_percentiles(self, dummy_signal):
        with pytest.raises(ValueError, match="'percentiles' must be None"):
            dummy_signal.rescale_intensity(
                in_range=(1, 254), percentiles=(1, 99),
            )

    def test_rescale_intensity_raises_in_range_relative(self, dummy_signal):
        with pytest.raises(ValueError, match="'in_range' must be None if "):
            dummy_signal.rescale_intensity(
                in_range=(1, 254), relative=True,
            )


class TestAdaptiveHistogramEqualizationEBSD:
    def test_adaptive_histogram_equalization(self):
        """Test setup of equalization only. Tests of the result of the
        actual equalization are found elsewhere.
        """

        s = load(KIKUCHIPY_FILE)

        # These window sizes should work without issue
        for kernel_size in [None, 10]:
            s.adaptive_histogram_equalization(kernel_size=kernel_size)

        # These window sizes should throw errors
        with pytest.raises(ValueError, match="invalid literal for int()"):
            s.adaptive_histogram_equalization(kernel_size=("wrong", "size"))
        with pytest.raises(ValueError, match="Incorrect value of `shape"):
            s.adaptive_histogram_equalization(kernel_size=(10, 10, 10))

    def test_lazy_adaptive_histogram_equalization(self):
        s = load(KIKUCHIPY_FILE, lazy=True)
        s.adaptive_histogram_equalization()
        assert isinstance(s.data, da.Array)


class TestAverageNeighbourPatternsEBSD:
    # Test different window data
    @pytest.mark.parametrize(
        "window, window_shape, lazy, answer, kwargs",
        [
            (
                "circular",
                (3, 3),
                False,
                # fmt: off
                np.array(
                    [
                        7, 4, 6, 6, 3, 7, 7, 3, 2, 4, 4, 6, 4, 2, 5, 4, 3, 5, 5,
                        5, 3, 5, 3, 8, 6, 5, 5, 5, 2, 6, 4, 3, 3, 4, 1, 1, 6, 4,
                        6, 4, 3, 4, 5, 5, 3, 5, 3, 3, 3, 3, 5, 3, 4, 5, 5, 3, 7,
                        4, 4, 2, 3, 4, 1, 5, 3, 6, 3, 4, 1, 1, 4, 4, 7, 6, 3, 4,
                        6, 4, 3, 6, 3,
                    ],
                ),
                # fmt: on
                None,
            ),
            (
                "rectangular",
                (2, 3),
                False,
                # fmt: off
                np.array(
                    [
                        7, 6, 6, 7, 3, 6, 7, 4, 3, 4, 5, 5, 6, 2, 7, 5, 3, 5, 4,
                        5, 5, 6, 1, 8, 5, 5, 7, 6, 3, 7, 5, 2, 5, 6, 3, 3, 5, 3,
                        5, 4, 3, 6, 5, 3, 3, 5, 4, 5, 4, 2, 6, 5, 4, 5, 5, 2, 7,
                        3, 3, 2, 3, 3, 2, 6, 3, 5, 3, 4, 3, 3, 4, 3, 6, 4, 5, 3,
                        4, 3, 3, 5, 4,
                    ],
                ),
                # fmt: on
                None,
            ),
            (
                "gaussian",
                (3, 3),
                True,
                # fmt: off
                np.array(
                    [
                        6, 3, 7, 5, 2, 5, 6, 2, 3, 5, 3, 5, 4, 3, 6, 5, 3, 3, 5,
                        4, 4, 4, 2, 6, 5, 4, 5, 5, 3, 7, 4, 3, 3, 4, 3, 2, 5, 4,
                        5, 4, 3, 4, 4, 4, 3, 5, 4, 4, 4, 3, 5, 4, 5, 5, 5, 2, 7,
                        3, 3, 1, 3, 3, 2, 6, 3, 5, 3, 4, 3, 3, 4, 3, 6, 4, 4, 3,
                        4, 3, 3, 5, 4,
                    ],
                ),
                # fmt: on
                {"std": 2},  # standard deviation
            ),
        ],
    )
    def test_average_neighbour_patterns(
        self, dummy_signal, window, window_shape, lazy, answer, kwargs,
    ):
        if lazy:
            dummy_signal = dummy_signal.as_lazy()

        if kwargs is None:
            dummy_signal.average_neighbour_patterns(
                window=window, window_shape=window_shape,
            )
        else:
            dummy_signal.average_neighbour_patterns(
                window=window, window_shape=window_shape, **kwargs,
            )

        answer = answer.reshape((3, 3, 3, 3)).astype(np.uint8)
        assert np.allclose(dummy_signal.data, answer)
        assert dummy_signal.data.dtype == answer.dtype

    def test_average_neighbour_patterns_no_averaging(self, dummy_signal):
        answer = dummy_signal.data.copy()
        with pytest.warns(UserWarning, match="A window of shape .* was "):
            dummy_signal.average_neighbour_patterns(
                window="rectangular", window_shape=(1, 1),
            )
        assert np.allclose(dummy_signal.data, answer)
        assert dummy_signal.data.dtype == answer.dtype

    def test_average_neighbour_patterns_one_nav_dim(self, dummy_signal):
        dummy_signal_1d = dummy_signal.inav[:, 0]
        dummy_signal_1d.average_neighbour_patterns(window_shape=(3,))
        # fmt: off
        answer = np.array(
            [
                7, 6, 6, 7, 3, 6, 7, 4, 3, 4, 5, 5, 6, 2, 7, 5, 3, 5, 4, 5, 5,
                6, 1, 8, 5, 5, 7,
            ],
            dtype=np.uint8
        ).reshape(dummy_signal_1d.axes_manager.shape)
        # fmt: on
        assert np.allclose(dummy_signal_1d.data, answer)
        assert dummy_signal.data.dtype == answer.dtype

    def test_average_neighbour_patterns_window_1d(self, dummy_signal):
        dummy_signal.average_neighbour_patterns(window_shape=(3,))
        # fmt: off
        answer = np.array(
            [
                6, 3, 6, 6, 5, 6, 7, 1, 1, 6, 3, 8, 3, 0, 4, 5, 4, 5, 4, 4, 1,
                4, 4, 8, 5, 4, 4, 5, 2, 6, 5, 4, 4, 5, 1, 0, 6, 5, 8, 3, 2, 2,
                4, 6, 4, 5, 4, 2, 5, 4, 7, 4, 4, 6, 6, 1, 6, 4, 4, 4, 4, 1, 1,
                4, 4, 8, 2, 3, 0, 2, 5, 3, 8, 5, 1, 5, 6, 6, 4, 5, 4,
            ],
            dtype=np.uint8
        ).reshape(dummy_signal.axes_manager.shape)
        # fmt: on
        assert np.allclose(dummy_signal.data, answer)
        assert dummy_signal.data.dtype == answer.dtype

    def test_average_neighbour_patterns_pass_window(self, dummy_signal):
        w = Window()
        dummy_signal.average_neighbour_patterns(w)
        # fmt: off
        answer = np.array(
            [
                7, 4, 6, 6, 3, 7, 7, 3, 2, 4, 4, 6, 4, 2, 5, 4, 3, 5, 5, 5, 3,
                5, 3, 8, 6, 5, 5, 5, 2, 6, 4, 3, 3, 4, 1, 1, 6, 4, 6, 4, 3, 4,
                5, 5, 3, 5, 3, 3, 3, 3, 5, 3, 4, 5, 5, 3, 7, 4, 4, 2, 3, 4, 1,
                5, 3, 6, 3, 4, 1, 1, 4, 4, 7, 6, 3, 4, 6, 4, 3, 6, 3,
            ],
            dtype=np.uint8
        ).reshape(dummy_signal.axes_manager.shape)
        # fmt: on
        assert np.allclose(dummy_signal.data, answer)
        assert dummy_signal.data.dtype == answer.dtype


class TestRebin:
    def test_rebin(self, dummy_signal):
        ebsd_node = metadata_nodes("ebsd")

        # Passing new_shape, only scaling in signal space
        new_shape = (3, 3, 2, 1)
        new_binning = dummy_signal.axes_manager.shape[3] / new_shape[3]
        s2 = dummy_signal.rebin(new_shape=new_shape)
        assert s2.axes_manager.shape == new_shape
        assert s2.metadata.get_item(ebsd_node + ".binning") == new_binning

        # Passing scale, also scaling in navigation space
        scale = (3, 1, 3, 2)
        s2 = dummy_signal.rebin(scale=scale)
        expected_new_shape = [
            int(i / j) for i, j in zip(dummy_signal.axes_manager.shape, scale)
        ]
        assert s2.axes_manager.shape == tuple(expected_new_shape)
        assert s2.metadata.get_item(ebsd_node + ".binning") == float(scale[2])

        # Passing lazy signal to out parameter, only scaling in signal space but
        # upscaling
        scale = (1, 1, 1, 0.5)
        expected_new_shape = [
            int(i / j) for i, j in zip(dummy_signal.axes_manager.shape, scale)
        ]
        s2 = dummy_signal.copy().as_lazy()
        s3 = dummy_signal.rebin(scale=scale, out=s2)
        assert isinstance(s2, LazyEBSD)
        assert s2.axes_manager.shape == tuple(expected_new_shape)
        assert s2.metadata.get_item(ebsd_node + ".binning") == float(scale[3])
        assert s3 is None


class TestVirtualBackscatterElectronImaging:
    @pytest.mark.parametrize("out_signal_axes", [None, (0, 1), ("x", "y")])
    def test_virtual_backscatter_electron_imaging(
        self, dummy_signal, out_signal_axes
    ):
        dummy_signal.axes_manager.navigation_axes[0].name = "x"
        dummy_signal.axes_manager.navigation_axes[1].name = "y"

        roi = RectangularROI(left=0, top=0, right=1, bottom=1)
        dummy_signal.plot_virtual_bse_intensity(
            roi, out_signal_axes=out_signal_axes
        )

        close("all")

    def test_get_virtual_image(self, dummy_signal):
        roi = RectangularROI(left=0, top=0, right=1, bottom=1)
        virtual_image_signal = dummy_signal.get_virtual_bse_intensity(roi)
        assert (
            virtual_image_signal.data.shape
            == dummy_signal.axes_manager.navigation_shape
        )

    def test_virtual_backscatter_electron_imaging_raises(self, dummy_signal):
        roi = RectangularROI(0, 0, 1, 1)
        with pytest.raises(ValueError):
            _ = dummy_signal.get_virtual_bse_intensity(
                roi, out_signal_axes=(0, 1, 2)
            )


class TestDecomposition:
    def test_decomposition(self, dummy_signal):
        dummy_signal.change_dtype(np.float32)
        dummy_signal.decomposition()
        assert isinstance(dummy_signal, EBSD)

    def test_lazy_decomposition(self, dummy_signal):
        lazy_signal = dummy_signal.as_lazy()
        lazy_signal.change_dtype(np.float32)
        lazy_signal.decomposition()
        assert isinstance(lazy_signal, LazyEBSD)

    @pytest.mark.parametrize(
        "components, dtype_out, mean_intensity",
        [
            (None, np.float16, 4.520),
            (None, np.float32, 4.518695),
            (3, np.float16, 4.516),
            ([0, 1, 3], np.float16, 4.504),
        ],
    )
    def test_get_decomposition_model(
        self, dummy_signal, components, dtype_out, mean_intensity
    ):

        # Decomposition
        dummy_signal.change_dtype(np.float32)
        dummy_signal.decomposition(algorithm="SVD")

        # Get decomposition model
        model_signal = dummy_signal.get_decomposition_model(
            components=components, dtype_out=dtype_out
        )

        # Check data shape, signal class and image intensities in model
        # signal
        assert model_signal.data.shape == dummy_signal.data.shape
        assert isinstance(model_signal, EBSD)
        assert np.allclose(model_signal.data.mean(), mean_intensity, atol=1e-3)

    @pytest.mark.parametrize(
        "components, mean_intensity",
        [(None, 132.1), (3, 122.9), ([0, 1, 3], 116.8)],
    )
    def test_get_decomposition_model_lazy(
        self, dummy_signal, components, mean_intensity
    ):
        # Decomposition
        lazy_signal = dummy_signal.as_lazy()
        lazy_signal.change_dtype(np.float32)
        lazy_signal.decomposition(algorithm="PCA", output_dimension=9)

        # Signal type
        assert isinstance(lazy_signal, LazyEBSD)

        # Turn factors and loadings into dask arrays
        lazy_signal.learning_results.factors = da.from_array(
            lazy_signal.learning_results.factors
        )
        lazy_signal.learning_results.loadings = da.from_array(
            lazy_signal.learning_results.loadings
        )

        # Get decomposition model
        model_signal = lazy_signal.get_decomposition_model(
            components=components, dtype_out=np.float32
        )

        # Check data shape, signal class and image intensities in model
        # signal after rescaling to 8 bit unsigned integer
        assert model_signal.data.shape == lazy_signal.data.shape
        assert isinstance(model_signal, LazyEBSD)
        model_signal.rescale_intensity(relative=True, dtype_out=np.uint8)
        model_mean = model_signal.data.mean().compute()
        assert np.allclose(model_mean, mean_intensity, atol=0.1)

    @pytest.mark.parametrize(
        "components, mean_intensity", [(None, 132.1), (3, 122.9)]
    )
    def test_get_decomposition_model_write(
        self, dummy_signal, components, mean_intensity, tmp_path
    ):
        lazy_signal = dummy_signal.as_lazy()
        dtype_in = lazy_signal.data.dtype

        # Decomposition
        lazy_signal.change_dtype(np.float32)
        lazy_signal.decomposition(algorithm="PCA", output_dimension=9)
        lazy_signal.change_dtype(dtype_in)

        with pytest.raises(AttributeError, match="Output directory has to be"):
            lazy_signal.get_decomposition_model_write()

        # Current time stamp is added to output file name
        lazy_signal.get_decomposition_model_write(dir_out=tmp_path)

        # Reload file to check...
        fname_out = "tests.h5"
        lazy_signal.get_decomposition_model_write(
            components=components, dir_out=tmp_path, fname_out=fname_out
        )
        s_reload = load(os.path.join(tmp_path, fname_out))

        # ... data type, data shape and mean intensity
        assert s_reload.data.dtype == lazy_signal.data.dtype
        assert s_reload.data.shape == lazy_signal.data.shape
        assert np.allclose(s_reload.data.mean(), mean_intensity, atol=1e-1)


class TestLazy:
    def test_compute(self, dummy_signal):
        lazy_signal = dummy_signal.as_lazy()

        lazy_signal.compute()
        assert isinstance(lazy_signal, EBSD)
        assert lazy_signal._lazy is False

    def test_change_dtype(self, dummy_signal):
        lazy_signal = dummy_signal.as_lazy()

        assert isinstance(lazy_signal, LazyEBSD)
        lazy_signal.change_dtype("uint16")
        assert isinstance(lazy_signal, LazyEBSD)


class TestGetDynamicBackgroundEBSD:
    def test_get_dynamic_background_spatial(self, dummy_signal):
        dtype_out = dummy_signal.data.dtype
        bg = dummy_signal.get_dynamic_background(
            filter_domain="spatial", std=2, truncate=3,
        )

        assert bg.data.dtype == dtype_out
        assert isinstance(bg, EBSD)

    def test_get_dynamic_background_frequency(self, dummy_signal):
        dtype_out = np.float32
        bg = dummy_signal.get_dynamic_background(
            filter_domain="frequency", std=2, truncate=3, dtype_out=dtype_out,
        )

        assert bg.data.dtype == dtype_out
        assert isinstance(bg, EBSD)

    def test_get_dynamic_background_raises(self, dummy_signal):
        filter_domain = "Vasselheim"
        with pytest.raises(ValueError, match=f"{filter_domain} must be"):
            _ = dummy_signal.get_dynamic_background(filter_domain=filter_domain)

    def test_get_dynamic_background_lazy(self, dummy_signal):
        lazy_signal = dummy_signal.as_lazy()

        bg = lazy_signal.get_dynamic_background()

        assert isinstance(bg, LazyEBSD)

        bg.compute()

        assert isinstance(bg, EBSD)


class TestGetImageQualityEBSD:
    @pytest.mark.parametrize(
        "normalize, lazy, answer",
        [
            (
                True,
                False,
                np.array(
                    [
                        [-0.0241, -0.0625, -0.0052],
                        [-0.0317, -0.0458, -0.0956],
                        [-0.1253, 0.0120, -0.2385],
                    ],
                    dtype=np.float64,
                ),
            ),
            (
                False,
                True,
                np.array(
                    [
                        [0.2694, 0.2926, 0.2299],
                        [0.2673, 0.1283, 0.2032],
                        [0.1105, 0.2671, 0.2159],
                    ],
                    dtype=np.float64,
                ),
            ),
        ],
    )
    def test_get_image_quality(self, dummy_signal, normalize, lazy, answer):
        if lazy:
            dummy_signal = dummy_signal.as_lazy()

        iq = dummy_signal.get_image_quality(normalize=normalize)

        if lazy:
            iq = iq.compute()

        assert np.allclose(iq, answer, atol=1e-4)


class TestFFTFilterEBSD:
    @pytest.mark.parametrize(
        "shift, transfer_function, kwargs, dtype_out, expected_spectrum_sum",
        [
            (True, "modified_hann", {}, None, 5.2000),
            (
                True,
                "lowpass",
                {"cutoff": 30, "cutoff_width": 15},
                np.float64,
                6.1428,
            ),
            (
                False,
                "highpass",
                {"cutoff": 2, "cutoff_width": 1},
                np.float32,
                5.4155,
            ),
            (False, "gaussian", {"sigma": 2}, None, 6.2621),
        ],
    )
    def test_fft_filter_frequency(
        self,
        dummy_signal,
        shift,
        transfer_function,
        kwargs,
        dtype_out,
        expected_spectrum_sum,
    ):
        if dtype_out is None:
            dtype_out = np.float32
        dummy_signal.data = dummy_signal.data.astype(dtype_out)

        shape = dummy_signal.axes_manager.signal_shape
        w = Window(transfer_function, shape=shape, **kwargs)

        dummy_signal.fft_filter(
            transfer_function=w, function_domain="frequency", shift=shift,
        )

        assert isinstance(dummy_signal, EBSD)
        assert dummy_signal.data.dtype == dtype_out
        assert np.allclose(
            np.sum(fft_spectrum(dummy_signal.inav[0, 0].data)),
            expected_spectrum_sum,
            atol=1e-4,
        )

    def test_fft_filter_spatial(self, dummy_signal):
        dummy_signal.change_dtype(np.float32)
        p = dummy_signal.inav[0, 0].deepcopy().data

        # Sobel operator
        w = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

        dummy_signal.fft_filter(
            transfer_function=w, function_domain="spatial", shift=False,
        )
        p2 = dummy_signal.inav[0, 0].data
        assert not np.allclose(p, p2, atol=1e-1)

        # What Barnes' FFT filter does is the same as correlating the
        # spatial kernel with the pattern, using
        # scipy.ndimage.correlate()
        p3 = correlate(input=p, weights=w)

        # We rescale intensities afterwards, so the same must be done
        # here, using skimage.exposure.rescale_intensity()
        p3 = rescale_intensity(p3, out_range=p3.dtype.type)

        assert np.allclose(p2, p3)

    def test_fft_filter_raises(self, dummy_signal):
        function_domain = "Underdark"
        with pytest.raises(ValueError, match=f"{function_domain} must be "):
            dummy_signal.fft_filter(
                transfer_function=np.arange(9).reshape((3, 3)) / 9,
                function_domain=function_domain,
            )

    def test_fft_filter_lazy(self, dummy_signal):
        lazy_signal = dummy_signal.as_lazy()
        w = np.arange(9).reshape(lazy_signal.axes_manager.signal_shape)
        lazy_signal.fft_filter(
            transfer_function=w, function_domain="frequency", shift=False
        )

        assert isinstance(lazy_signal, LazyEBSD)
        assert lazy_signal.data.dtype == dummy_signal.data.dtype


class TestNormalizeIntensityEBSD:
    @pytest.mark.parametrize(
        "num_std, divide_by_square_root, dtype_out, answer",
        [
            (
                1,
                True,
                np.float32,
                np.array(
                    [
                        [0.0653, 0.2124, 0.0653],
                        [0.3595, 0.2124, 0.0653],
                        [0.2124, -0.5229, -0.6700],
                    ]
                ),
            ),
            (
                2,
                True,
                np.float32,
                np.array(
                    [
                        [0.0326, 0.1062, 0.0326],
                        [0.1797, 0.1062, 0.0326],
                        [0.1062, -0.2614, -0.3350],
                    ]
                ),
            ),
            (
                1,
                False,
                np.float32,
                np.array(
                    [
                        [0.1961, 0.6373, 0.1961],
                        [1.0786, 0.6373, 0.1961],
                        [0.6373, -1.5689, -2.0101],
                    ]
                ),
            ),
            (1, False, None, np.array([[0, 0, 0], [1, 0, 0], [0, -1, -2]])),
        ],
    )
    def test_normalize_intensity(
        self, dummy_signal, num_std, divide_by_square_root, dtype_out, answer
    ):
        int16 = np.int16
        if dtype_out is None:
            dummy_signal.data = dummy_signal.data.astype(int16)

        dummy_signal.normalize_intensity(
            num_std=num_std,
            divide_by_square_root=divide_by_square_root,
            dtype_out=dtype_out,
        )

        if dtype_out is None:
            dtype_out = int16
        else:
            assert np.allclose(np.mean(dummy_signal.data), 0, atol=1e-6)

        assert isinstance(dummy_signal, EBSD)
        assert dummy_signal.data.dtype == dtype_out
        assert np.allclose(dummy_signal.inav[0, 0].data, answer, atol=1e-4)

    def test_normalize_intensity_lazy(self, dummy_signal):
        dummy_signal.data = dummy_signal.data.astype(np.float32)
        lazy_signal = dummy_signal.as_lazy()

        lazy_signal.normalize_intensity()

        assert isinstance(lazy_signal, LazyEBSD)
        assert np.allclose(np.mean(lazy_signal.data.compute()), 0, atol=1e-6)


class TestEBSDxmapProperty:
    def test_init_xmap(self, dummy_signal):
        """The attribute is set correctly."""
        assert dummy_signal.xmap is None

        ssim = load(EMSOFT_FILE)
        xmap = ssim.xmap
        assert isinstance(xmap, CrystalMap)
        assert xmap.phases[0].name == "ni"


class TestEBSDdetectorProperty:
    def test_init_detector(self):
        """The attribute is set correctly."""
        pass


class TestPatternMatching:
    def test_match_patterns(self, dummy_signal):
        """Scores are all 1.0 for a dictionary containing all patterns
        from dummy_signal().
        """
        s_dict = EBSD(dummy_signal.data.reshape(-1, 3, 3))
        s_dict._xmap = CrystalMap(Rotation(np.zeros((9, 4))))
        xmap = dummy_signal.match_patterns(s_dict)

        assert np.allclose(xmap.scores[:, 0], 1)
