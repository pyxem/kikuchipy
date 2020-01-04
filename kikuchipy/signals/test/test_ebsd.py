# -*- coding: utf-8 -*-
# Copyright 2019-2020 The KikuchiPy developers
#
# This file is part of KikuchiPy.
#
# KikuchiPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# KikuchiPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with KikuchiPy. If not, see <http://www.gnu.org/licenses/>.

import os

import dask.array as da
import hyperspy.api as hs
from hyperspy.misc.utils import DictionaryTreeBrowser
import matplotlib
import numpy as np
import pytest

import kikuchipy as kp

matplotlib.use("Agg")  # For plotting

DIR_PATH = os.path.dirname(__file__)
KIKUCHIPY_FILE = os.path.join(DIR_PATH, "../../data/kikuchipy/patterns.h5")


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
            if isinstance(output_dict[key], np.ndarray):
                assert input_dict[key].all() == output_dict[key].all()
            else:
                assert input_dict[key] == output_dict[key]


class TestEBSD:
    def test_init(self):
        # Signal shape
        array0 = np.zeros(shape=(10, 10, 10, 10))
        s0 = kp.signals.EBSD(array0)
        assert array0.shape == s0.axes_manager.shape
        # Cannot initialise signal with one signal dimension
        with pytest.raises(ValueError):
            kp.signals.EBSD(np.zeros(10))
        # Shape of one-pattern signal
        array1 = np.zeros(shape=(10, 10))
        s1 = kp.signals.EBSD(array1)
        assert array1.shape == s1.axes_manager.shape
        # SEM metadata
        kp_md = kp.util.io.kikuchipy_metadata()
        sem_node = kp.util.io.metadata_nodes(ebsd=False)
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
        assert lazy_signal.__class__ == kp.signals.LazyEBSD
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
        ebsd_node = kp.util.io.metadata_nodes(sem=False)
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
        assert x.units, y.units == "\u03BC" + "m"

    def test_set_detector_calibration(self, dummy_signal):
        delta = 70
        dummy_signal.set_detector_calibration(delta=delta)
        dx, dy = dummy_signal.axes_manager.signal_axes
        centre = np.array(dummy_signal.axes_manager.signal_shape) / 2 * delta
        assert dx.units, dy.units == "\u03BC" + "m"
        assert dx.scale, dy.scale == delta
        assert dx.offset, dy.offset == -centre


class TestIntensityCorrection:
    @pytest.mark.parametrize(
        "operation, relative",
        [
            ("subtract", False),
            ("subtract", True),
            ("divide", False),
            ("divide", True),
        ],
    )
    def test_static_background_correction(
        self, dummy_signal, dummy_background, operation, relative
    ):
        """This test uses a hard-coded answer. If specifically
        improvements to the intensities produced by this correction is
        to be made, these hard-coded answers will have to be
        recalculated for the tests to pass.
        """

        dummy_signal.static_background_correction(
            operation=operation, relative=relative, static_bg=dummy_background
        )
        if operation == "subtract" and relative is True:
            # fmt: off
            answer = np.array(
                [
                    115, 162, 115, 185, 185, 139, 162, 46, 46, 208, 185, 185,
                    185, 46, 208, 208, 185, 185, 0, 92, 69, 139, 92, 231, 92,
                    92, 255, 185, 46, 162, 162, 139, 208, 208, 69, 92, 92, 23,
                    208, 23, 69, 23, 69, 69, 162, 185, 162, 0, 115, 208, 185,
                    185, 162, 46, 92, 46, 139, 92, 139, 23, 46, 46, 46, 115,
                    231, 185, 115, 185, 23, 69, 231, 92, 208, 115, 69, 162, 162,
                    162, 69, 139, 255
                ]
            )
            # fmt: on
        elif operation == "subtract" and relative is False:
            # fmt: off
            answer = np.array(
                [
                    127, 212, 127, 255, 255, 170, 212, 0, 0, 255, 218, 218, 218,
                    0, 255, 255, 218, 218, 0, 92, 69, 139, 92, 231, 92, 92, 255,
                    218, 0, 182, 182, 145, 255, 255, 36, 72, 95, 0, 255, 0, 63,
                    0, 63, 63, 191, 226, 198, 0, 141, 255, 226, 226, 198, 56,
                    153, 51, 255, 153, 255, 0, 51, 51, 51, 113, 255, 198, 113,
                    198, 0, 56, 255, 85, 191, 63, 0, 127, 127, 127, 0, 95, 255
                ]
            )
            # fmt: on
        elif operation == "divide" and relative is True:
            # fmt: off
            answer = np.array(
                [
                    127, 191, 127, 223, 255, 159, 191, 31, 0, 229, 223, 204,
                    223, 0, 255, 255, 223, 255, 0, 63, 50, 106, 56, 191, 63, 63,
                    255, 196, 0, 167, 182, 157, 255, 255, 36, 60, 113, 0, 255,
                    0, 47, 0, 70, 70, 236, 174, 163, 0, 109, 255, 191, 191, 163,
                    0, 153, 47, 229, 143, 255, 0, 47, 47, 0, 113, 255, 181, 113,
                    226, 0, 56, 255, 75, 132, 51, 10, 102, 119, 102, 0, 76, 255
                ]
            )
            # fmt: on
        else:  # operation == 'divide' and relative is False
            # fmt: off
            answer = np.array(
                [
                    85, 127, 85, 148, 170, 106, 127, 21, 0, 153, 148, 136, 148,
                    0, 170, 170, 148, 170, 0, 63, 50, 106, 56, 191, 63, 63, 255,
                    136, 21, 118, 127, 113, 170, 170, 42, 56, 68, 0, 153, 0, 28,
                    0, 42, 42, 141, 136, 127, 0, 85, 198, 148, 148, 127, 0, 68,
                    21, 101, 63, 113, 0, 21, 21, 0, 85, 191, 136, 85, 170, 0,
                    42, 191, 56, 153, 85, 50, 127, 141, 127, 42, 106, 255
                ]
            )
            # fmt: on
        answer = answer.reshape((3, 3, 3, 3)).astype(np.uint8)
        assert dummy_signal.data.all() == answer.all()

    @pytest.mark.parametrize(
        "static_bg, error, match",
        [
            (
                np.ones((3, 3), dtype=np.int8),
                ValueError,
                "Static background dtype_out",
            ),
            (None, OSError, "Static background is not a numpy or dask array"),
            (np.ones((3, 2), dtype=np.uint8), OSError, "Pattern"),
        ],
    )
    def test_incorrect_static_background_pattern(
        self, dummy_signal, static_bg, error, match
    ):
        """Test for expected error messages when passing an incorrect
        static background pattern to `static_background_correction().`
        """

        ebsd_node = kp.util.io.metadata_nodes(sem=False)
        dummy_signal.metadata.set_item(
            ebsd_node + ".static_background", static_bg
        )
        with pytest.raises(error, match=match):
            dummy_signal.static_background_correction()

    def test_lazy_static_background_correction(
        self, dummy_signal, dummy_background
    ):
        dummy_signal = dummy_signal.as_lazy()
        dummy_signal.static_background_correction(static_bg=dummy_background)
        assert isinstance(dummy_signal.data, da.Array)

    @pytest.mark.parametrize(
        "operation, sigma",
        [("subtract", 2), ("subtract", 3), ("divide", 2), ("divide", 3)],
    )
    def test_dynamic_background_correction(
        self, dummy_signal, operation, sigma
    ):
        """This test uses a hard-coded answer. If specifically
        improvements to the intensities produced by this correction is
        to be made, these hard-coded answers will have to be
        recalculated for the tests to pass.
        """

        dummy_signal.dynamic_background_correction(
            operation=operation, sigma=sigma
        )
        if operation == "subtract" and sigma == 2:
            # fmt: off
            answer = np.array(
                [
                    182, 218, 182, 255, 218, 182, 218, 36, 0, 255, 191, 223,
                    223, 0, 223, 255, 223, 191, 0, 85, 85, 141, 56, 255, 85, 85,
                    255, 255, 0, 218, 182, 109, 255, 255, 36, 36, 113, 0, 255,
                    0, 28, 0, 56, 56, 141, 255, 191, 0, 127, 223, 223, 223, 191,
                    0, 170, 42, 255, 127, 170, 0, 42, 42, 0, 141, 255, 226, 113,
                    170, 0, 56, 255, 56, 255, 72, 36, 145, 109, 145, 0, 109, 255
                ]
            )
            # fmt: on
        elif operation == "subtract" and sigma == 3:
            # fmt: off
            answer = np.array(
                [
                    182, 218, 218, 255, 218, 182, 218, 36, 0, 255, 191, 223,
                    191, 0, 223, 223, 223, 159, 0, 85, 85, 141, 56, 255, 85, 85,
                    255, 255, 0, 218, 182, 109, 255, 255, 36, 36, 141, 28, 255,
                    28, 56, 0, 85, 85, 141, 255, 191, 0, 127, 223, 223, 223,
                    191, 0, 170, 42, 255, 127, 170, 0, 42, 42, 0, 141, 255, 226,
                    113, 170, 0, 56, 255, 56, 255, 72, 36, 145, 109, 109, 0,
                    109, 218
                ]
            )
            # fmt: on
        elif operation == "divide" and sigma == 2:
            # fmt: off
            answer = np.array(
                [
                    182, 218, 182, 255, 218, 182, 218, 36, 0, 191, 148, 170,
                    223, 0, 170, 255, 223, 191, 0, 85, 85, 141, 56, 255, 85, 85,
                    255, 255, 0, 218, 182, 109, 255, 255, 36, 36, 113, 0, 255,
                    0, 28, 0, 56, 56, 141, 255, 191, 0, 127, 223, 223, 223, 191,
                    0, 170, 42, 255, 127, 170, 0, 42, 42, 0, 141, 255, 226, 113,
                    170, 0, 56, 255, 56, 255, 72, 36, 145, 109, 145, 0, 109, 255
                ],
            )
            # fmt: on
        else:  # operation == 'divide' and sigma == 3:
            # fmt: off
            answer = np.array(
                [
                    182, 218, 242, 255, 218, 182, 218, 36, 0, 255, 198, 226,
                    198, 0, 226, 226, 237, 170, 0, 85, 85, 141, 56, 255, 85, 85,
                    255, 255, 0, 218, 182, 109, 255, 255, 36, 36, 226, 0, 255,
                    0, 56, 0, 113, 113, 141, 255, 191, 0, 127, 223, 223, 223,
                    191, 0, 170, 42, 255, 127, 170, 0, 42, 42, 0, 141, 255, 226,
                    113, 170, 0, 56, 255, 56, 255, 72, 36, 145, 109, 101, 0,
                    109, 189
                ]
            )
            # fmt: on
        answer = answer.reshape((3, 3, 3, 3)).astype(np.uint8)
        assert dummy_signal.data.all() == answer.all()
        assert dummy_signal.data.dtype == answer.dtype

    def test_lazy_dynamic_background_correction(self, dummy_signal):
        dummy_signal = dummy_signal.as_lazy()
        dummy_signal.dynamic_background_correction()
        assert isinstance(dummy_signal.data, da.Array)

    @pytest.mark.parametrize(
        "relative, dtype_out",
        [(True, None), (True, np.float32), (False, None), (False, np.float32)],
    )
    def test_rescale_intensities(self, dummy_signal, relative, dtype_out):
        """This test uses a hard-coded answer. If specifically
        improvements to the intensities produced by this correction is
        to be made, these hard-coded answers will have to be
        recalculated for the tests to pass.
        """

        dummy_signal.rescale_intensities(relative=relative, dtype_out=dtype_out)
        if relative is True and dtype_out is None:
            # fmt: off
            answer = np.array(
                [
                    141, 170, 141, 198, 170, 141, 170, 28, 0, 255, 198, 226,
                    198, 0, 226, 226, 198, 170, 0, 85, 85, 141, 56, 255, 85, 85,
                    255, 226, 28, 198, 170, 113, 226, 226, 56, 56, 113, 0, 255,
                    0, 28, 0, 56, 56, 141, 226, 170, 0, 113, 198, 198, 198, 170,
                    0, 113, 28, 170, 85, 113, 0, 28, 28, 0, 141, 255, 226, 113,
                    170, 0, 56, 255, 56, 255, 113, 85, 170, 141, 170, 56, 141,
                    255
                ],
                dtype=np.uint8,
            )
            # fmt: on
        elif relative is True and dtype_out == np.float32:
            # fmt: off
            answer = np.array(
                [
                    0.5555556, 0.6666667, 0.5555556, 0.7777778, 0.6666667,
                    0.5555556, 0.6666667, 0.11111111, 0., 1., 0.7777778,
                    0.8888889, 0.7777778, 0., 0.8888889, 0.8888889, 0.7777778,
                    0.6666667, 0., 0.33333334, 0.33333334, 0.5555556,
                    0.22222222, 1., 0.33333334, 0.33333334, 1., 0.8888889,
                    0.11111111, 0.7777778, 0.6666667, 0.44444445, 0.8888889,
                    0.8888889, 0.22222222, 0.22222222, 0.44444445, 0., 1., 0.,
                    0.11111111, 0., 0.22222222, 0.22222222, 0.5555556,
                    0.8888889, 0.6666667, 0., 0.44444445, 0.7777778, 0.7777778,
                    0.7777778, 0.6666667, 0., 0.44444445, 0.11111111, 0.6666667,
                    0.33333334, 0.44444445, 0., 0.11111111, 0.11111111, 0.,
                    0.5555556, 1., 0.8888889, 0.44444445, 0.6666667, 0.,
                    0.22222222, 1., 0.22222222, 1., 0.44444445, 0.33333334,
                    0.6666667, 0.5555556, 0.6666667, 0.22222222, 0.5555556, 1.
                ],
                dtype=np.float32
            )
            # fmt: on
        elif relative is False and dtype_out is None:
            # fmt: off
            answer = np.array(
                [
                    182, 218, 182, 255, 218, 182, 218, 36, 0, 255, 198, 226,
                    198, 0, 226, 226, 198, 170, 0, 85, 85, 141, 56, 255, 85, 85,
                    255, 255, 0, 218, 182, 109, 255, 255, 36, 36, 113, 0, 255,
                    0, 28, 0, 56, 56, 141, 255, 191, 0, 127, 223, 223, 223, 191,
                    0, 170, 42, 255, 127, 170, 0, 42, 42, 0, 141, 255, 226, 113,
                    170, 0, 56, 255, 56, 255, 72, 36, 145, 109, 145, 0, 109,
                    255
                ],
                dtype=np.uint8
            )
            # fmt: on
        else:  # relative is False and dtype_out == np.float32
            # fmt: off
            answer = np.array(
                [
                    0.71428573, 0.85714287, 0.71428573, 1., 0.85714287,
                    0.71428573, 0.85714287, 0.14285715, 0., 1., 0.7777778,
                    0.8888889, 0.7777778, 0., 0.8888889, 0.8888889, 0.7777778,
                    0.6666667, 0., 0.33333334, 0.33333334, 0.5555556,
                    0.22222222, 1., 0.33333334, 0.33333334, 1., 1., 0.,
                    0.85714287, 0.71428573, 0.42857143, 1., 1., 0.14285715,
                    0.14285715, 0.44444445, 0., 1., 0., 0.11111111, 0.,
                    0.22222222, 0.22222222, 0.5555556, 1., 0.75, 0., 0.5,
                    0.875, 0.875, 0.875, 0.75, 0., 0.6666667, 0.16666667, 1.,
                    0.5, 0.6666667, 0., 0.16666667, 0.16666667, 0., 0.5555556,
                    1., 0.8888889, 0.44444445, 0.6666667, 0., 0.22222222, 1.,
                    0.22222222, 1., 0.2857143, 0.14285715, 0.5714286,
                    0.42857143, 0.5714286, 0., 0.42857143, 1.
                ],
                dtype=np.float32
            )
            # fmt: on
        answer = answer.reshape((3, 3, 3, 3))
        assert dummy_signal.data.all() == answer.all()
        assert dummy_signal.data.dtype == answer.dtype

    def test_lazy_rescale_intensities(self, dummy_signal):
        dummy_signal = dummy_signal.as_lazy()
        dummy_signal.rescale_intensities()
        assert isinstance(dummy_signal.data, da.Array)

    def test_adaptive_histogram_equalization(self):
        """Test setup of equalization only. Tests of the result of the
        actual equalization are found elsewhere.
        """

        s = kp.load(KIKUCHIPY_FILE)

        # These kernel sizes should work without issue
        for kernel_size in [None, 10]:
            s.adaptive_histogram_equalization(kernel_size=kernel_size)

        # These kernel sizes should throw errors
        with pytest.raises(ValueError, match="invalid literal for int()"):
            s.adaptive_histogram_equalization(kernel_size=("wrong", "size"))
        with pytest.raises(ValueError, match="Incorrect value of `kernel_size"):
            s.adaptive_histogram_equalization(kernel_size=(10, 10, 10))

    def test_lazy_adaptive_histogram_equalization(self):
        s = kp.load(KIKUCHIPY_FILE, lazy=True)
        s.adaptive_histogram_equalization()
        assert isinstance(s.data, da.Array)


class TestVirtualBackscatterElectronImaging:
    def test_virtual_backscatter_electron_imaging(self, dummy_signal):
        roi = hs.roi.RectangularROI(left=0, top=0, right=1, bottom=1)
        dummy_signal.virtual_backscatter_electron_imaging(roi)

    def test_virtual_image(self, dummy_signal):
        roi = hs.roi.RectangularROI(left=0, top=0, right=1, bottom=1)
        virtual_image_signal = dummy_signal.get_virtual_image(roi)
        assert (
            virtual_image_signal.data.shape
            == dummy_signal.axes_manager.navigation_shape
        )


class TestDecomposition:
    def test_decomposition(self, dummy_signal):
        dummy_signal.change_dtype(np.float32)
        dummy_signal.decomposition()
        assert isinstance(dummy_signal, kp.signals.EBSD)

    def test_lazy_decomposition(self, dummy_signal):
        lazy_signal = dummy_signal.as_lazy()
        lazy_signal.change_dtype(np.float32)
        lazy_signal.decomposition()
        assert isinstance(lazy_signal, kp.signals.LazyEBSD)

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
        dummy_signal.decomposition(algorithm="svd")

        # Get decomposition model
        model_signal = dummy_signal.get_decomposition_model(
            components=components, dtype_out=dtype_out
        )

        # Check data shape, signal class and pattern intensities in model
        # signal
        assert model_signal.data.shape == dummy_signal.data.shape
        assert isinstance(model_signal, kp.signals.EBSD)
        np.testing.assert_almost_equal(
            model_signal.data.mean(), mean_intensity, decimal=3
        )

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

        # Check data shape, signal class and pattern intensities in model
        # signal after rescaling to 8 bit unsigned integer
        assert model_signal.data.shape == lazy_signal.data.shape
        assert isinstance(model_signal, kp.signals.LazyEBSD)
        model_signal.rescale_intensities(relative=True, dtype_out=np.uint8)
        model_mean = model_signal.data.mean().compute()
        np.testing.assert_almost_equal(model_mean, mean_intensity, decimal=1)

    @pytest.mark.parametrize(
        "components, mean_intensity", [(None, 132.1975), (3, 123.0987)]
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
        fname_out = "test.h5"
        lazy_signal.get_decomposition_model_write(
            components=components, dir_out=tmp_path, fname_out=fname_out
        )
        s_reload = kp.load(os.path.join(tmp_path, fname_out))

        # ... data type, data shape and mean intensity
        assert s_reload.data.dtype == lazy_signal.data.dtype
        assert s_reload.data.shape == lazy_signal.data.shape
        np.testing.assert_almost_equal(
            s_reload.data.mean(), mean_intensity, decimal=4
        )

    def test_rebin(self, dummy_signal):
        ebsd_node = kp.util.io.metadata_nodes(sem=False)

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
        assert isinstance(s2, kp.signals.LazyEBSD)
        assert s2.axes_manager.shape == tuple(expected_new_shape)
        assert s2.metadata.get_item(ebsd_node + ".binning") == float(scale[3])
        assert s3 is None

    def test_compute(self, dummy_signal):
        lazy_signal = dummy_signal.as_lazy()

        lazy_signal.compute()
        assert isinstance(lazy_signal, kp.signals.EBSD)
        assert lazy_signal._lazy is False
