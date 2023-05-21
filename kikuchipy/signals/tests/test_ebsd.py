# Copyright 2019-2023 The kikuchipy developers
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

import logging
import os

import dask.array as da
import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.ndimage import correlate
from skimage.exposure import rescale_intensity

import kikuchipy as kp
from kikuchipy.conftest import assert_dictionary


DIR_PATH = os.path.dirname(__file__)
NORDIF_FILE = os.path.join(DIR_PATH, "../../data/nordif/Pattern.dat")
KIKUCHIPY_FILE = os.path.join(DIR_PATH, "../../data/kikuchipy_h5ebsd/patterns.h5")
EMSOFT_FILE = os.path.join(DIR_PATH, "../../data/emsoft_ebsd/simulated_ebsd.h5")


class TestEBSD:
    def test_init(self):
        # Signal shape
        array0 = np.zeros(shape=(10, 10, 10, 10))
        s0 = kp.signals.EBSD(array0)
        assert array0.shape == s0.axes_manager.shape

        # Cannot initialise signal with one signal dimension
        with pytest.raises(ValueError):
            _ = kp.signals.EBSD(np.zeros(10))

        # Shape of one-image signal
        array1 = np.zeros(shape=(10, 10))
        s1 = kp.signals.EBSD(array1)
        assert array1.shape == s1.axes_manager.shape

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


class TestEBSDXmapProperty:
    def test_init_xmap(self):
        """The attribute is set correctly."""
        ssim = kp.load(EMSOFT_FILE)
        xmap = ssim.xmap
        assert xmap.phases[0].name == "ni"

    def test_attribute_carry_over_from_lazy(self):
        ssim = kp.load(EMSOFT_FILE, lazy=True)
        xmap_lazy = ssim.xmap.deepcopy()
        assert xmap_lazy.phases[0].name == "ni"

        ssim.compute()
        xmap = ssim.xmap
        assert xmap.phases[0].name == "ni"
        assert np.allclose(xmap.rotations.data, xmap_lazy.rotations.data)

    def test_set_xmap(self, get_single_phase_xmap):
        s = kp.data.nickel_ebsd_large(lazy=True)
        nav_shape = s._navigation_shape_rc
        step_sizes = (1.5, 1.5)

        # Should succeed
        xmap_good = get_single_phase_xmap(nav_shape=nav_shape, step_sizes=step_sizes)
        s.xmap = xmap_good

        # Should fail
        xmap_bad = get_single_phase_xmap(
            nav_shape=nav_shape[::-1], step_sizes=step_sizes
        )

        with pytest.raises(ValueError, match=r"Crystal map shape \(75, 55\) and "):
            s.xmap = xmap_bad

        s.axes_manager["x"].scale = 2
        with pytest.warns(UserWarning, match=r"Crystal map step size\(s\) \[1.5, 1.5"):
            s.xmap = xmap_good

        s2 = s.inav[:, :-2]
        with pytest.raises(ValueError, match=r"Crystal map shape \(55, 75\) and "):
            s2.axes_manager["x"].scale = 1
            s2.axes_manager["x"].name = "x2"
            with pytest.warns(UserWarning, match="The signal navigation axes"):
                s2.xmap = xmap_good

    def test_attribute_carry_over_from_deepcopy(self, get_single_phase_xmap):
        s = kp.data.nickel_ebsd_small(lazy=True)
        nav_axes = s.axes_manager.navigation_axes[::-1]
        nav_shape = tuple(a.size for a in nav_axes)
        nav_scales = tuple(a.scale for a in nav_axes)

        xmap = get_single_phase_xmap(nav_shape=nav_shape, step_sizes=nav_scales)
        s.xmap = xmap

        s2 = s.deepcopy()
        r1 = s.xmap.rotations.data
        r2 = s2.xmap.rotations.data
        assert not np.may_share_memory(r1, r2)
        assert np.allclose(r1, r2)

        s._static_background = -1
        s3 = s.deepcopy()
        assert s3.static_background == -1


class TestEBSDDetectorProperty:
    def test_attribute_carry_over_from_deepcopy(self, dummy_signal):
        dummy_signal2 = dummy_signal.deepcopy()

        pc1 = dummy_signal.detector.pc
        pc2 = dummy_signal2.detector.pc
        assert not np.may_share_memory(pc1, pc2)
        assert np.allclose(pc1, pc2)

    def test_attribute_carry_over_from_lazy(self, dummy_signal):
        dummy_signal_lazy = dummy_signal.deepcopy().as_lazy()
        dummy_signal_lazy.compute()
        pc = dummy_signal.detector.pc
        pc_lazy = dummy_signal_lazy.detector.pc
        assert not np.may_share_memory(pc, pc_lazy)
        assert np.allclose(pc, pc_lazy)

    def test_set_detector(self):
        s = kp.data.nickel_ebsd_small(lazy=True)
        sig_shape = s._signal_shape_rc

        # Success
        detector_good = kp.detectors.EBSDDetector(shape=sig_shape)
        s.detector = detector_good

        # Failure
        with pytest.raises(ValueError, match=r"Detector shape \(59, 60\) must be "):
            s.detector = kp.detectors.EBSDDetector(shape=(59, 60))
        with pytest.raises(ValueError, match="Detector must have exactly one "):
            s.detector = kp.detectors.EBSDDetector(
                shape=sig_shape, pc=np.ones((3, 4, 3))
            )

    @pytest.mark.parametrize(
        "detector, signal_nav_shape, compatible, error_msg_start",
        [
            (((1,), (5, 5)), (2, 3), True, None),
            (((2, 3), (5, 5)), (2, 3), True, None),
            (((1,), (5, 4)), (2, 3), False, r"Detector shape \(5, 4\) must be equal "),
            (((3, 2), (5, 5)), (2, 3), False, "Detector must have exactly"),
            (((2, 3), (5, 5)), (), False, "Detector must have exactly"),
        ],
        indirect=["detector"],
    )
    def test_compatible_with_signal(
        self, detector, signal_nav_shape, compatible, error_msg_start
    ):
        s = kp.signals.EBSD(np.ones(signal_nav_shape + (5, 5), dtype=int))
        func_kwargs = dict(
            detector=detector,
            nav_shape=s._navigation_shape_rc,
            sig_shape=s._signal_shape_rc,
        )
        assert (
            kp.signals.util._detector._detector_is_compatible_with_signal(**func_kwargs)
            == compatible
        )
        if not compatible:
            with pytest.raises(ValueError, match=error_msg_start):
                kp.signals.util._detector._detector_is_compatible_with_signal(
                    raise_if_not=True, **func_kwargs
                )

    def test_detector_shape(self):
        s1 = kp.signals.EBSD(np.ones((1, 2, 3, 4)))
        assert s1._navigation_shape_rc == (1, 2)
        assert s1.detector.shape == s1._signal_shape_rc == (3, 4)

        s2 = kp.signals.EBSD(np.ones((1, 2, 4, 3)))
        assert s2._navigation_shape_rc == (1, 2)
        assert s2.detector.shape == s2._signal_shape_rc == (4, 3)


class TestStaticBackgroundProperty:
    def test_background_carry_over_from_deepcopy(self, dummy_signal):
        dummy_signal2 = dummy_signal.deepcopy()
        bg1 = dummy_signal.static_background
        bg2 = dummy_signal2.static_background
        assert not np.may_share_memory(bg1, bg2)
        assert np.allclose(bg1, bg2)

    def test_background_carry_over_from_lazy(self, dummy_signal):
        dummy_signal_lazy = dummy_signal.deepcopy().as_lazy()
        assert isinstance(dummy_signal_lazy.static_background, da.Array)
        dummy_signal_lazy.compute()
        bg = dummy_signal.static_background
        bg_lazy = dummy_signal_lazy.static_background
        assert isinstance(bg_lazy, np.ndarray)
        assert not np.may_share_memory(bg, bg_lazy)
        assert np.allclose(bg, bg_lazy)

    def test_set_background(self):
        s = kp.data.nickel_ebsd_small(lazy=True)
        sig_shape = s._signal_shape_rc
        # Success
        bg_good = np.arange(np.prod(sig_shape), dtype=s.data.dtype).reshape(sig_shape)
        s.static_background = bg_good
        # Warns, but allows
        with pytest.warns(
            UserWarning,
            match="Background pattern has different data type from patterns",
        ):
            dtype = np.uint16
            s.static_background = bg_good.astype(dtype)
            assert s.static_background.dtype == dtype
        with pytest.warns(
            UserWarning, match="Background pattern has different shape from patterns"
        ):
            s.static_background = bg_good[:, :-2]
            assert s.static_background.shape == (sig_shape[0], sig_shape[1] - 2)


class TestRemoveStaticBackgroundEBSD:
    @pytest.mark.parametrize(
        "operation, answer",
        [
            (
                "subtract",
                # fmt: off
                np.array(
                    [
                        127, 212, 127, 255, 255, 170, 212, 0, 0, 255, 218, 218, 218, 0,
                        255, 255, 218, 218, 0, 92, 69, 139, 92, 231, 92, 92, 255, 218,
                        0, 182, 182, 145, 255, 255, 36, 72, 95, 0, 255, 0, 63, 0, 63,
                        63, 191, 226, 198, 0, 141, 255, 226, 226, 198, 56, 153, 51, 255,
                        153, 255, 0, 51, 51, 51, 113, 255, 198, 113, 198, 0, 56, 255,
                        85, 191, 63, 0, 127, 127, 127, 0, 95, 255
                    ]
                ),
                # fmt: on
            ),
            (
                "divide",
                # fmt: off
                np.array(
                    [
                        127, 191, 127, 223, 255, 159, 191, 31, 0, 229, 223, 204, 223, 0,
                        255, 255, 223, 255, 0, 63, 51, 106, 56, 191, 63, 63, 255, 196,
                        0, 167, 182, 157, 255, 255, 36, 60, 113, 0, 255, 0, 47, 0, 70,
                        70, 236, 174, 163, 0, 109, 255, 191, 191, 163, 0, 153, 47, 229,
                        143, 255, 0, 47, 47, 0, 113, 255, 181, 113, 226, 0, 56, 255, 75,
                        132, 51, 10, 102, 119, 102, 0, 76, 255
                    ]
                )
                # fmt: on
            ),
        ],
    )
    def test_remove_static_background(
        self, dummy_signal, dummy_background, operation, answer
    ):
        """This tests uses a hard-coded answer. If specifically
        improvements to the intensities produced by this correction is
        to be made, these hard-coded answers will have to be
        recalculated for the tests to pass.
        """

        dummy_signal.remove_static_background(
            operation=operation, static_bg=dummy_background
        )
        answer = answer.reshape((3, 3, 3, 3)).astype(np.uint8)
        assert np.allclose(dummy_signal.data, answer)

    @pytest.mark.parametrize(
        "static_bg, error, match",
        [
            (np.ones((3, 3), dtype=np.int8), ValueError, "Static background dtype_out"),
            (
                None,
                ValueError,
                "`EBSD.static_background` is not a valid array",
            ),
            (np.ones((3, 2), dtype=np.uint8), ValueError, "Signal"),
        ],
    )
    def test_incorrect_static_background_pattern(
        self, dummy_signal, static_bg, error, match
    ):
        """Test for expected error messages when passing an incorrect
        static background pattern to `remove_static_background().`
        """
        # Circumvent setter of static_background
        dummy_signal._static_background = static_bg
        with pytest.raises(error, match=match):
            dummy_signal.remove_static_background()

    def test_lazy_remove_static_background(self, dummy_signal, dummy_background):
        dummy_signal = dummy_signal.as_lazy()
        dummy_signal.remove_static_background(static_bg=dummy_background)
        assert isinstance(dummy_signal.data, da.Array)
        dummy_signal.static_background = da.from_array(dummy_background)
        dummy_signal.remove_static_background()
        dummy_signal.remove_static_background(static_bg=dummy_signal.static_background)

    def test_remove_static_background_scalebg(self, dummy_signal, dummy_background):
        dummy_signal2 = dummy_signal.deepcopy()
        dummy_signal.remove_static_background(scale_bg=True, static_bg=dummy_background)
        dummy_signal2.remove_static_background(
            scale_bg=False, static_bg=dummy_background
        )

        p1 = dummy_signal.inav[0, 0].data
        p2 = dummy_signal2.inav[0, 0].data

        assert not np.allclose(p1, p2, atol=0.1)
        assert np.allclose(p1, np.array([[15, 150, 15], [180, 255, 120], [150, 0, 75]]))

    def test_non_square_patterns(self):
        s = kp.data.nickel_ebsd_small()
        s = s.isig[:, :-5]  # Remove bottom five rows
        static_bg = s.mean(axis=(0, 1))
        static_bg.change_dtype(np.uint8)
        s.remove_static_background(static_bg=static_bg.data)

    def test_inplace(self, dummy_signal):
        # Current signal is unaffected
        s = dummy_signal.deepcopy()
        s2 = dummy_signal.remove_static_background(inplace=False)
        assert np.allclose(s.data, dummy_signal.data)

        # Custom properties carry over
        assert isinstance(s2, kp.signals.EBSD)
        assert np.allclose(s2.static_background, dummy_signal.static_background)
        assert np.allclose(s2.detector.pc, dummy_signal.detector.pc)
        assert np.allclose(s2.xmap.rotations.data, dummy_signal.xmap.rotations.data)

        # Operating on current signal gives same result as output
        dummy_signal.remove_static_background()
        assert np.allclose(s2.data, dummy_signal.data)

        # Operating on lazy signal returns lazy signal
        s3 = s.as_lazy()
        s4 = s3.remove_static_background(inplace=False)
        assert isinstance(s4, kp.signals.LazyEBSD)
        s4.compute()
        assert np.allclose(s4.data, dummy_signal.data)

    def test_lazy_output(self, dummy_signal):
        with pytest.raises(
            ValueError, match="`lazy_output=True` requires `inplace=False`"
        ):
            _ = dummy_signal.remove_static_background(lazy_output=True)

        s2 = dummy_signal.remove_static_background(inplace=False, lazy_output=True)
        assert isinstance(s2, kp.signals.LazyEBSD)

        s3 = dummy_signal.as_lazy()
        s4 = s3.remove_static_background(inplace=False, lazy_output=False)
        assert isinstance(s4, kp.signals.EBSD)


class TestRemoveDynamicBackgroundEBSD:
    @pytest.mark.parametrize(
        "operation, std, answer",
        [
            (
                "subtract",
                2,
                # fmt: off
                # Ten numbers on each line
                np.array(
                    [
                        170, 215, 181, 255, 221, 188, 221, 32, 0, 255,
                        198, 228, 199, 0, 230, 229, 201, 174, 0, 84,
                        77, 147, 48, 255, 81, 74, 249, 246, 0, 216,
                        177, 109, 255, 250, 40, 44, 120, 2, 255, 8,
                        32, 0, 67, 63, 145, 254, 195, 0, 120, 229,
                        237, 222, 196, 1, 164, 34, 255, 128, 173, 0,
                        47, 49, 7, 133, 245, 218, 110, 166, 0, 59,
                        255, 60, 255, 71, 35, 145, 108, 144, 0, 108,
                        253,
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
                        181, 218, 182, 255, 218, 182, 218, 36, 0, 255,
                        198, 226, 198, 0, 226, 226, 198, 170, 0, 84,
                        84, 142, 56, 255, 84, 84, 254, 254, 0, 218,
                        181, 109, 255, 254, 36, 36, 113, 0, 255, 0,
                        28, 0, 57, 57, 141, 255, 191, 0, 127, 223,
                        223, 223, 191, 0, 169, 42, 255, 127, 170, 0,
                        42, 42, 0, 141, 254, 226, 113, 169, 0, 56,
                        255, 56, 255, 72, 36, 145, 109, 145, 0, 109,
                        254,
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
                        176, 217, 186, 254, 225, 194, 225, 39, 0, 255,
                        199, 228, 199, 0, 231, 230, 202, 174, 0, 93,
                        88, 159, 60, 255, 91, 86, 245, 241, 0, 214,
                        174, 107, 255, 247, 37, 38, 127, 0, 255, 0,
                        30, 0, 67, 63, 150, 255, 199, 0, 128, 234,
                        244, 224, 201, 0, 166, 42, 254, 133, 180, 0,
                        47, 48, 0, 132, 238, 212, 109, 164, 0, 56,
                        255, 57, 255, 72, 36, 146, 109, 145, 0, 109,
                        252,
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
                        181, 218, 182, 255, 219, 182, 219, 36, 0, 255,
                        198, 226, 198, 0, 226, 226, 198, 170, 0, 85,
                        85, 142, 56, 255, 85, 85, 254, 254, 0, 218,
                        181, 109, 254, 254, 36, 36, 114, 0, 255, 0,
                        28, 0, 57, 57, 142, 255, 191, 0, 127, 223,
                        224, 223, 191, 0, 169, 42, 255, 127, 170, 0,
                        42, 42, 0, 141, 253, 225, 113, 169, 0, 56,
                        254, 56, 255, 72, 36, 145, 109, 145, 0, 109,
                        254,
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
            operation=operation, std=std, filter_domain="spatial"
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
            operation=operation, std=std, filter_domain=filter_domain
        )

        assert dummy_signal.data.dtype == dtype_out
        assert np.allclose(dummy_signal.inav[0, 0].data, answer, atol=1e-4)

    def test_remove_dynamic_background_raises(self, dummy_signal):
        filter_domain = "wildmount"
        with pytest.raises(ValueError, match=f"{filter_domain} must be "):
            dummy_signal.remove_dynamic_background(filter_domain=filter_domain)

    @pytest.mark.filterwarnings("ignore:invalid value")
    def test_inplace(self, dummy_signal):
        # Current signal is unaffected
        s = dummy_signal.deepcopy()
        s2 = dummy_signal.remove_dynamic_background(inplace=False)
        assert np.allclose(s.data, dummy_signal.data)

        # Custom properties carry over
        assert isinstance(s2, kp.signals.EBSD)
        assert np.allclose(s2.static_background, dummy_signal.static_background)
        assert np.allclose(s2.detector.pc, dummy_signal.detector.pc)
        assert np.allclose(s2.xmap.rotations.data, dummy_signal.xmap.rotations.data)

        # Operating on current signal gives same result as output
        dummy_signal.remove_dynamic_background()
        assert np.allclose(s2.data, dummy_signal.data)

        # Operating on lazy signal returns lazy signal
        s3 = s.as_lazy()
        s4 = s3.remove_dynamic_background(inplace=False)
        assert isinstance(s4, kp.signals.LazyEBSD)
        s4.compute()
        assert np.allclose(s4.data, dummy_signal.data)

    @pytest.mark.filterwarnings("ignore:invalid value")
    def test_lazy_output(self, dummy_signal):
        with pytest.raises(
            ValueError, match="`lazy_output=True` requires `inplace=False`"
        ):
            _ = dummy_signal.remove_dynamic_background(lazy_output=True)

        s2 = dummy_signal.remove_dynamic_background(inplace=False, lazy_output=True)
        assert isinstance(s2, kp.signals.LazyEBSD)

        s3 = dummy_signal.as_lazy()
        s4 = s3.remove_dynamic_background(inplace=False, lazy_output=False)
        assert isinstance(s4, kp.signals.EBSD)


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
    def test_rescale_intensity(self, dummy_signal, relative, dtype_out, answer, capsys):
        """This tests uses a hard-coded answer. If specifically
        improvements to the intensities produced by this correction is
        to be made, these hard-coded answers will have to be
        recalculated for the tests to pass.
        """
        dummy_signal.rescale_intensity(
            relative=relative, dtype_out=dtype_out, show_progressbar=True
        )
        out, _ = capsys.readouterr()
        assert "Completed" in out

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
    def test_rescale_intensity_percentiles(self, dummy_signal, percentiles, answer):
        dummy_signal.data = dummy_signal.data.astype(np.float32)
        dtype_out = np.uint8
        dummy_signal.rescale_intensity(percentiles=percentiles, dtype_out=dtype_out)

        assert dummy_signal.data.dtype == dtype_out
        assert np.allclose(dummy_signal.inav[0, 0].data, answer)

    def test_rescale_intensity_in_range(self, dummy_signal):
        dummy_data = dummy_signal.deepcopy().data

        dummy_signal.rescale_intensity()

        assert dummy_signal.data.dtype == dummy_data.dtype
        assert not np.allclose(dummy_signal.data, dummy_data, atol=1)

    def test_rescale_intensity_raises_in_range_percentiles(self, dummy_signal):
        with pytest.raises(ValueError, match="'percentiles' must be None"):
            dummy_signal.rescale_intensity(in_range=(1, 254), percentiles=(1, 99))

    def test_rescale_intensity_raises_in_range_relative(self, dummy_signal):
        with pytest.raises(ValueError, match="'in_range' must be None if "):
            dummy_signal.rescale_intensity(in_range=(1, 254), relative=True)

    def test_inplace(self, dummy_signal):
        # Current signal is unaffected
        s = dummy_signal.deepcopy()
        s2 = dummy_signal.rescale_intensity(inplace=False)
        assert np.allclose(s.data, dummy_signal.data)

        # Custom properties carry over
        assert isinstance(s2, kp.signals.EBSD)
        assert np.allclose(s2.static_background, dummy_signal.static_background)
        assert np.allclose(s2.detector.pc, dummy_signal.detector.pc)
        assert np.allclose(s2.xmap.rotations.data, dummy_signal.xmap.rotations.data)

        # Operating on current signal gives same result as output
        dummy_signal.rescale_intensity()
        assert np.allclose(s2.data, dummy_signal.data)

        # Operating on lazy signal returns lazy signal
        s3 = s.as_lazy()
        s4 = s3.rescale_intensity(inplace=False)
        assert isinstance(s4, kp.signals.LazyEBSD)
        s4.compute()
        assert np.allclose(s4.data, dummy_signal.data)

    def test_lazy_output(self, dummy_signal):
        with pytest.raises(
            ValueError, match="`lazy_output=True` requires `inplace=False`"
        ):
            _ = dummy_signal.rescale_intensity(lazy_output=True)

        s2 = dummy_signal.rescale_intensity(inplace=False, lazy_output=True)
        assert isinstance(s2, kp.signals.LazyEBSD)

        s3 = dummy_signal.as_lazy()
        s4 = s3.rescale_intensity(inplace=False, lazy_output=False)
        assert isinstance(s4, kp.signals.EBSD)


class TestAdaptiveHistogramEqualizationEBSD:
    def test_adaptive_histogram_equalization(self, capsys):
        """Test setup of equalization only. Tests of the result of the
        actual equalization are found elsewhere.
        """
        s = kp.load(KIKUCHIPY_FILE)

        # These window sizes should work without issue
        for kernel_size, show_progressbar in zip([None, 10], [True, False]):
            s.adaptive_histogram_equalization(
                kernel_size=kernel_size, show_progressbar=show_progressbar
            )
            out, _ = capsys.readouterr()
            if show_progressbar:
                assert "Completed" in out
            else:
                assert not out

        # These window sizes should throw errors
        with pytest.raises(ValueError, match="invalid literal for int()"):
            s.adaptive_histogram_equalization(kernel_size=("wrong", "size"))
        with pytest.raises(ValueError, match="Incorrect value of `shape"):
            s.adaptive_histogram_equalization(kernel_size=(10, 10, 10))

    def test_lazy_adaptive_histogram_equalization(self):
        s = kp.load(KIKUCHIPY_FILE, lazy=True)
        s.adaptive_histogram_equalization()
        assert isinstance(s.data, da.Array)

    def test_inplace(self):
        s = kp.data.nickel_ebsd_small()

        # Current signal is unaffected
        s2 = s.deepcopy()
        s3 = s2.adaptive_histogram_equalization(inplace=False)
        assert np.allclose(s.data, s2.data)

        # Custom properties carry over
        assert isinstance(s3, kp.signals.EBSD)
        assert np.allclose(s3.static_background, s.static_background)
        assert np.allclose(s3.detector.pc, s.detector.pc)
        assert np.allclose(s3.xmap.rotations.data, s.xmap.rotations.data)

        # Operating on current signal gives same result as output
        s2.adaptive_histogram_equalization()
        assert np.allclose(s3.data, s2.data)

        # Operating on lazy signal returns lazy signal
        s4 = s.as_lazy()
        s5 = s4.adaptive_histogram_equalization(inplace=False)
        assert isinstance(s5, kp.signals.LazyEBSD)
        s5.compute()
        assert np.allclose(s5.data, s2.data)

    def test_lazy_output(self):
        s = kp.data.nickel_ebsd_small()
        with pytest.raises(
            ValueError, match="`lazy_output=True` requires `inplace=False`"
        ):
            _ = s.adaptive_histogram_equalization(lazy_output=True)

        s2 = s.adaptive_histogram_equalization(inplace=False, lazy_output=True)
        assert isinstance(s2, kp.signals.LazyEBSD)

        s3 = s.as_lazy()
        s4 = s3.adaptive_histogram_equalization(inplace=False, lazy_output=False)
        assert isinstance(s4, kp.signals.EBSD)


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
                # One pattern per line
                np.array(
                    [
                        255, 109, 218, 218, 36, 236, 255, 36, 0,
                        143, 111, 255, 159, 0, 207, 159, 63, 175,
                        135, 119, 34, 119, 0, 255, 153, 119, 102,
                        182, 24, 255, 121, 109, 85, 133, 0, 12,
                        255, 107, 228, 80, 40, 107, 161, 147, 0,
                        204, 0, 51, 51, 51, 229, 25, 76, 255,
                        194, 105, 255, 135, 149, 60, 105, 119, 0,
                        204, 102, 255, 89, 127, 0, 12, 140, 127,
                        255, 185, 0, 69, 162, 46, 0, 208, 0
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
                # One pattern per line
                np.array(
                    [
                        255, 223, 223, 255, 0, 223, 255, 63, 0,
                        109, 145, 145, 200, 0, 255, 163, 54, 127,
                        119, 136, 153, 170, 0, 255, 153, 136, 221,
                        212, 42, 255, 127, 0, 141, 184, 14, 28,
                        210, 45, 180, 135, 0, 255, 210, 15, 30,
                        200, 109, 182, 109, 0, 255, 182, 145, 182,
                        150, 34, 255, 57, 81, 0, 57, 69, 11,
                        255, 38, 191, 63, 114, 38, 51, 89, 0,
                        255, 117, 137, 19, 117, 0, 0, 176, 58
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
                # one pattern per line
                np.array(
                    [
                        218, 46, 255, 139, 0, 150, 194, 3, 11,
                        211, 63, 196, 145, 0, 255, 211, 33, 55,
                        175, 105, 155, 110, 0, 255, 169, 135, 177,
                        184, 72, 255, 112, 59, 62, 115, 55, 0,
                        255, 51, 225, 107, 21, 122, 85, 47, 0,
                        255, 129, 152, 77, 0, 169, 48, 187, 170,
                        153, 36, 255, 63, 86, 0, 57, 69, 4,
                        254, 45, 206, 58, 115, 16, 33, 98, 0,
                        255, 121, 117, 32, 121, 14, 0, 174, 66
                    ],
                ),
                # fmt: on
                {"std": 2},  # standard deviation
            ),
        ],
    )
    def test_average_neighbour_patterns(
        self, dummy_signal, window, window_shape, lazy, answer, kwargs, capsys
    ):
        if lazy:
            dummy_signal = dummy_signal.as_lazy()
        if kwargs is None:
            dummy_signal.average_neighbour_patterns(
                window=window,
                window_shape=window_shape,
                show_progressbar=True,
            )
            out, _ = capsys.readouterr()
            assert "Completed" in out

        else:
            dummy_signal.average_neighbour_patterns(
                window=window, window_shape=window_shape, **kwargs
            )

        d = dummy_signal.data
        if lazy:
            d = d.compute()
        print(d)

        answer = answer.reshape((3, 3, 3, 3)).astype(np.uint8)
        assert np.allclose(dummy_signal.data, answer)
        assert dummy_signal.data.dtype == answer.dtype

    def test_average_neighbour_patterns_no_averaging(self, dummy_signal):
        answer = dummy_signal.data.copy()
        with pytest.warns(UserWarning, match="A window of shape .* was "):
            dummy_signal.average_neighbour_patterns(
                window="rectangular", window_shape=(1, 1)
            )
        assert np.allclose(dummy_signal.data, answer)
        assert dummy_signal.data.dtype == answer.dtype

    def test_average_neighbour_patterns_one_nav_dim(self, dummy_signal):
        dummy_signal_1d = dummy_signal.inav[:, 0]
        dummy_signal_1d.average_neighbour_patterns(window_shape=(3,))
        # fmt: off
        answer = np.array(
            [
                255, 223, 223, 255, 0, 223, 255, 63, 0, 109, 145, 145, 200, 0,
                255, 163, 54, 127, 119, 136, 153, 170, 0, 255, 153, 136, 221
            ],
            dtype=np.uint8
        ).reshape(dummy_signal_1d.axes_manager.shape)
        # fmt: on
        assert np.allclose(dummy_signal_1d.data, answer)
        assert dummy_signal.data.dtype == answer.dtype

    def test_average_neighbour_patterns_window_1d(self, dummy_signal):
        dummy_signal.average_neighbour_patterns(window_shape=(3,))
        # fmt: off
        # One pattern per line
        answer = np.array(
            [
                233, 106, 212, 233, 170, 233, 255, 21, 0,
                191, 95, 255, 95, 0, 111, 143, 127, 159,
                98, 117, 0, 117, 117, 255, 137, 117, 117,
                239, 95, 255, 223, 191, 175, 207, 31, 0,
                155, 127, 255, 56, 0, 14, 70, 155, 85,
                175, 111, 0, 143, 127, 255, 95, 127, 191,
                231, 0, 255, 162, 139, 139, 162, 23, 0,
                135, 135, 255, 60, 105, 0, 60, 165, 105,
                255, 127, 0, 127, 163, 182, 109, 145, 109
            ],
            dtype=np.uint8
        ).reshape(dummy_signal.axes_manager.shape)
        # fmt: on
        assert np.allclose(dummy_signal.data, answer)
        assert dummy_signal.data.dtype == answer.dtype

    def test_average_neighbour_patterns_pass_window(self, dummy_signal):
        w = kp.filters.Window()
        dummy_signal.average_neighbour_patterns(w)
        # fmt: off
        # One pattern per line
        answer = np.array(
            [
                255, 109, 218, 218, 36, 236, 255, 36, 0,
                143, 111, 255, 159, 0, 207, 159, 63, 175,
                135, 119, 34, 119, 0, 255, 153, 119, 102,
                182, 24, 255, 121, 109, 85, 133, 0, 12,
                255, 107, 228, 80, 40, 107, 161, 147, 0,
                204, 0, 51, 51, 51, 229, 25, 76, 255,
                194, 105, 255, 135, 149, 60, 105, 119, 0,
                204, 102, 255, 89, 127, 0, 12, 140, 127,
                255, 185, 0, 69, 162, 46, 0, 208, 0
            ],
            dtype=np.uint8
        ).reshape(dummy_signal.axes_manager.shape)
        # fmt: on
        assert np.allclose(dummy_signal.data, answer)
        assert dummy_signal.data.dtype == answer.dtype

    def test_average_neighbour_patterns_lazy(self):
        """Fixes https://github.com/pyxem/kikuchipy/issues/230."""
        chunks = ((3, 3, 4, 3, 4, 3, 4, 3, 3, 4, 3, 4, 3, 4, 3, 4), (75,), (6,), (6,))
        s = kp.signals.LazyEBSD(da.zeros((55, 75, 6, 6), chunks=chunks, dtype=np.uint8))
        s.average_neighbour_patterns()
        s.compute()

    def test_average_neighbour_patterns_control(self):
        """Compare averaged array to array built up manually.

        Also test Numba function directly.
        """
        shape = (3, 3, 3, 3)
        data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

        # Reference
        data_desired = np.zeros_like(data)
        window_sums = np.array([[3, 4, 3], [4, 5, 4], [3, 4, 3]])
        for i in range(shape[0]):
            for j in range(shape[1]):
                p = np.zeros(shape[2:], dtype=data.dtype)
                for k in [(i - 1, j), (i, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                    if -1 not in k and 3 not in k:
                        p += data[k]
                p /= window_sums[i, j]
                data_desired[i, j] = kp.pattern.rescale_intensity(p)

        # Averaging
        s = kp.signals.EBSD(data)
        s.average_neighbour_patterns()

        assert np.allclose(s.data, data_desired)

        # Test Numba function
        window = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])[:, :, None, None]
        correlated_patterns = correlate(data, weights=window, mode="constant", cval=0)
        rescaled_patterns = (
            kp.pattern.chunk._rescale_neighbour_averaged_patterns.py_func(
                correlated_patterns, window_sums, correlated_patterns.dtype, -1, 1
            )
        )
        assert np.allclose(rescaled_patterns, s.data)

    def test_inplace(self, dummy_signal):
        # Current signal is unaffected
        s2 = dummy_signal.deepcopy()
        s3 = s2.average_neighbour_patterns(inplace=False)
        assert np.allclose(dummy_signal.data, s2.data)

        # Custom properties carry over
        assert isinstance(s3, kp.signals.EBSD)
        assert np.allclose(s3.static_background, dummy_signal.static_background)
        assert np.allclose(s3.detector.pc, dummy_signal.detector.pc)
        assert np.allclose(s3.xmap.rotations.data, dummy_signal.xmap.rotations.data)

        # Operating on current signal gives same result as output
        s2.average_neighbour_patterns()
        assert np.allclose(s3.data, s2.data)

        # Operating on lazy signal returns lazy signal
        s4 = dummy_signal.as_lazy()
        s5 = s4.average_neighbour_patterns(inplace=False)
        assert isinstance(s5, kp.signals.LazyEBSD)
        s5.compute()
        assert np.allclose(s5.data, s2.data)

    def test_lazy_output(self, dummy_signal):
        with pytest.raises(
            ValueError, match="`lazy_output=True` requires `inplace=False`"
        ):
            _ = dummy_signal.average_neighbour_patterns(lazy_output=True)

        s2 = dummy_signal.average_neighbour_patterns(inplace=False, lazy_output=True)
        assert isinstance(s2, kp.signals.LazyEBSD)

        s3 = dummy_signal.as_lazy()
        s4 = s3.average_neighbour_patterns(inplace=False, lazy_output=False)
        assert isinstance(s4, kp.signals.EBSD)


class TestVirtualBackscatterElectronImaging:
    @pytest.mark.parametrize("out_signal_axes", [None, (0, 1), ("x", "y")])
    def test_virtual_backscatter_electron_imaging(self, dummy_signal, out_signal_axes):
        dummy_signal.axes_manager.navigation_axes[0].name = "x"
        dummy_signal.axes_manager.navigation_axes[1].name = "y"

        roi = hs.roi.RectangularROI(left=0, top=0, right=1, bottom=1)
        dummy_signal.plot_virtual_bse_intensity(roi, out_signal_axes=out_signal_axes)

        plt.close("all")

    def test_get_virtual_image(self, dummy_signal):
        roi = hs.roi.RectangularROI(left=0, top=0, right=1, bottom=1)
        virtual_image_signal = dummy_signal.get_virtual_bse_intensity(roi)
        assert (
            virtual_image_signal.data.shape
            == dummy_signal.axes_manager.navigation_shape
        )

    def test_virtual_backscatter_electron_imaging_raises(self, dummy_signal):
        roi = hs.roi.RectangularROI(0, 0, 1, 1)
        with pytest.raises(ValueError):
            _ = dummy_signal.get_virtual_bse_intensity(roi, out_signal_axes=(0, 1, 2))


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
        dummy_signal.decomposition()

        # Get decomposition model
        model_signal = dummy_signal.get_decomposition_model(
            components=components, dtype_out=dtype_out
        )

        # Check data shape, signal class and image intensities in model
        # signal
        assert model_signal.data.shape == dummy_signal.data.shape
        assert isinstance(model_signal, kp.signals.EBSD)
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
        assert isinstance(lazy_signal, kp.signals.LazyEBSD)

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
        assert isinstance(model_signal, kp.signals.LazyEBSD)
        model_signal.rescale_intensity(relative=True, dtype_out=np.uint8)
        model_mean = model_signal.data.mean().compute()
        assert np.allclose(model_mean, mean_intensity, atol=0.1)

    @pytest.mark.parametrize("components, mean_intensity", [(None, 132.1), (3, 122.9)])
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
        s_reload = kp.load(os.path.join(tmp_path, fname_out))

        # ... data type, data shape and mean intensity
        assert s_reload.data.dtype == lazy_signal.data.dtype
        assert s_reload.data.shape == lazy_signal.data.shape
        assert np.allclose(s_reload.data.mean(), mean_intensity, atol=1e-1)


class TestLazy:
    def test_compute(self, dummy_signal):
        lazy_signal = dummy_signal.as_lazy()

        lazy_signal.compute()
        assert isinstance(lazy_signal, kp.signals.EBSD)
        assert lazy_signal._lazy is False

    def test_change_dtype(self, dummy_signal):
        lazy_signal = dummy_signal.as_lazy()

        assert isinstance(lazy_signal, kp.signals.LazyEBSD)
        lazy_signal.change_dtype("uint16")
        assert isinstance(lazy_signal, kp.signals.LazyEBSD)


class TestGetDynamicBackgroundEBSD:
    def test_get_dynamic_background_spatial(self, dummy_signal, capsys):
        dtype_out = dummy_signal.data.dtype
        bg = dummy_signal.get_dynamic_background(
            filter_domain="spatial", std=2, truncate=3, show_progressbar=True
        )
        out, _ = capsys.readouterr()
        assert "Completed" in out

        assert bg.data.dtype == dtype_out
        assert isinstance(bg, kp.signals.EBSD)

    def test_get_dynamic_background_frequency(self, dummy_signal):
        dtype_out = np.float32
        bg = dummy_signal.get_dynamic_background(
            filter_domain="frequency", std=2, truncate=3, dtype_out=dtype_out
        )

        assert bg.data.dtype == dtype_out
        assert isinstance(bg, kp.signals.EBSD)

    def test_get_dynamic_background_raises(self, dummy_signal):
        filter_domain = "Vasselheim"
        with pytest.raises(ValueError, match=f"{filter_domain} must be"):
            _ = dummy_signal.get_dynamic_background(filter_domain=filter_domain)

    def test_get_dynamic_background_lazy(self, dummy_signal):
        lazy_signal = dummy_signal.as_lazy()

        bg = lazy_signal.get_dynamic_background()
        assert isinstance(bg, kp.signals.LazyEBSD)

        bg.compute()
        assert isinstance(bg, kp.signals.EBSD)

    def test_lazy_output(self, dummy_signal):
        s = dummy_signal.get_dynamic_background(lazy_output=True)
        assert isinstance(s, kp.signals.LazyEBSD)

        # Custom properties carry over
        assert np.allclose(s.static_background, dummy_signal.static_background)
        assert np.allclose(s.detector.pc, dummy_signal.detector.pc)
        assert np.allclose(s.xmap.rotations.data, dummy_signal.xmap.rotations.data)

        s2 = dummy_signal.as_lazy()
        s3 = s2.get_dynamic_background()
        assert isinstance(s3, kp.signals.LazyEBSD)

        s4 = s2.get_dynamic_background(lazy_output=False)
        assert isinstance(s4, kp.signals.EBSD)


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
        capsys,
    ):
        if dtype_out is None:
            dtype_out = np.float32
        dummy_signal.data = dummy_signal.data.astype(dtype_out)

        shape = dummy_signal.axes_manager.signal_shape
        w = kp.filters.Window(transfer_function, shape=shape, **kwargs)

        dummy_signal.fft_filter(
            transfer_function=w,
            function_domain="frequency",
            shift=shift,
            show_progressbar=True,
        )
        out, _ = capsys.readouterr()
        assert "Completed" in out

        assert isinstance(dummy_signal, kp.signals.EBSD)
        assert dummy_signal.data.dtype == dtype_out
        assert np.allclose(
            np.sum(kp.pattern.fft_spectrum(dummy_signal.inav[0, 0].data)),
            expected_spectrum_sum,
            atol=1e-4,
        )

    def test_fft_filter_spatial(self, dummy_signal):
        dummy_signal.change_dtype(np.float32)
        p = dummy_signal.inav[0, 0].deepcopy().data

        # Sobel operator
        w = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

        dummy_signal.fft_filter(
            transfer_function=w, function_domain="spatial", shift=False
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

        assert isinstance(lazy_signal, kp.signals.LazyEBSD)
        assert lazy_signal.data.dtype == dummy_signal.data.dtype

    def test_inplace(self, dummy_signal):
        filter_kw = dict(
            transfer_function=np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
            function_domain="spatial",
        )

        # Current signal is unaffected
        s2 = dummy_signal.deepcopy()
        s3 = s2.fft_filter(inplace=False, **filter_kw)
        assert np.allclose(dummy_signal.data, s2.data)

        # Custom properties carry over
        assert isinstance(s3, kp.signals.EBSD)
        assert np.allclose(s3.static_background, dummy_signal.static_background)
        assert np.allclose(s3.detector.pc, dummy_signal.detector.pc)
        assert np.allclose(s3.xmap.rotations.data, dummy_signal.xmap.rotations.data)

        # Operating on current signal gives same result as output
        s2.fft_filter(**filter_kw)
        assert np.allclose(s3.data, s2.data)

        # Operating on lazy signal returns lazy signal
        s4 = dummy_signal.as_lazy()
        s5 = s4.fft_filter(inplace=False, **filter_kw)
        assert isinstance(s5, kp.signals.LazyEBSD)
        s5.compute()
        assert np.allclose(s5.data, s2.data)

    def test_lazy_output(self, dummy_signal):
        filter_kw = dict(
            transfer_function=np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
            function_domain="spatial",
        )
        with pytest.raises(
            ValueError, match="`lazy_output=True` requires `inplace=False`"
        ):
            _ = dummy_signal.fft_filter(lazy_output=True, **filter_kw)

        s2 = dummy_signal.fft_filter(inplace=False, lazy_output=True, **filter_kw)
        assert isinstance(s2, kp.signals.LazyEBSD)

        s3 = dummy_signal.as_lazy()
        s4 = s3.fft_filter(inplace=False, lazy_output=False, **filter_kw)
        assert isinstance(s4, kp.signals.EBSD)


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
        self, dummy_signal, num_std, divide_by_square_root, dtype_out, answer, capsys
    ):
        if dtype_out is None:
            dummy_signal.change_dtype(np.int16)

        dummy_signal.normalize_intensity(
            num_std=num_std,
            divide_by_square_root=divide_by_square_root,
            dtype_out=dtype_out,
            show_progressbar=True,
        )
        out, _ = capsys.readouterr()
        assert "Completed" in out

        if dtype_out is None:
            dtype_out = np.int16
        else:
            assert np.allclose(np.mean(dummy_signal.data), 0, atol=1e-6)

        assert isinstance(dummy_signal, kp.signals.EBSD)
        assert dummy_signal.data.dtype == dtype_out
        assert np.allclose(dummy_signal.inav[0, 0].data, answer, atol=1e-4)

    def test_normalize_intensity_lazy(self, dummy_signal):
        dummy_signal.data = dummy_signal.data.astype(np.float32)
        lazy_signal = dummy_signal.as_lazy()

        lazy_signal.normalize_intensity()

        assert isinstance(lazy_signal, kp.signals.LazyEBSD)
        assert np.allclose(np.mean(lazy_signal.data.compute()), 0, atol=1e-6)

    def test_inplace(self, dummy_signal):
        # Current signal is unaffected
        s = dummy_signal.deepcopy()
        s2 = dummy_signal.normalize_intensity(inplace=False)
        assert np.allclose(s.data, dummy_signal.data)

        # Custom properties carry over
        assert isinstance(s2, kp.signals.EBSD)
        assert np.allclose(s2.static_background, dummy_signal.static_background)
        assert np.allclose(s2.detector.pc, dummy_signal.detector.pc)
        assert np.allclose(s2.xmap.rotations.data, dummy_signal.xmap.rotations.data)

        # Operating on current signal gives same result as output
        dummy_signal.normalize_intensity()
        assert np.allclose(s2.data, dummy_signal.data)

        # Operating on lazy signal returns lazy signal
        s3 = s.as_lazy()
        s4 = s3.normalize_intensity(inplace=False)
        assert isinstance(s4, kp.signals.LazyEBSD)
        s4.compute()
        assert np.allclose(s4.data, dummy_signal.data)

    def test_lazy_output(self, dummy_signal):
        with pytest.raises(
            ValueError, match="`lazy_output=True` requires `inplace=False`"
        ):
            _ = dummy_signal.normalize_intensity(lazy_output=True)

        s2 = dummy_signal.normalize_intensity(inplace=False, lazy_output=True)
        assert isinstance(s2, kp.signals.LazyEBSD)

        s3 = dummy_signal.as_lazy()
        s4 = s3.normalize_intensity(inplace=False, lazy_output=False)
        assert isinstance(s4, kp.signals.EBSD)


class TestAverageNeighbourDotProductMap:
    def test_adp_0d(self):
        s = kp.data.nickel_ebsd_small().inav[0, 0]
        with pytest.raises(ValueError, match="Signal must have at least one"):
            _ = s.get_average_neighbour_dot_product_map()

    def test_adp_1d(self):
        s = kp.data.nickel_ebsd_small().inav[0]
        adp = s.get_average_neighbour_dot_product_map()
        assert np.allclose(adp, [0.997470, 0.997457, 0.99744], atol=1e-5)

    def test_adp_2d(self):
        s = kp.data.nickel_ebsd_small()
        adp = s.get_average_neighbour_dot_product_map()
        assert np.allclose(
            adp,
            [
                [0.995679, 0.996117, 0.997220],
                [0.996363, 0.996561, 0.997252],
                [0.995731, 0.996134, 0.997048],
            ],
            atol=1e-5,
        )
        assert adp.dtype == np.float32

    @pytest.mark.parametrize(
        "window, desired_adp_map",
        [
            (
                "rectangular",
                [
                    [0.995135, 0.995891, 0.997144],
                    [0.995425, 0.996032, 0.997245],
                    [0.995160, 0.995959, 0.997019],
                ],
            ),
            (
                "circular",
                [
                    [0.995679, 0.996117, 0.997220],
                    [0.996363, 0.996561, 0.997252],
                    [0.995731, 0.996134, 0.997048],
                ],
            ),
        ],
    )
    def test_adp_window(self, window, desired_adp_map):
        s = kp.data.nickel_ebsd_small()
        w = kp.filters.Window(window=window)
        adp = s.get_average_neighbour_dot_product_map(window=w)
        assert np.allclose(adp, desired_adp_map, atol=1e-5)

    @pytest.mark.parametrize(
        "zero_mean, desired_adp_map",
        [
            (
                True,
                [
                    [0.995679, 0.996117, 0.997220],
                    [0.996363, 0.996561, 0.997252],
                    [0.995731, 0.996134, 0.997048],
                ],
            ),
            (
                False,
                [
                    [0.999663, 0.999699, 0.999785],
                    [0.999717, 0.999733, 0.999786],
                    [0.999666, 0.999698, 0.999769],
                ],
            ),
        ],
    )
    def test_adp_zero_mean(self, zero_mean, desired_adp_map):
        s = kp.data.nickel_ebsd_small()
        adp = s.get_average_neighbour_dot_product_map(zero_mean=zero_mean)
        assert np.allclose(adp, desired_adp_map, atol=1e-5)

    @pytest.mark.parametrize(
        "normalize, desired_adp_map",
        [
            (
                True,
                [
                    [0.995679, 0.996117, 0.997220],
                    [0.996363, 0.996561, 0.997252],
                    [0.995731, 0.996134, 0.997048],
                ],
            ),
            (
                False,
                [
                    [6402544, 6398041.5, 6434939.5],
                    [6411949.5, 6409170, 6464348],
                    [6451061, 6456555.5, 6489456],
                ],
            ),
        ],
    )
    def test_adp_normalize(self, normalize, desired_adp_map):
        s = kp.data.nickel_ebsd_small()
        adp = s.get_average_neighbour_dot_product_map(normalize=normalize)
        assert np.allclose(adp, desired_adp_map, atol=1e-5)

    def test_adp_dtype_out(self):
        s = kp.data.nickel_ebsd_small()
        dtype1 = np.float32
        adp1 = s.get_average_neighbour_dot_product_map(normalize=False)
        assert adp1.dtype == dtype1
        dtype2 = np.int32
        adp2 = s.get_average_neighbour_dot_product_map(normalize=True, dtype_out=dtype2)
        assert adp2.dtype == dtype2

    def test_adp_lazy(self):
        s = kp.data.nickel_ebsd_small(lazy=True)
        adp = s.get_average_neighbour_dot_product_map()

        assert np.allclose(
            adp.compute(),
            [
                [0.995679, 0.996117, 0.997220],
                [0.996363, 0.996561, 0.997252],
                [0.995731, 0.996134, 0.997048],
            ],
            atol=1e-5,
        )
        assert adp.dtype == np.float32

    def test_adp_lazy2(self):
        s = kp.data.nickel_ebsd_large()
        s_lazy = s.as_lazy()
        adp = s.get_average_neighbour_dot_product_map()
        adp_lazy = s_lazy.get_average_neighbour_dot_product_map()

        assert adp.shape == adp_lazy.shape
        assert adp.dtype == adp_lazy.dtype
        assert np.allclose(adp, adp_lazy, equal_nan=True)

    @pytest.mark.parametrize(
        "window",
        [
            kp.filters.Window(window="circular", shape=(3, 3)),
            kp.filters.Window(window="rectangular", shape=(3, 2)),
            kp.filters.Window(window="rectangular", shape=(2, 3)),
        ],
    )
    def test_adp_dp_matrices(self, window, capsys):
        s = kp.data.nickel_ebsd_large()

        dp_matrices = s.get_neighbour_dot_product_matrices(
            window=window, show_progressbar=True
        )
        out, _ = capsys.readouterr()
        assert "Completed" in out

        adp1 = s.get_average_neighbour_dot_product_map(window=window)
        adp2 = s.get_average_neighbour_dot_product_map(
            dp_matrices=dp_matrices, show_progressbar=False
        )
        out, _ = capsys.readouterr()
        assert not out

        assert np.allclose(adp1, adp2)

    @pytest.mark.parametrize("slices", [(0,), (slice(0, 1), slice(None))])
    def test_adp_dp_matrices_shapes(self, slices, capsys):
        s = kp.data.nickel_ebsd_small().inav[slices]
        dp_matrices = s.get_neighbour_dot_product_matrices()

        adp1 = s.get_average_neighbour_dot_product_map(show_progressbar=True)
        out, _ = capsys.readouterr()
        assert "Completed" in out

        adp2 = s.get_average_neighbour_dot_product_map(dp_matrices=dp_matrices)

        assert np.allclose(adp1, adp2)


class TestNeighbourDotProductMatrices:
    def test_dp_matrices_0d(self):
        s = kp.data.nickel_ebsd_small().inav[0, 0]
        with pytest.raises(ValueError, match="Signal must have at least one"):
            _ = s.get_neighbour_dot_product_matrices()

    def test_dp_matrices_1d(self):
        s = kp.data.nickel_ebsd_small().inav[0]
        dp_matrices = s.get_neighbour_dot_product_matrices()

        assert dp_matrices.shape == s.axes_manager.navigation_shape + (3,)
        assert dp_matrices.dtype == np.float32
        assert np.allclose(
            dp_matrices,
            [
                [np.nan, 1, 0.997470],
                [0.997470, 1, 0.997444],
                [0.997444, 1, np.nan],
            ],
            atol=1e-5,
            equal_nan=True,
        )

    def test_dp_matrices_2d(self):
        s = kp.data.nickel_ebsd_small()
        dp_matrices = s.get_neighbour_dot_product_matrices()

        assert dp_matrices.shape == s.axes_manager.navigation_shape + (3, 3)
        assert dp_matrices.dtype == np.float32
        assert np.allclose(
            dp_matrices[1, 1],
            [
                [np.nan, 0.997347, np.nan],
                [0.994177, 1, 0.997358],
                [np.nan, 0.997360, np.nan],
            ],
            atol=1e-5,
            equal_nan=True,
        )

    def test_dp_matrices_lazy(self):
        s = kp.data.nickel_ebsd_large()
        s_lazy = s.as_lazy()
        dp_matrices = s.get_neighbour_dot_product_matrices()
        dp_matrices_lazy = s_lazy.get_neighbour_dot_product_matrices()

        assert dp_matrices.shape == dp_matrices_lazy.shape[:2] + (3, 3)
        assert dp_matrices.dtype == dp_matrices_lazy.dtype
        assert np.allclose(dp_matrices, dp_matrices_lazy, equal_nan=True)

    @pytest.mark.parametrize(
        "window, desired_dp_matrices_11",
        [
            (
                kp.filters.Window(window="circular", shape=(3, 3)),
                [
                    [np.nan, 0.997347, np.nan],
                    [0.994177, 1, 0.997358],
                    [np.nan, 0.997360, np.nan],
                ],
            ),
            (
                kp.filters.Window(window="rectangular", shape=(3, 3)),
                [
                    [0.994048, 0.997347, 0.996990],
                    [0.994177, 1, 0.997358],
                    [0.994017, 0.997360, 0.996960],
                ],
            ),
            (
                kp.filters.Window(window="rectangular", shape=(3, 2)),
                [[0.994048, 0.997347], [0.994177, 1], [0.994017, 0.997360]],
            ),
            (
                kp.filters.Window(window="rectangular", shape=(2, 3)),
                [[0.994048, 0.997347, 0.996990], [0.994177, 1, 0.997358]],
            ),
        ],
    )
    def test_dp_matrices_window(self, window, desired_dp_matrices_11):
        s = kp.data.nickel_ebsd_small()
        dp_matrices = s.get_neighbour_dot_product_matrices(window=window)

        assert np.allclose(
            dp_matrices[1, 1], desired_dp_matrices_11, atol=1e-5, equal_nan=True
        )

    @pytest.mark.parametrize("dtype_out", [np.float16, np.float32, np.float64])
    def test_dp_matrices_dtype_out(self, dtype_out):
        s = kp.data.nickel_ebsd_small()
        dp_matrices = s.get_neighbour_dot_product_matrices(dtype_out=dtype_out)

        assert dp_matrices.dtype == dtype_out

    @pytest.mark.parametrize(
        "zero_mean, desired_dp_matrices11",
        [
            (
                True,
                [
                    [np.nan, 0.997347, np.nan],
                    [0.994177, 1, 0.997358],
                    [np.nan, 0.997360, np.nan],
                ],
            ),
            (
                False,
                [
                    [np.nan, 0.999796, np.nan],
                    [0.999547, 1, 0.999794],
                    [np.nan, 0.999796, np.nan],
                ],
            ),
        ],
    )
    def test_dp_matrices_zero_mean(self, zero_mean, desired_dp_matrices11):
        s = kp.data.nickel_ebsd_small()
        dp_matrices = s.get_neighbour_dot_product_matrices(zero_mean=zero_mean)

        assert np.allclose(
            dp_matrices[1, 1], desired_dp_matrices11, atol=1e-5, equal_nan=True
        )

    @pytest.mark.parametrize(
        "normalize, desired_dp_matrices11",
        [
            (
                True,
                [
                    [np.nan, 0.997347, np.nan],
                    [0.994177, 1, 0.997358],
                    [np.nan, 0.997360, np.nan],
                ],
            ),
            (
                False,
                [
                    [np.nan, 6393165.5, np.nan],
                    [6375199, 6403340, 6439387],
                    [np.nan, 6428928, np.nan],
                ],
            ),
        ],
    )
    def test_dp_matrices_normalize(self, normalize, desired_dp_matrices11):
        s = kp.data.nickel_ebsd_small()
        dp_matrices = s.get_neighbour_dot_product_matrices(normalize=normalize)

        assert np.allclose(
            dp_matrices[1, 1], desired_dp_matrices11, atol=1e-5, equal_nan=True
        )

    def test_dp_matrices_large(self):
        nav_shape = (250, 137)
        s = kp.signals.LazyEBSD(da.ones(nav_shape + (96, 96), dtype=np.uint8))
        dp_matrices = s.get_neighbour_dot_product_matrices()
        assert dp_matrices.shape == nav_shape + (1, 1)


class TestSignal2DMethods:
    """Test methods inherited from Signal2D."""

    def test_as_lazy(self):
        """Lazy attribute and class change while metadata is constant."""
        s = kp.data.nickel_ebsd_small()
        s_lazy = s.as_lazy()
        assert s_lazy._lazy
        assert isinstance(s_lazy, kp.signals.LazyEBSD)
        assert_dictionary(s.metadata.as_dictionary(), s_lazy.metadata.as_dictionary())

    def test_change_dtype(self, dummy_signal):
        """Custom properties carry over and their data type are set
        correctly.
        """
        assert dummy_signal.data.dtype.name == "uint8"
        dummy_signal.change_dtype("float32")
        assert dummy_signal.data.dtype.name == "float32"
        assert dummy_signal.static_background.dtype.name == "float32"

    def test_squeeze(self, dummy_signal):
        """Custom properties carry over."""
        s2 = dummy_signal.squeeze()
        assert np.allclose(dummy_signal.static_background, s2.static_background)

    def test_fft(self, dummy_signal):
        """Test call to ``BaseSignal.fft()`` because it proved
        challenging to overwrite ``BaseSignal.change_dtype()`` in
        ``KikuchipySignal2D``, and the implementation may be unstable.
        """
        s_fft = dummy_signal.fft()
        assert isinstance(s_fft, hs.signals.ComplexSignal2D)
        assert not hasattr(s_fft, "_xmap")

    def test_set_signal_type(self, dummy_signal):
        """Custom properties does not carry over."""
        s_mp = dummy_signal.set_signal_type("EBSDMasterPattern")
        assert not hasattr(s_mp, "_xmap")

    def test_add_gaussian_noise(self, dummy_signal):
        """Custom properties carry over."""
        phase_id = dummy_signal.xmap.phase_id.copy()
        pc = dummy_signal.detector.pc.copy()

        dummy_signal.change_dtype("float32")
        dummy_signal.add_gaussian_noise(std=1)
        assert np.allclose(dummy_signal.xmap.phase_id, phase_id)
        assert np.allclose(dummy_signal.detector.pc, pc)

    def test_add_poissonian_noise(self, dummy_signal):
        """Custom properties carry over."""
        phase_id = dummy_signal.xmap.phase_id.copy()
        pc = dummy_signal.detector.pc.copy()

        dummy_signal.change_dtype("float32")
        dummy_signal.add_poissonian_noise()
        assert np.allclose(dummy_signal.xmap.phase_id, phase_id)
        assert np.allclose(dummy_signal.detector.pc, pc)

    def test_add_ramp(self, dummy_signal):
        """Custom properties carry over."""
        phase_id = dummy_signal.xmap.phase_id.copy()
        pc = dummy_signal.detector.pc.copy()

        dummy_signal.change_dtype("int64")
        dummy_signal.add_ramp(10, 10)
        assert np.allclose(dummy_signal.xmap.phase_id, phase_id)
        assert np.allclose(dummy_signal.detector.pc, pc)

    @pytest.mark.parametrize(
        "axis, start_end, nav_slices, sig_slices, sig_shape, pc_new",
        [
            # Nothing changes
            (
                2,
                (None, None),
                (slice(None), slice(None)),
                (slice(None), slice(None)),
                (3, 3),
                None,
            ),
            # Nothing changes
            (
                0,
                (0, 3),
                (slice(None), slice(0, 3)),
                (slice(None), slice(None)),
                (3, 3),
                None,
            ),
            # Keep first detector column
            (
                2,
                (0, 1),
                (slice(None), slice(None)),
                (slice(None), slice(0, 1)),
                (3, 1),
                [1.385, 0.5, 0.538],
            ),
            # Keep last two detector rows
            (
                3,
                (1, 3),
                (slice(None), slice(None)),
                (slice(1, 3), slice(None)),
                (2, 3),
                [0.462, 0.25, 0.808],
            ),
            # Keep first navigation column
            (
                0,
                (0, 1),
                (slice(None), slice(0, 1)),
                (slice(None), slice(None)),
                (3, 3),
                None,
            ),
            # Keep last two navigation columns
            (
                1,
                (1, 3),
                (slice(1, 3), slice(None)),
                (slice(None), slice(None)),
                (3, 3),
                None,
            ),
        ],
    )
    def test_crop(
        self, dummy_signal, axis, start_end, nav_slices, sig_slices, sig_shape, pc_new
    ):
        """Custom properties are cropped correctly."""
        xmap_old = dummy_signal.xmap
        phase_id = xmap_old.phase_id.reshape(xmap_old.shape)
        pc = dummy_signal.detector.pc.copy()
        static_bg_old = dummy_signal.static_background.copy()

        start, end = start_end
        dummy_signal.crop(axis=axis, start=start, end=end)

        xmap = dummy_signal.xmap
        phase_id_new = xmap.phase_id.reshape(xmap.shape)

        assert np.allclose(phase_id_new, phase_id[nav_slices])
        assert np.allclose(dummy_signal.static_background, static_bg_old[sig_slices])
        assert dummy_signal.detector.shape == sig_shape
        if pc_new is None:
            assert np.allclose(dummy_signal.detector.pc, pc[nav_slices])
        else:
            assert np.allclose(dummy_signal.detector.pc_average, pc_new, atol=1e-3)

    def test_crop_single_pc(self, dummy_signal):
        """Cropping navigation dimension works with single PC."""
        dummy_signal.detector.pc = dummy_signal.detector.pc_average
        dummy_signal.crop(2, start=0, end=1)
        assert np.allclose(
            dummy_signal.detector.pc_average, [1.385, 0.5, 0.538], atol=1e-3
        )

    def test_crop_real_data(self):
        """Cropping works on real data."""
        s = kp.data.nickel_ebsd_small(lazy=True)
        xmap_old = s.xmap.deepcopy()
        det_old = s.detector.deepcopy()
        static_bg_old = s.static_background.copy()

        s.crop(0, start=0, end=2)
        assert np.allclose(s.xmap.rotations.data, xmap_old[:, :2].rotations.data)
        assert np.allclose(s.detector.pc, det_old.pc[:, :2])

        s.crop(3, start=1, end=59)
        assert np.allclose(s.static_background, static_bg_old[1:-1, :])
        assert s.detector.shape == (58, 60)

    @pytest.mark.parametrize(
        "top_bottom_left_right, sig_slices, sig_shape",
        [
            # Nothing changes
            (
                (None, None, None, None),
                (slice(None), slice(None)),
                (3, 3),
            ),
            # Keep first detector column
            (
                (0, 1, None, None),
                (slice(0, 1), slice(None)),
                (1, 3),
            ),
            # Keep bottom right (2, 1) detector pixels
            (
                (1, 3, 2, 3),
                (slice(1, 3), slice(2, 3)),
                (2, 1),
            ),
        ],
    )
    def test_crop_image(
        self, dummy_signal, top_bottom_left_right, sig_slices, sig_shape
    ):
        """Custom properties are cropped correctly."""
        static_bg_old = dummy_signal.static_background.copy()

        dummy_signal.crop_image(*top_bottom_left_right)

        assert np.allclose(dummy_signal.static_background, static_bg_old[sig_slices])
        assert dummy_signal.detector.shape == sig_shape

    def test_inav(self, dummy_signal):
        """Slicing with inav carries over custom attributes."""
        xmap0 = dummy_signal.xmap
        rot0 = xmap0.rotations.reshape(*xmap0.shape)
        pc0 = dummy_signal.detector.pc
        x0 = xmap0.x.reshape(xmap0.shape)
        y0 = xmap0.y.reshape(xmap0.shape)

        # 2D
        s2 = dummy_signal.inav[:2, :3]
        assert np.allclose(s2.xmap.rotations.data, rot0[:3, :2].flatten().data)
        assert np.allclose(s2.detector.pc, pc0[:3, :2])

        # 1D
        s3 = dummy_signal.inav[:, 1]
        assert np.allclose(s3.xmap.rotations.data, rot0[1, :].flatten().data)
        assert np.allclose(s3.detector.pc, pc0[1, :])
        assert np.allclose(s3.xmap.x, x0[1, :].ravel())
        assert s3.xmap.y is None

        # 1D
        s4 = dummy_signal.inav[2, :]
        assert np.allclose(s4.xmap.rotations.data, rot0[:, 2].flatten().data)
        assert np.allclose(s4.detector.pc, pc0[:, 2])
        assert np.allclose(s4.xmap.y, y0[:, 2].ravel())
        assert s4.xmap.x is None

        # 0D
        s5 = dummy_signal.inav[0, 0]
        assert np.allclose(s5.xmap.rotations.data, rot0[0, 0].data)
        assert np.allclose(s5.detector.pc, pc0[0, 0])
        assert np.allclose(s5.xmap.x, x0[0, 0])
        assert s5.xmap.y is None

    def test_isig(self, dummy_signal):
        """Slicing with isig carries over custom attributes."""
        static_bg0 = dummy_signal.static_background
        det0 = dummy_signal.detector

        # 2D
        s2 = dummy_signal.isig[:2, :3]
        assert np.allclose(dummy_signal.static_background, static_bg0)
        assert np.allclose(dummy_signal.detector.pc, det0.pc)
        assert np.allclose(s2.static_background, static_bg0[:3, :2])
        assert s2.detector.shape == (3, 2)
        assert np.allclose(s2.detector.pc_average, [0.692, 0.5, 0.538], atol=1e-3)

        # 1D
        s4 = dummy_signal.isig[:, 1]
        assert isinstance(s4, hs.signals.Signal1D)

        # 1D
        s5 = dummy_signal.isig[2, :]
        assert isinstance(s5, hs.signals.Signal1D)

        # 0D
        s6 = dummy_signal.isig[0, 0]
        assert isinstance(s6, hs.signals.BaseSignal)

    def test_rebin(self):
        """Rebinning carries over custom attributes or not as
        appropriate.
        """
        s = kp.data.nickel_ebsd_small()
        static_bg0 = s.static_background.copy()
        det0 = s.detector.deepcopy()
        xmap0 = s.xmap.deepcopy()

        # Nothing to do
        s2 = s.rebin(scale=(1, 1, 1, 1))
        assert np.allclose(s2.static_background, static_bg0)
        assert np.allclose(s2.detector.pc, det0.pc)
        assert np.allclose(s2.xmap.rotations.data, xmap0.rotations.data)

        # Bin navigation shape
        s3 = s.rebin(scale=(1, 2, 1, 1))
        assert s3.static_background is None
        assert s3.detector.navigation_shape == (1, 3)
        assert np.allclose(s3.detector.pc, 0.5)
        assert s3.xmap is None

        # Bin signal shape
        new_sig_shape = (30, 60)
        s4 = s.rebin(new_shape=(3, 3) + new_sig_shape[::-1], dtype=np.float32)
        assert s4.static_background.shape == new_sig_shape
        assert s4.data.dtype == s4.static_background.dtype == np.float32
        assert s4.detector.shape == new_sig_shape
        assert s4.detector.binning == 1
        assert np.allclose(s4.detector.pc, det0.pc)
        assert np.allclose(s4.xmap.rotations.data, xmap0.rotations.data)

        # Bin signal shape by passing `scale`
        s5 = s4.deepcopy()
        s.rebin(scale=(1, 1, 1, 2), out=s5)
        assert s5.static_background.shape == new_sig_shape
        assert s5.data.dtype == s5.static_background.dtype
        assert s5.detector.shape == new_sig_shape
        assert s5.detector.binning == 1
        assert np.allclose(s5.detector.pc, det0.pc)
        assert np.allclose(s5.xmap.rotations.data, xmap0.rotations.data)

        # Bin both
        s6 = s.rebin(scale=(3, 3, 2, 2))
        assert s6.static_background is None
        assert s6.detector.navigation_shape == (1, 1)
        assert s6.detector.binning == 16
        assert np.allclose(s6.detector.pc, 0.5)
        assert s6.xmap is None

        # Upscale
        s7 = s.rebin(scale=(0.5, 0.5, 0.5, 0.5))
        assert s7.static_background is None
        assert s7.detector.navigation_shape == (6, 6)
        assert s7.detector.binning == 4
        assert np.allclose(s7.detector.pc, 0.5)
        assert s7.xmap is None

    def test_update_custom_properties(self, caplog):
        s = kp.data.nickel_ebsd_small()

        s.detector.pc = s.detector.pc[:2, :1]
        with caplog.at_level(logging.DEBUG, logger="kikuchipy"):
            _ = s.inav[2:, 2:]
        assert "Could not slice EBSD.detector.pc attribute array" in caplog.text

        s._xmap = s.xmap[:1, :2]
        with caplog.at_level(logging.DEBUG, logger="kikuchipy"):
            _ = s.inav[2:, 2:]
        assert "Could not slice EBSD.xmap attribute" in caplog.text


class TestExtractGrid:
    def test_extract_grid_raises(self):
        s = kp.data.nickel_ebsd_small(lazy=True)
        with pytest.raises(ValueError):
            _ = s.extract_grid((4, 3))
        with pytest.raises(ValueError):
            _ = s.extract_grid(4)

    def test_extract_grid(self):
        s = kp.data.nickel_ebsd_large()
        s2 = s.extract_grid((5, 4))

        assert s2.data.shape == (4, 5, 60, 60)
        assert np.allclose(s.static_background, s2.static_background)
        assert s2.detector.navigation_shape == (4, 5)

        nav_shape = s._navigation_shape_rc
        idx, spacing = kp.signals.util.grid_indices(
            (4, 5), nav_shape, return_spacing=True
        )
        idx_tuple = tuple(idx)
        mask = np.zeros(nav_shape, dtype=bool)
        mask[idx_tuple] = True
        mask = mask.ravel()

        assert np.allclose(s.xmap[mask].rotations.data, s2.xmap.rotations.data)
        assert np.allclose(s.detector.pc[idx_tuple], s2.detector.pc)
        assert np.allclose(s.data[idx_tuple], s2.data)

        scales = [a.scale for a in s2.axes_manager.navigation_axes]
        desired_scales = np.array(
            [
                a.scale * sp
                for a, sp in zip(s.axes_manager.navigation_axes, spacing[::-1])
            ]
        )
        assert np.allclose(scales, desired_scales)

    def test_extract_grid_lazy(self):
        s = kp.data.nickel_ebsd_large(lazy=True)
        s2, idx = s.extract_grid((2, 3), return_indices=True)
        assert isinstance(s2, kp.signals.LazyEBSD)
        assert np.allclose(
            idx,
            np.array([[[14, 14], [28, 28], [42, 42]], [[25, 50], [25, 50], [25, 50]]]),
        )

    def test_extract_grid_dummy_data(self):
        s = kp.signals.EBSD(np.zeros((10, 10, 3, 3), dtype="uint8"))
        s.detector.pc = np.zeros((2, 3, 3))
        s2 = s.extract_grid((3, 4))
        assert s2.xmap is None
        assert np.allclose(s2.detector.pc, 0.5)

    def test_extract_grid_1d(self):
        s = kp.data.nickel_ebsd_large(lazy=True)

        s2 = s.inav[0]
        s3, idx3 = s2.extract_grid((3,), return_indices=True)
        assert s3.data.shape == (3, 60, 60)
        assert np.allclose(idx3, [14, 28, 42])

        s4 = s.inav[:, 0]
        s5, idx5 = s4.extract_grid((6,), return_indices=True)
        assert s5.data.shape == (6, 60, 60)
        assert np.allclose(idx5, [10, 21, 32, 43, 54, 65])


class TestDownsample:
    def test_downsample(self):
        s = kp.data.nickel_ebsd_small()
        s2 = s.deepcopy()

        s2.downsample(factor=2)

        # Initial signal unaffected
        assert s.data.shape == (3, 3, 60, 60)
        assert s.detector.shape == (60, 60)
        assert s.static_background.shape == (60, 60)

        # Correct shape and data type
        assert s2.data.shape == (3, 3, 30, 30)
        assert s2.data.dtype == s.data.dtype
        assert s2.detector.shape == (30, 30)
        assert s2.static_background.shape == (30, 30)
        assert s2.static_background.dtype == s.static_background.dtype

    def test_downsample_lazy(self):
        s = kp.data.nickel_ebsd_small(lazy=True)

        s.downsample(factor=3)

        assert isinstance(s, kp.signals.LazyEBSD)
        assert isinstance(s.data, da.Array)
        assert isinstance(s.static_background, np.ndarray)

        s.compute()

        assert isinstance(s, kp.signals.EBSD)
        assert isinstance(s.data, np.ndarray)
        assert isinstance(s.static_background, np.ndarray)

    def test_downsample_dtype(self):
        s = kp.data.nickel_ebsd_small()

        s2 = s.deepcopy()
        s2.downsample(factor=2, dtype_out="uint16")
        assert s2.data.dtype.name == "uint16"

    def test_downsample_raises(self):
        s = kp.data.nickel_ebsd_small(lazy=True)

        with pytest.raises(ValueError, match="Binning `factor` 2.5 must be an integer"):
            s.downsample(2.5)

        with pytest.raises(ValueError, match="Binning `factor` 1 must be an integer >"):
            s.downsample(1)

        with pytest.raises(ValueError, match="Binning `factor` 7 must be a divisor of"):
            s.downsample(7)

    def test_inplace(self):
        s = kp.data.nickel_ebsd_small()

        # Current signal is unaffected
        s2 = s.deepcopy()
        s3 = s2.downsample(2, inplace=False)
        assert np.allclose(s.data, s2.data)

        # Custom properties carry over
        assert isinstance(s3, kp.signals.EBSD)
        assert s3.static_background.shape == (30, 30)
        assert np.allclose(s3.detector.pc, s.detector.pc)
        assert np.allclose(s3.xmap.rotations.data, s.xmap.rotations.data)

        # Operating on current signal gives same result as output
        s2.downsample(2)
        assert np.allclose(s3.data, s2.data)

        # Operating on lazy signal returns lazy signal
        s4 = s.as_lazy()
        s5 = s4.downsample(2, inplace=False)
        assert isinstance(s5, kp.signals.LazyEBSD)
        s5.compute()
        assert np.allclose(s5.data, s2.data)

    def test_lazy_output(self):
        s = kp.data.nickel_ebsd_small()
        with pytest.raises(
            ValueError, match="`lazy_output=True` requires `inplace=False`"
        ):
            _ = s.downsample(2, lazy_output=True)

        s2 = s.downsample(2, inplace=False, lazy_output=True)
        assert isinstance(s2, kp.signals.LazyEBSD)

        s3 = s.as_lazy()
        s4 = s3.downsample(2, inplace=False, lazy_output=False)
        assert isinstance(s4, kp.signals.EBSD)
