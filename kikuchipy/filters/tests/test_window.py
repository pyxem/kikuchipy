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

import os

from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.colorbar import Colorbar
from matplotlib.pyplot import imread
import numpy as np
import pytest
from scipy.signal.windows import gaussian, general_gaussian

from kikuchipy.filters.window import (
    highpass_fft_filter,
    lowpass_fft_filter,
    modified_hann,
    distance_to_origin,
    Window,
)

# Window data used to check results in tests
CIRCULAR33 = np.array([0, 1, 0, 1, 1, 1, 0, 1, 0]).reshape(3, 3)
# fmt: off
CIRCULAR54 = np.array(
    [0., 0., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 0., 0.,
     1., 0.]
).reshape(5, 4)
# fmt: on
RECTANGULAR33 = np.ones(9).reshape(3, 3)
RECTANGULAR3 = np.ones(3)
GAUSS33_STD1 = np.outer(gaussian(3, 1), gaussian(3, 1))
GAUSS55_STD2 = np.outer(gaussian(5, 2), gaussian(5, 2))
GAUSS33_CIRCULAR = np.array(
    [0.0, 0.60653066, 0.0, 0.60653066, 1.0, 0.60653066, 0.0, 0.60653066, 0.0]
).reshape(3, 3)
GAUSS5_STD2 = gaussian(5, 2)
GENERAL_GAUSS55_PWR05_STD2 = np.outer(
    general_gaussian(5, 0.5, 2), general_gaussian(5, 0.5, 2)
)
CUSTOM = np.arange(25).reshape(5, 5)


class TestWindow:
    @pytest.mark.parametrize(
        (
            "window, window_type, shape, kwargs, answer_shape, "
            "answer_coeff, answer_circular"
        ),
        [
            ("circular", "circular", (3, 3), None, (3, 3), CIRCULAR33, True),
            (CUSTOM, "custom", (10, 20), None, CUSTOM.shape, CUSTOM, False),
            ("gaussian", "gaussian", (5, 5), 2, (5, 5), GAUSS55_STD2, False),
        ],
    )
    def test_init(
        self,
        window,
        window_type,
        shape,
        kwargs,
        answer_shape,
        answer_coeff,
        answer_circular,
    ):
        if kwargs is None:
            w = Window(window=window, shape=shape)
        else:
            w = Window(window=window, shape=shape, kwargs=kwargs)

        assert w.is_valid()
        assert w.name == window_type
        assert w.shape == answer_shape
        assert w.circular is answer_circular
        np.testing.assert_array_almost_equal(w.data, answer_coeff)

    @pytest.mark.parametrize(
        "window, shape, error_type, match",
        [
            (
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                (5, 5),
                ValueError,
                "Window <class 'list'> must be of type numpy.ndarray,",
            ),
            ("boxcar", (5, -5), ValueError, "All window axes .* must be > 0",),
            (
                "boxcar",
                (5, 5.1),
                TypeError,
                "Window shape .* must be a sequence of ints.",
            ),
        ],
    )
    def test_init_raises_errors(self, window, shape, error_type, match):
        with pytest.raises(error_type, match=match):
            _ = Window(window=window, shape=shape)

    @pytest.mark.parametrize("Nx", [3, 5, 7, 8])
    def test_init_passing_nx(self, Nx):
        w = Window(Nx=Nx)
        assert w.shape == (Nx,)

    def test_init_from_array(self):
        a = np.arange(5)
        w = Window(a)

        assert isinstance(w, Window)
        assert w.name == "custom"
        assert w.circular is False
        assert np.sum(a) == np.sum(w)

        w2 = w[1:]
        assert isinstance(w2, Window)
        assert w2.name == "custom"
        assert np.sum(a[1:]) == np.sum(w2)

    def test_init_cast_with_view(self):
        a = np.arange(5)
        w = a.view(Window)
        assert isinstance(w, Window)

    def test_array_finalize_returns_none(self):
        w = Window()
        assert w.__array_finalize__(None) is None

    def test_init_general_gaussian(self):
        window = "general_gaussian"
        shape = (5, 5)
        w = Window(window=window, shape=shape, p=0.5, std=2,)
        assert w.is_valid()
        np.testing.assert_array_almost_equal(w.data, GENERAL_GAUSS55_PWR05_STD2)
        assert w.name == window
        assert w.shape == shape

    def test_representation(self):
        w = Window()
        object_type = str(type(w)).strip(">'").split(".")[-1]
        assert w.__repr__() == (
            f"{object_type} {w.shape} {w.name}\n"
            "[[0. 1. 0.]\n [1. 1. 1.]\n [0. 1. 0.]]"
        )

    def test_is_valid(self):
        change_attribute = np.array([0, 0, 0, 1])

        # Change one attribute at a time and check whether the window is valid
        for i in range(len(change_attribute)):
            w = Window()

            valid_window = True
            if sum(change_attribute[:3]) == 1:
                valid_window = False

            if change_attribute[0]:  # Set type from str to int
                w.name = 1
            elif change_attribute[1]:  # Add a third axis
                w = np.expand_dims(w, 1)
            elif change_attribute[2]:  # Change circular boolean value to str
                w.circular = "True"

            # Roll axis to change which attribute to change next time
            change_attribute = np.roll(change_attribute, 1)

            assert w.is_valid() == valid_window

    @pytest.mark.parametrize(
        "window, shape, answer_coeff, answer_circular, answer_type",
        [
            # Changes type as well
            ("rectangular", (3, 3), CIRCULAR33, True, "circular"),
            ("boxcar", (3, 3), CIRCULAR33, True, "circular"),
            # Does nothing since window has only one axis
            ("rectangular", (3,), RECTANGULAR3, False, "rectangular"),
            # Behaves as expected
            ("gaussian", (3, 3), GAUSS33_CIRCULAR, True, "gaussian"),
            # Even axis
            ("rectangular", (5, 4), CIRCULAR54, True, "circular"),
        ],
    )
    def test_make_circular(
        self, window, shape, answer_coeff, answer_circular, answer_type
    ):
        k = Window(window=window, shape=shape)
        k.make_circular()

        np.testing.assert_array_almost_equal(k, answer_coeff)
        assert k.name == answer_type
        assert k.circular is answer_circular

    @pytest.mark.parametrize(
        "shape, compatible",
        [
            ((3,), True),
            ((3, 3), True),
            ((3, 4), False),
            ((4, 3), False),
            ((4, 4), False),
        ],
    )
    def test_shape_compatible(self, dummy_signal, shape, compatible):
        w = Window(shape=shape)
        assert (
            w.shape_compatible(dummy_signal.axes_manager.navigation_shape)
            == compatible
        )

    def test_plot_default_values(self):
        w = Window()
        fig, im, cbar = w.plot()

        np.testing.assert_array_almost_equal(w, im.get_array().data)
        assert im.cmap.name == "viridis"
        assert isinstance(fig, Figure)
        assert isinstance(im, AxesImage)
        assert isinstance(cbar, Colorbar)

    def test_plot_invalid_window(self):
        w = Window()
        w.name = 1
        assert w.is_valid() is False
        with pytest.raises(ValueError, match="Window is invalid."):
            w.plot()

    @pytest.mark.parametrize(
        "window, answer_coeff, cmap, textcolors, cmap_label",
        [
            ("circular", CIRCULAR33, "viridis", ["k", "w"], "Coefficient",),
            ("rectangular", RECTANGULAR33, "inferno", ["b", "r"], "Coeff.",),
        ],
    )
    def test_plot(
        self, window, answer_coeff, cmap, textcolors, cmap_label, tmp_path
    ):
        w = Window(window=window)

        fig, im, cbar = w.plot(
            cmap=cmap, textcolors=textcolors, cmap_label=cmap_label
        )

        np.testing.assert_array_almost_equal(w, answer_coeff)
        np.testing.assert_array_almost_equal(im.get_array().data, answer_coeff)
        assert isinstance(fig, Figure)
        assert isinstance(im, AxesImage)
        assert isinstance(cbar, Colorbar)

        # Check that the figure can be written to and read from file
        os.chdir(tmp_path)
        fname = "tests.png"
        fig.savefig(fname)
        _ = imread(fname)

    def test_plot_one_axis(self):
        w = Window(window="gaussian", shape=(5,), std=2)
        fig, im, cbar = w.plot()

        # Compare to global window GAUSS5_STD2
        np.testing.assert_array_almost_equal(w, GAUSS5_STD2)
        np.testing.assert_array_almost_equal(
            im.get_array().data[:, 0], GAUSS5_STD2
        )

    @pytest.mark.parametrize(
        "shape, c, w_c, answer",
        [
            (
                (5, 5),
                1,
                1,
                # fmt: off
                np.array([
                    [0.0012, 0.0470, 0.1353, 0.0470, 0.0012],
                    [0.0470, 0.7095, 1., 0.7095, 0.0470],
                    [0.1353, 1., 1., 1., 0.1353],
                    [0.0470, 0.7095, 1., 0.7095, 0.0470],
                    [0.0012, 0.0470, 0.1353, 0.0470, 0.0012],
                ])
                # fmt: on
            ),
            (
                (6, 5),
                2,
                1,
                # fmt: off
                    np.array([
                        [0.0057, 0.0670, 0.1353, 0.0670, 0.0057],
                        [0.2534, 0.8945, 1., 0.8945, 0.2534],
                        [0.8945, 1., 1., 1., 0.8945],
                        [1., 1., 1., 1., 1.],
                        [0.8945, 1., 1., 1., 0.8945],
                        [0.2534, 0.8945, 1., 0.8945, 0.2534],
                    ])
                # fmt: on
            ),
        ],
    )
    def test_lowpass_fft_filter_direct(self, shape, c, w_c, answer):
        w = lowpass_fft_filter(shape=shape, cutoff=c, cutoff_width=w_c)

        assert w.shape == answer.shape
        assert np.allclose(w, answer, atol=1e-4)

    def test_lowpass_fft_filter_equal(self):
        shape = (96, 96)
        c = 30
        w_c = c // 2
        w1 = Window("lowpass", cutoff=c, cutoff_width=w_c, shape=shape)
        w2 = lowpass_fft_filter(shape=shape, cutoff=c)

        assert np.allclose(w1, w2)

    @pytest.mark.parametrize(
        "shape, c, w_c, answer",
        [
            (
                (5, 5),
                2,
                2,
                # fmt: off
                np.array([
                    [1, 1, 1, 1, 1],
                    [1, 0.8423, 0.6065, 0.8423, 1],
                    [1, 0.6065, 0.1353, 0.6065, 1],
                    [1, 0.8423, 0.6065, 0.8423, 1],
                    [1, 1, 1, 1, 1],
                ])
                # fmt: on
            ),
            (
                (6, 5),
                2,
                1,
                # fmt: off
                    np.array([
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 0.5034, 0.1353, 0.5034, 1],
                        [1, 0.1353, 0.0003, 0.1353, 1],
                        [1, 0.5034, 0.1353, 0.5034, 1],
                        [1, 1, 1, 1, 1],
                    ])
                # fmt: on
            ),
        ],
    )
    def test_highpass_fft_filter_direct(self, shape, c, w_c, answer):
        w = highpass_fft_filter(shape=shape, cutoff=c, cutoff_width=w_c)

        assert w.shape == answer.shape
        assert np.allclose(w, answer, atol=1e-4)

    def test_highpass_fft_filter_equal(self):
        shape = (96, 96)
        c = 30
        w_c = c // 2
        w1 = Window("highpass", cutoff=c, cutoff_width=w_c, shape=shape)
        w2 = highpass_fft_filter(shape=shape, cutoff=c)

        assert np.allclose(w1, w2)

    @pytest.mark.parametrize(
        "Nx, answer",
        [
            (3, np.array([0.5, 1, 0.5])),
            # fmt: off
            (11, np.array(
                [
                    0.1423, 0.4154, 0.6548, 0.8412, 0.9594, 1., 0.9594, 0.8412,
                    0.6548, 0.4154, 0.1423
                ])
             ),
            # fmt: on
        ],
    )
    def test_modified_hann_direct(self, Nx, answer):
        w = modified_hann.py_func(Nx)

        assert np.allclose(w, answer, atol=1e-4)

    @pytest.mark.parametrize(
        "Nx, answer", [(96, 61.1182), (801, 509.9328)],
    )
    def test_modified_hann_direct_sum(self, Nx, answer):
        # py_func ensures coverage for a Numba decorated function
        w = modified_hann.py_func(Nx)

        assert np.allclose(np.sum(w), answer, atol=1e-4)

    def test_modified_hann_equal(self):
        w1 = Window("modified_hann", shape=(30,))
        w2 = modified_hann(Nx=30)

        assert np.allclose(w1, w2)

    @pytest.mark.parametrize(
        "shape, origin, answer",
        [
            (
                (5, 5),
                None,
                np.array(
                    [
                        [2.8284, 2.2360, 2, 2.2360, 2.8284],
                        [2.2360, 1.4142, 1, 1.4142, 2.2360],
                        [2, 1, 0, 1, 2],
                        [2.2360, 1.4142, 1, 1.4142, 2.2360],
                        [2.8284, 2.2360, 2, 2.2360, 2.8284],
                    ]
                ),
            ),
            ((5,), (2,), np.array([2, 1, 0, 1, 2]),),
            (
                (4, 4),
                (2, 3),
                np.array(
                    [
                        [3.6055, 2.8284, 2.2360, 2],
                        [3.1622, 2.2360, 1.4142, 1],
                        [3, 2, 1, 0],
                        [3.1622, 2.2360, 1.4142, 1],
                    ]
                ),
            ),
        ],
    )
    def test_distance_to_origin(self, shape, origin, answer):
        r = distance_to_origin(shape=shape, origin=origin)
        assert np.allclose(r, answer, atol=1e-4)

    @pytest.mark.parametrize(
        "std, shape, answer",
        [
            (0.001, (1,), np.array([[1]])),
            (-0.5, (1,), Window("gaussian", std=0.5, shape=(1,))),
            (
                0.5,
                (3, 3),
                Window(
                    np.array(
                        [
                            [0.01134374, 0.08381951, 0.01134374],
                            [0.08381951, 0.61934703, 0.08381951],
                            [0.01134374, 0.08381951, 0.01134374],
                        ]
                    )
                ),
            ),
        ],
    )
    def test_gaussian(self, std, shape, answer):
        w = Window("gaussian", std=std, shape=shape)
        w = w / (2 * np.pi * std ** 2)
        w = w / np.sum(w)

        assert np.allclose(w, answer)

    @pytest.mark.parametrize(
        "shape, desired_n_neighbours",
        [
            ((3, 3), (1, 1)),
            ((3,), (1,)),
            ((7, 5), (3, 2),),
            ((6, 5), (2, 2)),
            ((5, 7), (2, 3)),
        ],
    )
    def test_n_neighbours(self, shape, desired_n_neighbours):
        assert Window(shape=shape).n_neighbours == desired_n_neighbours
