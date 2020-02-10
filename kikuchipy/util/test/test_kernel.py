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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.signal.windows import gaussian, general_gaussian

import kikuchipy as kp

# Kernel coefficients used to check results in tests
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


class TestKernel:
    @pytest.mark.parametrize(
        (
            "kernel, kernel_type, kernel_size, kwargs, answer_size, "
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
        kernel,
        kernel_type,
        kernel_size,
        kwargs,
        answer_size,
        answer_coeff,
        answer_circular,
    ):
        if kwargs is None:
            k = kp.util.Kernel(kernel=kernel, kernel_size=kernel_size)
        else:
            k = kp.util.Kernel(
                kernel=kernel, kernel_size=kernel_size, kwargs=kwargs
            )

        assert k.is_valid()
        assert k.type == kernel_type
        assert k.coefficients.shape == answer_size
        assert k.circular is answer_circular
        np.testing.assert_array_almost_equal(k.coefficients, answer_coeff)

    def test_init_general_gaussian(self):
        kernel_type = "general_gaussian"
        kernel_size = (5, 5)
        k = kp.util.Kernel(
            kernel=kernel_type, kernel_size=kernel_size, p=0.5, std=2,
        )
        assert k.is_valid()
        np.testing.assert_array_almost_equal(
            k.coefficients, GENERAL_GAUSS55_PWR05_STD2
        )
        assert k.type == kernel_type
        assert k.coefficients.shape == kernel_size

    @pytest.mark.parametrize(
        "kernel, kernel_size, error_type, match",
        [
            (
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                (5, 5),
                ValueError,
                "Kernel <class 'list'> must be of type numpy.ndarray or dask.",
            ),
            ("boxcar", (5, -5), ValueError, "All kernel axes .* must be > 0",),
            (
                "boxcar",
                (5, 5.1),
                TypeError,
                "Kernel size .* must be a sequence of ints.",
            ),
        ],
    )
    def test_init_raises_errors(self, kernel, kernel_size, error_type, match):
        with pytest.raises(error_type, match=match):
            kp.util.Kernel(kernel=kernel, kernel_size=kernel_size)

    def test_representation(self):
        k = kp.util.Kernel()
        assert k.__repr__() == f"{k.type}, {k.coefficients.shape}"

    def test_is_valid(self):
        change_attribute = np.array([0, 0, 0, 0, 1])

        # Change one attribute at a time and check whether the kernel is valid
        for i in range(5):  # Four attributes to change + one with valid kernel
            k = kp.util.Kernel()

            valid_kernel = True
            if sum(change_attribute[:4]) == 1:
                valid_kernel = False

            if change_attribute[0]:  # Set type from str to int
                k.type = 1
            elif change_attribute[1]:  # Change coefficients object to int
                k.coefficients = 1
            elif change_attribute[2]:  # Add a third axis
                k.coefficients = np.expand_dims(k.coefficients, 1)
            elif change_attribute[3]:  # Change circular boolean value to str
                k.circular = "True"

            # Roll axis to change which attribute to change next time
            change_attribute = np.roll(change_attribute, 1)

            assert k.is_valid() == valid_kernel

    @pytest.mark.parametrize(
        "kernel, kernel_size, answer_coeff, answer_circular, answer_type",
        [
            # Changes type as well
            ("rectangular", (3, 3), CIRCULAR33, True, "circular"),
            ("boxcar", (3, 3), CIRCULAR33, True, "circular"),
            # Does nothing since kernel has only one axis
            ("rectangular", (3,), RECTANGULAR3, False, "rectangular"),
            # Behaves as expected
            ("gaussian", (3, 3), GAUSS33_CIRCULAR, True, "gaussian"),
            # Even axis
            ("rectangular", (5, 4), CIRCULAR54, True, "circular"),
        ],
    )
    def test_make_circular(
        self, kernel, kernel_size, answer_coeff, answer_circular, answer_type
    ):
        k = kp.util.Kernel(kernel=kernel, kernel_size=kernel_size)
        k.make_circular()

        np.testing.assert_array_almost_equal(k.coefficients, answer_coeff)
        assert k.type == answer_type
        assert k.circular is answer_circular

    @pytest.mark.parametrize(
        "kernel_size, raises_error",
        [((3,), False), ((3, 3), False), ((3, 4), True),],
    )
    def test_scan_compatible(self, dummy_signal, kernel_size, raises_error):
        k = kp.util.Kernel(kernel_size=kernel_size)
        if raises_error:
            with pytest.raises(ValueError, match="Kernel of size .* is "):
                k.scan_compatible(dummy_signal)

    def test_scan_compatible_raises(self):
        k = kp.util.Kernel(kernel_size=(3, 4))
        with pytest.raises(AttributeError, match="Scan <class 'int'> must be"):
            k.scan_compatible(1)

    def test_scan_compatible_1d_signal_2d_kernel(self, dummy_signal):
        k = kp.util.Kernel(kernel_size=(3, 3))
        with pytest.raises(ValueError):
            k.scan_compatible(dummy_signal.inav[:, 0])

    @pytest.mark.parametrize(
        "kernel_size, n_dims, new_shape",
        [
            ((5,), 2, (5, 1, 1)),
            ((3, 3), 4, (3, 3, 1, 1, 1, 1)),
            ((5, 10), 1, (5, 10, 1)),
        ],
    )
    def test_add_axes(self, kernel_size, n_dims, new_shape):
        k = kp.util.Kernel(kernel_size=kernel_size)
        k._add_axes(n_dims)
        assert k.is_valid() is False
        assert k.coefficients.shape == new_shape

    def test_plot_default_values(self):
        k = kp.util.Kernel()
        fig, im, cbar = k.plot()

        np.testing.assert_array_almost_equal(
            k.coefficients, im.get_array().data
        )
        assert im.cmap.name == "viridis"
        assert isinstance(fig, mpl.figure.Figure)
        assert isinstance(im, mpl.image.AxesImage)
        assert isinstance(cbar, mpl.colorbar.Colorbar)

    def test_plot_invalid_kernel(self):
        k = kp.util.Kernel()
        k.type = 1
        assert k.is_valid() is False
        with pytest.raises(ValueError, match="Kernel is invalid."):
            k.plot()

    @pytest.mark.parametrize(
        "kernel, answer_coeff, cmap, textcolors, cmap_label",
        [
            ("circular", CIRCULAR33, "viridis", ["k", "w"], "Coefficient",),
            ("rectangular", RECTANGULAR33, "inferno", ["b", "r"], "Coeff.",),
        ],
    )
    def test_plot(
        self, kernel, answer_coeff, cmap, textcolors, cmap_label, tmp_path
    ):
        k = kp.util.Kernel(kernel=kernel)

        fig, im, cbar = k.plot(
            cmap=cmap, textcolors=textcolors, cmap_label=cmap_label
        )

        np.testing.assert_array_almost_equal(k.coefficients, answer_coeff)
        np.testing.assert_array_almost_equal(im.get_array().data, answer_coeff)
        assert isinstance(fig, mpl.figure.Figure)
        assert isinstance(im, mpl.image.AxesImage)
        assert isinstance(cbar, mpl.colorbar.Colorbar)

        # Check that the figure can be written to and read from file
        os.chdir(tmp_path)
        fname = "test.png"
        fig.savefig(fname)
        fig_read = plt.imread(fname)

    def test_plot_one_axis(self):
        k = kp.util.Kernel(kernel="gaussian", kernel_size=(5,), std=2)
        fig, im, cbar = k.plot()

        # Compare to global kernel GAUSS5_STD2
        np.testing.assert_array_almost_equal(k.coefficients, GAUSS5_STD2)
        np.testing.assert_array_almost_equal(
            im.get_array().data[:, 0], GAUSS5_STD2
        )
