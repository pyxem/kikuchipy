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

import matplotlib as mpl
import numpy as np
import pytest

import kikuchipy as kp


class TestKernel:
    @pytest.mark.parametrize(
        "kernel, kernel_size, answer, match, error_type, kwargs",
        [
            # Standard circular kernel
            (
                "circular",
                (3, 3),
                # fmt: off
                np.array(
                    [
                        [0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0],
                    ],
                ),
                # fmt: on
                None,
                None,
                None,
            ),
            # Circular kernel with first dimension even
            (
                "circular",
                (2, 3),
                # fmt: off
                np.array(
                    [
                        [0, 1, 0],
                        [1, 1, 1],
                    ],
                ),
                # fmt: on
                None,
                None,
                None,
            ),
            # Circular kernel with second dimension even
            (
                "circular",
                (3, 2),
                # fmt: off
                np.array(
                        [
                            [0, 1],
                            [1, 1],
                            [0, 1],
                        ],
                    ),
                # fmt: on
                None,
                None,
                None,
            ),
            # Rectangular kernel
            ("rectangular", (2, 2), np.ones((2, 2)), None, None, None),
            # One keyword argument to scipy.signal.windows.get_window
            (
                "gaussian",
                (3, 3),
                # fmt: off
                    np.array(
                        [
                            [0.77880078, 0.8824969, 0.77880078],
                            [0.8824969, 1., 0.8824969],
                            [0.77880078, 0.8824969, 0.77880078]
                        ]
                    ),
                # fmt: on
                None,
                None,
                {"std": 2},
            ),
            # Two keyword arguments to scipy.signal.windows.get_window
            (
                "general_gaussian",
                (3, 2),
                # fmt: off
                    np.array(
                        [
                            [0.96734205, 0.96734205],
                            [0.99804878, 0.99804878],
                            [0.96734205, 0.96734205]
                        ],
                    ),
                # fmt: on
                None,
                None,
                {"sig": 2, "p": 2},
            ),
            # Integer kernel size
            ("rectangular", 3, np.ones(3), None, None, None),
            # Custom kernel
            (
                np.arange(9).reshape((3, 3)),
                (4, 2),  # Kernel size shouldn't matter with custom kernel
                np.arange(9).reshape((3, 3)),
                None,
                None,
                None,
            ),
            # Invalid custom kernel
            (
                [0, 1, 2, 3, 4, 5, 6, 7, 8],
                (9,),
                np.arange(9),
                "Kernel must be of type numpy.ndarray, however a kernel of ",
                ValueError,
                None,
            ),
            # Negative number as invalid kernel dimensions
            (
                "circular",
                (3, -3),
                # fmt: off
                    np.array(
                        [
                            [0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0],
                        ],
                    ),
                # fmt: on
                "Kernel dimensions must be positive, however .* was passed.",
                ValueError,
                None,
            ),
            # String as invalid kernel dimensions
            (
                "circular",
                "(3, -3)",
                # fmt: off
                    np.array(
                        [
                            [0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0],
                        ],
                    ),
                # fmt: on
                "Kernel dimensions must be an int or a tuple of ints, however ",
                TypeError,
                None,
            ),
            # Greater kernel dimension than scan dimension as invalid kernel
            # dimensions
            (
                "rectangular",
                (3, 4),
                np.ones(12).reshape((3, 4)),
                "Kernel size .* is too large for a scan of dimensions .*",
                ValueError,
                None,
            ),
        ],
    )
    def test_get_pattern_kernel(
        self,
        dummy_signal,
        kernel,
        kernel_size,
        answer,
        match,
        error_type,
        kwargs,
    ):
        if match is None:
            if kwargs is None:
                kernel = kp.util.kernel.get_kernel(
                    kernel=kernel,
                    kernel_size=kernel_size,
                    axes=dummy_signal.axes_manager,
                )
            else:
                kernel = kp.util.kernel.get_kernel(
                    kernel=kernel,
                    kernel_size=kernel_size,
                    axes=dummy_signal.axes_manager,
                    **kwargs,
                )
            np.testing.assert_array_almost_equal(kernel, answer)
        else:
            with pytest.raises(error_type, match=match):
                kp.util.kernel.get_kernel(
                    kernel=kernel,
                    kernel_size=kernel_size,
                    axes=dummy_signal.axes_manager,
                )

    def test_get_pattern_kernel_warns_kernel_dimensions(self, dummy_signal):
        with pytest.warns(
            UserWarning,
            match="Creates kernel of size .*, since input kernel size .* has",
        ):
            kp.util.kernel.get_kernel(
                kernel_size=(3, 3, 3), axes=dummy_signal.axes_manager
            )

    def test_get_pattern_kernel_invalid_axes_manager(self):
        with pytest.raises(AttributeError, match="A hyperspy.axes.AxesManager"):
            kp.util.kernel.get_kernel("circular", (3, 3), axes=1)

    def test_plot_kernel(self):
        kernel = kp.util.kernel.get_kernel()
        fig, im, cbar = kp.util.kernel.plot_kernel(kernel)

        assert isinstance(fig, mpl.figure.Figure)
        assert isinstance(im, mpl.image.AxesImage)
        assert isinstance(cbar, mpl.colorbar.Colorbar)
        np.testing.assert_array_equal(kernel, im.get_array().data)

    def test_plot_kernel_invalid_dimensions(self):
        kernel = np.arange(3 ** 3).reshape((3, 3, 3))
        with pytest.raises(ValueError, match="Can only plot a kernel of max. "):
            kp.util.kernel.plot_kernel(kernel)

    @pytest.mark.parametrize("cmap", ["inferno", "viridis"])
    def test_plot_kernel_cmap(self, cmap):
        kernel = kp.util.kernel.get_kernel(kernel_size=(4, 3))
        _, im, _ = kp.util.kernel.plot_kernel(kernel, cmap=cmap)

        assert im.cmap.name == cmap

    @pytest.mark.parametrize("textcolors", [["red", "blue"], ["r", "b"]])
    def test_plot_kernel_textcolors(self, textcolors):
        kernel = kp.util.kernel.get_kernel(kernel_size=(2, 4))
        _, _, _ = kp.util.kernel.plot_kernel(kernel, textcolors=textcolors)
