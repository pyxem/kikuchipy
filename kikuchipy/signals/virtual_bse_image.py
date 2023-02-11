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

from __future__ import annotations
from typing import List, Optional, Tuple, Union

import numpy as np

from kikuchipy.signals._kikuchipy_signal import KikuchipySignal2D, LazyKikuchipySignal2D


class VirtualBSEImage(KikuchipySignal2D):
    """Virtual backscatter electron (BSE) image(s).

    This class extends HyperSpy's Signal2D class for virtual BSE images.

    See the docstring of :class:`~hyperspy._signals.signal2d.Signal2D`
    for a list of attributes and methods.
    """

    _signal_type = "VirtualBSEImage"
    _alias_signal_types = ["virtual_backscatter_electron_image"]
    _lazy = False

    # -- Inherited methods included here for documentation purposes -- #

    def rescale_intensity(
        self,
        relative: bool = False,
        in_range: Union[Tuple[int, int], Tuple[float, float], None] = None,
        out_range: Union[Tuple[int, int], Tuple[float, float], None] = None,
        dtype_out: Union[
            str, np.dtype, type, Tuple[int, int], Tuple[float, float], None
        ] = None,
        percentiles: Union[Tuple[int, int], Tuple[float, float], None] = None,
        show_progressbar: Optional[bool] = None,
        inplace: bool = True,
        lazy_output: Optional[bool] = None,
    ) -> Union[None, VirtualBSEImage, LazyVirtualBSEImage]:
        return super().rescale_intensity(
            relative,
            in_range,
            out_range,
            dtype_out,
            percentiles,
            show_progressbar,
            inplace,
            lazy_output,
        )

    def normalize_intensity(
        self,
        num_std: int = 1,
        divide_by_square_root: bool = False,
        dtype_out: Union[str, np.dtype, type, None] = None,
        show_progressbar: Optional[bool] = None,
        inplace: bool = True,
        lazy_output: Optional[bool] = None,
    ) -> Union[None, VirtualBSEImage, LazyVirtualBSEImage]:
        return super().normalize_intensity(
            num_std,
            divide_by_square_root,
            show_progressbar,
            dtype_out,
            inplace,
            lazy_output,
        )

    def adaptive_histogram_equalization(
        self,
        kernel_size: Optional[Union[Tuple[int, int], List[int]]] = None,
        clip_limit: Union[int, float] = 0,
        nbins: int = 128,
        show_progressbar: Optional[bool] = None,
        inplace: bool = True,
        lazy_output: Optional[bool] = None,
    ) -> Union[None, VirtualBSEImage, LazyVirtualBSEImage]:
        return super().adaptive_histogram_equalization(
            kernel_size,
            clip_limit,
            nbins,
            show_progressbar,
            inplace,
            lazy_output,
        )


class LazyVirtualBSEImage(LazyKikuchipySignal2D, VirtualBSEImage):
    """Lazy implementation of
    :class:`~kikuchipy.signals.VirtualBSEImage`.

    See the documentation of ``VirtualBSEImage`` for attributes and
    methods.

    This class extends HyperSpy's
    :class:`~hyperspy._signals.signal2d.LazySignal2D` class for EBSD
    master patterns. See the documentation of that class for how to
    create this signal and the list of inherited attributes and methods.
    """

    def compute(self, *args, **kwargs) -> None:
        super().compute(*args, **kwargs)
