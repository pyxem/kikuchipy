# Copyright 2019-2022 The kikuchipy developers
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

from typing import Optional, Tuple, Union

import numpy as np

from kikuchipy.signals._kikuchipy_signal import KikuchipySignal2D


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
    ) -> None:
        super().rescale_intensity(
            relative, in_range, out_range, dtype_out, percentiles, show_progressbar
        )

    def normalize_intensity(
        self,
        num_std: int = 1,
        divide_by_square_root: bool = False,
        dtype_out: Union[str, np.dtype, type, None] = None,
        show_progressbar: Optional[bool] = None,
    ) -> None:
        super().normalize_intensity(num_std, divide_by_square_root, dtype_out)
