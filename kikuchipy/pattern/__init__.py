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

"""Single and chunk pattern processing used by signals."""

from kikuchipy.pattern._pattern import (
    fft,
    fft_filter,
    fft_frequency_vectors,
    fft_spectrum,
    get_dynamic_background,
    get_image_quality,
    ifft,
    normalize_intensity,
    remove_dynamic_background,
    rescale_intensity,
)
from kikuchipy.pattern import chunk
from kikuchipy.pattern import correlate

__all__ = [
    "chunk",
    "correlate",
    "fft",
    "fft_filter",
    "fft_frequency_vectors",
    "fft_spectrum",
    "get_dynamic_background",
    "get_image_quality",
    "ifft",
    "normalize_intensity",
    "remove_dynamic_background",
    "rescale_intensity",
]
