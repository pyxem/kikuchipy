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

"""Pattern filters used on signals."""

from kikuchipy.filters.window import (
    distance_to_origin,
    modified_hann,
    lowpass_fft_filter,
    highpass_fft_filter,
    Window,
)

__all__ = [
    "distance_to_origin",
    "highpass_fft_filter",
    "lowpass_fft_filter",
    "modified_hann",
    "Window",
]
