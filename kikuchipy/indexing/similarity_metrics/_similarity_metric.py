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

import abc
from typing import Optional

from hyperspy.axes import AxesManager
import numpy as np

NDIM_TO_CHUNKS = {
    2: {0: -1, 1: -1},
    3: {0: "auto", 1: -1, 2: -1},
    4: {0: "auto", 1: "auto", 2: -1, 3: -1},
}


class SimilarityMetric(abc.ABC):
    def __init__(
        self,
        experimental_navigation_dimension: int = 2,
        simulated_navigation_dimension: int = 1,
        signal_dimension: int = 2,
        signal_mask: float = 1,
        greater_is_better: bool = True,
        dtype: np.dtype = np.float32,
        can_rechunk: Optional[bool] = None,
    ):
        self.experimental_navigation_dimension = experimental_navigation_dimension
        self.simulated_navigation_dimension = simulated_navigation_dimension
        self.signal_dimension = signal_dimension
        self.signal_mask = signal_mask
        self.dtype = dtype
        self.can_rechunk = can_rechunk
        if greater_is_better:
            self.sign = 1
        else:
            self.sign = -1

    def __call__(self, experimental, simulated):
        experimental = self.prepare_all_experimental(experimental)
        simulated = self.prepare_chunk_simulated(simulated)
        return self.compare(experimental, simulated)

    @property
    def navigation_indices_in_experimental_array(self):
        return (0, 1, 2, 3)[: self.experimental_navigation_dimension]

    @property
    def signal_indices_in_experimental_array(self):
        exp_nav_ndim = self.experimental_navigation_dimension
        return (0, 1, 2, 3)[exp_nav_ndim : exp_nav_ndim + self.signal_dimension]

    @property
    def navigation_indices_in_simulated_array(self):
        return (0, 1, 2, 3)[: self.simulated_navigation_dimension]

    @property
    def signal_indices_in_simulated_array(self):
        sim_nav_ndim = self.simulated_navigation_dimension
        return (0, 1, 2, 3)[sim_nav_ndim : sim_nav_ndim + self.signal_dimension]

    @abc.abstractmethod
    def prepare_all_experimental(self, *args, **kwargs):
        return NotImplemented

    @abc.abstractmethod
    def prepare_chunk_simulated(self, *args, **kwargs):
        return NotImplemented

    @abc.abstractmethod
    def compare(self, *args, **kwargs):
        return NotImplemented

    def rechunk(self, patterns):
        chunks = NDIM_TO_CHUNKS[self.experimental_navigation_dimension]
        return patterns.rechunk(chunks)

    def set_shapes_from_axes_managers(
        self,
        experimental_axes_manager: AxesManager,
        simulated_axes_manager: AxesManager,
    ):
        exp_am = experimental_axes_manager
        sim_am = simulated_axes_manager
        self.experimental_navigation_dimension = exp_am.navigation_dimension
        self.simulated_navigation_dimension = sim_am.navigation_dimension
        self.signal_dimension = exp_am.signal_dimension
