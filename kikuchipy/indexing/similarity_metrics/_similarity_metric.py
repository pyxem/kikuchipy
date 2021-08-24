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

import numpy as np


class SimilarityMetric(abc.ABC):
    """Abstract class implementing a similarity metric to match 1D or
    2D gray-tone patterns.

    When writing a custom similarity metric class, the following methods
    must be implemented. Also listed are the attributes available within
    the methods.

    Methods
    -------
    prepare_experimental
    prepare_dictionary
    match

    Attributes
    ----------
    allowed_dtypes : list of numpy.dtype
        List of allowed array data types.
    navigation_dimension : int or None
    signal_mask : numpy.ndarray or None
    sign : int
        +1 if a greater match is better, -1 if a lower match is better.
    dtype : numpy.dtype
    rechunk : bool
    """

    allowed_dtypes = []

    def __init__(
        self,
        navigation_dimension: Optional[int] = None,
        signal_mask: Optional[np.ndarray] = None,
        greater_is_better: bool = True,
        dtype: np.dtype = np.float32,
        rechunk: bool = True,
    ):
        """Create a similarity metric matching 1D or 2D gray-tone
        patterns.

        Parameters
        ----------
        navigation_dimension
            Number of experimental navigation dimensions, typically 1 or
            2. If not given, this is set to None, so it must be set
            later.
        signal_mask
            A boolean mask equal to the experimental patterns' detector
            shape (n rows, n columns), where only pixels equal to False
            are matched. If not given, all pixels are used.
        greater_is_better
            True if a higher metric means a better match.
        dtype
            Which data type to cast the patterns to before matching to.
        rechunk
            Whether to allow rechunking of arrays before matching.
        """
        self.navigation_dimension = navigation_dimension
        self.signal_mask = signal_mask
        self.dtype = dtype
        self.rechunk = rechunk
        if greater_is_better:
            self.sign = 1
        else:
            self.sign = -1

    def __call__(self, experimental, dictionary):
        experimental = self.prepare_experimental(experimental)
        dictionary = dictionary.reshape((dictionary.shape[0], -1))
        dictionary = self.prepare_dictionary(dictionary)
        return self.match(experimental, dictionary)

    def __repr__(self):
        string = f"{self.__class__.__name__}: {np.dtype(self.dtype).name}, "
        if self.sign == 1:
            string += "greater is better"
        else:
            string += "lower is better"
        string += f", rechunk: {self.rechunk}, mask: {self.signal_mask is not None}"
        return string

    @abc.abstractmethod
    def prepare_experimental(self, *args, **kwargs):
        """Prepare experimental patterns before being sent to
        :meth:`match`.
        """
        return NotImplemented

    @abc.abstractmethod
    def prepare_dictionary(self, *args, **kwargs):
        """Prepare dictionary patterns before being sent to
        :meth:`match`.
        """
        return NotImplemented

    @abc.abstractmethod
    def match(self, *args, **kwargs):
        """Match experimental and dictionary patterns and return their
        similarities.
        """
        return NotImplemented

    def raise_error_if_invalid(self):
        """Raise a ValueError if `self.dtype` is not among
        `self.allowed_dtypes` and the latter is not an empty list.
        """
        allowed_dtypes = self.allowed_dtypes
        if len(allowed_dtypes) != 0 and self.dtype not in allowed_dtypes:
            raise ValueError(
                f"Data type {self.dtype} not among supported data types "
                f"{allowed_dtypes}"
            )
