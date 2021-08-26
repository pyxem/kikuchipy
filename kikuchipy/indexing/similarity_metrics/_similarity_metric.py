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
    """Abstract class implementing a similarity metric to match
    experimental and simulated EBSD patterns in a dictionary.

    For use in :meth:`~kikuchipy.signals.EBSD.dictionary_indexing` or
    directly on pattern arrays if a :meth:`__call__` method is
    implemented. Note that `dictionary_indexing()` will always reshape
    the dictionary pattern array to 2D (1 navigation dimension, 1 signal
    dimension) before calling :meth:`prepare_dictionary` and
    :meth:`match`.

    Take a look at the implementation of
    :class:`~kikuchipy.indexing.similarity_metrics.NormalizedCrossCorrelationMetric`
    for how to write a concrete custom metric.

    When writing a custom similarity metric class, the methods listed as
    `abstract` below must be implemented. Any number of custom
    parameters can be passed. Also listed are the attributes available
    to the methods if set properly during initialization or after.

    Attributes
    ----------
    allowed_dtypes : list of numpy.dtype
        List of allowed array data types used during matching.
    dtype : numpy.dtype
        Which data type to cast the patterns to before matching. Must be
        listed in `allowed_dtypes`.
    n_dictionary_patterns : int or None
        Number of dictionary patterns to match. This information might
        be necessary when reshaping the dictionary array in
        :meth:`prepare_dictionary`.
    n_experimental_patterns : int or None
        Number of experimental patterns to match. This information might
        be necessary when reshaping the dictionary array in
        :meth:`prepare_experimental`.
    sign : int or None
        +1 if a greater match is better, -1 if a lower match is better.
        This must be set in the inheriting class.
    signal_mask : numpy.ndarray or None
        A boolean mask equal to the experimental patterns' detector
        shape (n rows, n columns), where only pixels equal to False
        are matched.
    rechunk : bool
        Whether to allow rechunking of arrays before matching.
    """

    allowed_dtypes = []
    sign = None

    def __init__(
        self,
        n_experimental_patterns: Optional[int] = None,
        n_dictionary_patterns: Optional[int] = None,
        signal_mask: Optional[np.ndarray] = None,
        dtype: type = np.float32,
        rechunk: bool = False,
    ):
        """Create a similarity metric matching experimental and
        simulated EBSD patterns in a dictionary.

        Parameters
        ----------
        n_experimental_patterns
            Number of experimental patterns. If not given, this is set
            to None and must be set later. Must be at least 1.
        n_dictionary_patterns
            Number of dictionary patterns. If not given, this is set to
            None and must be set later. Must be at least 1.
        signal_mask
            A boolean mask equal to the experimental patterns' detector
            shape (n rows, n columns), where only pixels equal to False
            are matched. If not given, all pixels are used.
        dtype
            Which data type to cast the patterns to before matching to.
        rechunk
            Whether to allow rechunking of arrays before matching.
            Default is False.
        """
        self.n_experimental_patterns = n_experimental_patterns
        self.n_dictionary_patterns = n_dictionary_patterns
        self.signal_mask = signal_mask
        self.dtype = dtype
        self.rechunk = rechunk

    def __repr__(self):
        string = f"{self.__class__.__name__}: {np.dtype(self.dtype).name}, "
        sign_string = {1: "greater is better", -1: "lower is better"}
        string += sign_string[self.sign]
        string += f", rechunk: {self.rechunk}, "
        string += f"signal mask: {self.signal_mask is not None}"
        return string

    @abc.abstractmethod
    def prepare_experimental(self, *args, **kwargs):
        """Prepare experimental patterns before matching to dictionary
        patterns in :meth:`match`.
        """
        return NotImplemented  # pragma: no cover

    @abc.abstractmethod
    def prepare_dictionary(self, *args, **kwargs):
        """Prepare dictionary patterns before matching to experimental
        patterns in :meth:`match`.
        """
        return NotImplemented  # pragma: no cover

    @abc.abstractmethod
    def match(self, *args, **kwargs):
        """Match all experimental patterns to all dictionary patterns
        and return their similarities.
        """
        return NotImplemented  # pragma: no cover

    def raise_error_if_invalid(self):
        """Raise a ValueError if `self.dtype` is not among
        `self.allowed_dtypes` and the latter is not an empty list.
        """
        allowed_dtypes = self.allowed_dtypes
        if len(allowed_dtypes) != 0 and self.dtype not in allowed_dtypes:
            raise ValueError(
                f"Data type {np.dtype(self.dtype).name} not among supported data types "
                f"{allowed_dtypes}"
            )
