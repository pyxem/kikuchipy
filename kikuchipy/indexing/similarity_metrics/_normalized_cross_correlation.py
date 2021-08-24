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

from typing import Union

import dask
import dask.array as da
import numpy as np

from kikuchipy.indexing.similarity_metrics._similarity_metric import SimilarityMetric


class NormalizedCrossCorrelationMetric(SimilarityMetric):
    r"""Similarity metric implementing the normalized cross-correlation,
    or Pearson Correlation Coefficient :cite:`gonzalzez2017digital`

    .. math::

        r = \frac
            {\sum^n_{i=1}(x_i - \bar{x})(y_i - \bar{y})}
            {
                \sqrt{\sum ^n _{i=1}(x_i - \bar{x})^2}
                \sqrt{\sum ^n _{i=1}(y_i - \bar{y})^2}
            },

    where experimental patterns :math:`x` and simulated patterns
    :math:`y` are centered by subtracting out the mean of each pattern,
    and the sum of cross-products of the centered patterns is
    accumulated. The denominator adjusts the scales of the patterns to
    have equal units.

    Equivalent results are obtained with :func:`dask.array.tensordot`
    with ``axes=([2, 3], [1, 2]))`` for 4D and 3D experimental and
    simulated data sets, respectively.
    """
    allowed_dtypes = [np.float32, np.float64]

    def prepare_experimental(
        self, patterns: Union[np.ndarray, da.Array]
    ) -> Union[np.ndarray, da.Array]:
        n_patterns = int(np.prod(patterns.shape[: self.navigation_dimension]))
        patterns = patterns.reshape((n_patterns, -1))
        print(patterns.shape)
        return self._prepare_patterns(patterns)

    def prepare_dictionary(
        self, patterns: Union[np.ndarray, da.Array]
    ) -> Union[np.ndarray, da.Array]:
        # Reshaping to 2D array done in _dictionary_indexing.py
        return self._prepare_patterns(patterns)

    def match(
        self,
        experimental: Union[np.ndarray, da.Array],
        dictionary: Union[np.ndarray, da.Array],
    ) -> da.Array:
        return da.einsum(
            "ik,mk->im",
            experimental,
            dictionary,
            optimize=True,
            dtype=self.dtype,
        )

    def _prepare_patterns(
        self, patterns: Union[np.ndarray, da.Array]
    ) -> Union[np.ndarray, da.Array]:
        if isinstance(patterns, np.ndarray):
            dispatcher = np
        else:
            dispatcher = da
        patterns = patterns.astype(self.dtype)

        mask = self.signal_mask
        if mask is not None:
            with dask.config.set(**{"array.slicing.split_large_chunks": True}):
                patterns = patterns[:, mask.ravel()]

        if self.rechunk and dispatcher == da:
            patterns = patterns.rechunk(("auto", -1))

        patterns_mean = dispatcher.mean(patterns)
        patterns = patterns - patterns_mean

        patterns_norm = dispatcher.sqrt(
            dispatcher.sum(dispatcher.square(patterns), axis=1, keepdims=True)
        )
        patterns = patterns / patterns_norm

        return patterns
