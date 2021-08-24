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


class NormalizedDotProductMetric(SimilarityMetric):
    r"""Similarity metric implementing the normalized dot product
    :cite:`chen2015dictionary`

    .. math::

        \rho = \frac
        {\langle \mathbf{X}, \mathbf{Y} \rangle}
        {||\mathbf{X}|| \cdot ||\mathbf{Y}||},

    where :math:`{\langle \mathbf{X}, \mathbf{Y} \rangle}` is the dot
    (inner) product of the pattern vectors :math:`\mathbf{X}` and
    :math:`\mathbf{Y}`.
    """
    allowed_dtypes = [np.float32, np.float64]

    def prepare_experimental(
        self, patterns: Union[np.ndarray, da.Array]
    ) -> Union[np.ndarray, da.Array]:
        n_patterns = int(np.prod(patterns.shape[: self.navigation_dimension]))
        patterns = patterns.reshape((n_patterns, -1))
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

        patterns_norm = dispatcher.sqrt(
            dispatcher.sum(dispatcher.square(patterns), axis=1, keepdims=True)
        )
        patterns = patterns / patterns_norm

        return patterns
