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

from typing import List, Union

import dask
import dask.array as da
from numba import njit
import numpy as np

from kikuchipy.indexing.similarity_metrics import SimilarityMetric


class NormalizedCrossCorrelationMetric(SimilarityMetric):
    r"""Similarity metric implementing the normalized cross-correlation,
    or Pearson Correlation Coefficient :cite:`gonzalez2017digital`.

    The metric is defined as

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

    See :class:`~kikuchipy.indexing.SimilarityMetric` for the
    description of the initialization parameters and the list of
    attributes.
    """
    _allowed_dtypes: List[type] = [np.float32, np.float64]
    _sign: int = 1

    def __call__(
        self,
        experimental: Union[da.Array, np.ndarray],
        dictionary: Union[da.Array, np.ndarray],
    ) -> da.Array:
        """Compute the similarities between experimental patterns and
        simulated dictionary patterns.

        Before calling :meth:`match`, this method calls
        :meth:`prepare_experimental`, reshapes the dictionary patterns
        to 1 navigation dimension and 1 signal dimension, and calls
        :meth:`prepare_dictionary`.

        Parameters
        ----------
        experimental
            Experimental pattern array with as many patterns as
            :attr:`n_experimental_patterns`.
        dictionary
            Dictionary pattern array with as many patterns as
            :attr:`n_dictionary_patterns`.

        Returns
        -------
        similarities
        """
        experimental = self.prepare_experimental(experimental)
        dictionary = dictionary.reshape((self.n_dictionary_patterns, -1))
        dictionary = self.prepare_dictionary(dictionary)
        return self.match(experimental, dictionary)

    def prepare_experimental(
        self, patterns: Union[np.ndarray, da.Array]
    ) -> Union[np.ndarray, da.Array]:
        """Prepare experimental patterns before matching to dictionary
        patterns in :meth:`match`.

        Patterns are prepared by:
            1. Setting the data type to :attr:`dtype`.
            2. Excluding the experimental patterns where
               :attr:`navigation_mask` is ``False`` if the mask is set.
            3. Reshaping to shape ``(n_experimental_patterns, -1)``
            4. Applying a signal mask if :attr:`signal_mask` is set.
            5. Rechunking if :attr:`rechunk` is ``True``.
            6. Normalizing to a mean of 0 and a standard deviation of 1.

        Parameters
        ----------
        patterns
            Experimental patterns.

        Returns
        -------
        prepared_patterns
            Prepared experimental patterns.
        """
        patterns = da.asarray(patterns).astype(self.dtype)

        patterns = patterns.reshape((self.n_experimental_patterns, -1))

        if self.navigation_mask is not None:
            patterns = patterns[~self.navigation_mask.ravel()]

        if self.signal_mask is not None:
            patterns = self._mask_patterns(patterns)

        if self.rechunk:
            patterns = patterns.rechunk(("auto", -1))

        prepared_patterns = self._zero_mean_normalize_patterns(patterns)

        return prepared_patterns

    def prepare_dictionary(
        self, patterns: Union[np.ndarray, da.Array]
    ) -> Union[np.ndarray, da.Array]:
        """Prepare dictionary patterns before matching to experimental
        patterns in :meth:`match`.

        Patterns are prepared by:
            1. Setting the data type to :attr:`dtype`.
            2. Applying a signal mask if :attr:`signal_mask` is set.
            3. Normalizing to a mean of 0 and a standard deviation of 1.

        Parameters
        ----------
        patterns
            Dictionary patterns.

        Returns
        -------
        prepared_patterns
            Prepared dictionary patterns.
        """

        patterns = patterns.astype(self.dtype)

        if self.signal_mask is not None:
            patterns = self._mask_patterns(patterns)

        prepared_patterns = self._zero_mean_normalize_patterns(patterns)

        return prepared_patterns

    def match(
        self,
        experimental: Union[np.ndarray, da.Array],
        dictionary: Union[np.ndarray, da.Array],
    ) -> da.Array:
        """Match all experimental patterns to all dictionary patterns
        and return their similarities.

        Parameters
        ----------
        experimental
            Experimental patterns.
        dictionary
            Dictionary patterns.

        Returns
        -------
        scores
            Normalized cross-correlation scores.
        """
        return da.einsum(
            "ik,mk->im", experimental, dictionary, optimize=True, dtype=self.dtype
        )

    def _mask_patterns(
        self, patterns: Union[da.Array, np.ndarray]
    ) -> Union[da.Array, np.ndarray]:
        with dask.config.set(**{"array.slicing.split_large_chunks": False}):
            patterns = patterns[:, ~self.signal_mask.ravel()]
        return patterns

    @staticmethod
    def _zero_mean_normalize_patterns(
        patterns: Union[da.Array, np.ndarray]
    ) -> Union[da.Array, np.ndarray]:
        if isinstance(patterns, np.ndarray):
            return _zero_mean_normalize_patterns_numpy(patterns)
        else:
            return _zero_mean_normalize_patterns_dask(patterns)


@njit("float64(float32[:], float32[:], float32)", cache=True, nogil=True, fastmath=True)
def _ncc_single_patterns_1d_float32_exp_centered(
    exp: np.ndarray, sim: np.ndarray, exp_squared_norm: float
) -> float:
    """Return the normalized cross-correlation (NCC) coefficient
    between two 1D patterns.

    Parameters
    ----------
    exp
        1D array of shape (n_pixels,) and data type 32-bit floats
        already centered.
    sim
        1D array of shape (n_pixels,) and data type 32-bit floats.
    exp_squared_norm
        Squared norm of experimental pattern as 32-bit float.

    Returns
    -------
    ncc
        NCC coefficient as 64-bit float.
    """
    sim -= np.mean(sim)
    return np.divide(
        np.sum(exp * sim), np.sqrt(exp_squared_norm * np.sum(np.square(sim)))
    )


def _zero_mean_normalize_patterns_numpy(patterns: np.ndarray) -> np.ndarray:
    patterns_mean = np.mean(patterns, axis=1, keepdims=True)
    patterns -= patterns_mean
    patterns_norm = np.sqrt(np.sum(np.square(patterns), axis=1, keepdims=True))
    patterns /= patterns_norm
    return patterns


def _zero_mean_normalize_patterns_dask(patterns: da.Array) -> da.Array:
    patterns_mean = da.mean(patterns, axis=1, keepdims=True)
    patterns -= patterns_mean
    patterns_norm = da.sqrt(da.sum(da.square(patterns), axis=1, keepdims=True))
    patterns /= patterns_norm
    return patterns
