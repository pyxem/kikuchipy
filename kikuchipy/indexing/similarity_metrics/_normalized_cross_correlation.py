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

import dask.array as da
import numba as nb
import numpy as np

from kikuchipy.indexing.similarity_metrics._similarity_metric import SimilarityMetric


class NormalizedCrossCorrelationMetric(SimilarityMetric):
    @property
    def einsum_signature(self) -> str:
        return _einsum_signature(
            self.experimental_navigation_dimension,
            #            self.signal_dimension
            1,
        )

    @property
    def prepare_chunk_of_simulated_patterns_func(self):
        sig_dim = str(self.signal_dimension)
        dtype_itemsize = str(np.dtype(self.dtype).itemsize)
        return prepare_simulated_patterns_functions[sig_dim][dtype_itemsize]

    def prepare_all_experimental(self, patterns: da.Array, n_per_iteration) -> da.Array:
        patterns = patterns.astype(self.dtype)

        n_experimental = int(
            np.prod(patterns.shape[: self.experimental_navigation_dimension])
        )
        patterns = patterns.reshape((n_experimental, -1))
        if not isinstance(self.signal_mask, int):
            patterns = patterns[:, self.signal_mask]

        if self.can_rechunk:
            patterns = patterns.rechunk((n_per_iteration, -1))

        #        sig_axes = self.signal_indices_in_experimental_array
        patterns_mean = da.mean(patterns, axis=1, keepdims=True)
        patterns = patterns - patterns_mean
        patterns_norm = da.sqrt(da.sum(da.square(patterns), axis=1, keepdims=True))
        patterns = patterns / patterns_norm

        #        patterns = patterns * self.signal_mask

        return patterns

    def prepare_chunk_simulated(self, patterns: np.ndarray) -> np.ndarray:
        if not isinstance(self.signal_mask, int):
            patterns = patterns[:, self.signal_mask]

        patterns = self.prepare_chunk_of_simulated_patterns_func(patterns)
        #        patterns = patterns * self.signal_mask

        #        n_simulated = patterns.shape[0]
        #        patterns = patterns.reshape((n_simulated, -1))

        return patterns

    def compare(self, experimental: da.Array, simulated: np.ndarray) -> da.Array:
        return da.einsum(
            #            self.einsum_signature,
            "ij,mj->im",
            experimental,
            simulated,
            optimize="greedy",
            dtype=self.dtype,
        )


def _einsum_signature(experimental_navigation_dimension, signal_dimension):
    signal_signature = "kl"[:signal_dimension]
    exp_nav_signature = "ij"[:experimental_navigation_dimension]
    sim_nav_signature = "m"
    exp = exp_nav_signature + signal_signature
    sim = sim_nav_signature + signal_signature
    out = exp_nav_signature + sim_nav_signature
    return f"{exp},{sim}->{out}"


# ------------------ Single pattern Numba functions ------------------ #


@nb.jit("float32[:](float32[:])", cache=True, nogil=True, nopython=True)
def _zero_mean_normalize_pattern1d_float32(pattern: np.ndarray) -> np.ndarray:
    pattern_mean = np.mean(pattern)
    pattern = pattern - pattern_mean
    pattern_norm = np.sqrt(np.sum(np.square(pattern)))
    return pattern / pattern_norm


@nb.jit("float32[:, :](float32[:, :])", cache=True, nogil=True, nopython=True)
def _zero_mean_normalize_pattern2d_float32(pattern: np.ndarray) -> np.ndarray:
    pattern_mean = np.mean(pattern)
    pattern = pattern - pattern_mean
    pattern_norm = np.sqrt(np.sum(np.square(pattern)))
    return pattern / pattern_norm


@nb.jit("float64[:](float64[:])", cache=True, nogil=True, nopython=True)
def _zero_mean_normalize_pattern1d_float64(pattern: np.ndarray) -> np.ndarray:
    pattern_mean = np.mean(pattern)
    pattern = pattern - pattern_mean
    pattern_norm = np.sqrt(np.sum(np.square(pattern)))
    return pattern / pattern_norm


@nb.jit("float64[:, :](float64[:, :])", cache=True, nogil=True, nopython=True)
def _zero_mean_normalize_pattern2d_float64(pattern: np.ndarray) -> np.ndarray:
    pattern_mean = np.mean(pattern)
    pattern = pattern - pattern_mean
    pattern_norm = np.sqrt(np.sum(np.square(pattern)))
    return pattern / pattern_norm


# ----------------- Multiple pattern Numba functions ----------------- #


@nb.jit("float32[:, :](float32[:, :])", cache=True, nogil=True, nopython=True)
def _prepare_simulated_patterns1d_float32(sim: np.ndarray) -> np.ndarray:
    simulated_prepared = np.zeros_like(sim)
    for i in nb.prange(sim.shape[0]):
        simulated_prepared[i] = _zero_mean_normalize_pattern1d_float32(sim[i])
    return simulated_prepared


@nb.jit("float64[:, :](float64[:, :])", cache=True, nogil=True, nopython=True)
def _prepare_simulated_patterns1d_float64(sim: np.ndarray) -> np.ndarray:
    simulated_prepared = np.zeros_like(sim)
    for i in nb.prange(sim.shape[0]):
        simulated_prepared[i] = _zero_mean_normalize_pattern1d_float64(sim[i])
    return simulated_prepared


@nb.jit("float32[:, :, :](float32[:, :, :])", cache=True, nogil=True, nopython=True)
def _prepare_simulated_patterns2d_float32(sim: np.ndarray) -> np.ndarray:
    simulated_prepared = np.zeros_like(sim)
    for i in nb.prange(sim.shape[0]):
        simulated_prepared[i] = _zero_mean_normalize_pattern2d_float32(sim[i])
    return simulated_prepared


@nb.jit("float64[:, :, :](float64[:, :, :])", cache=True, nogil=True, nopython=True)
def _prepare_simulated_patterns2d_float64(sim: np.ndarray) -> np.ndarray:
    simulated_prepared = np.zeros_like(sim)
    for i in nb.prange(sim.shape[0]):
        simulated_prepared[i] = _zero_mean_normalize_pattern2d_float64(sim[i])
    return simulated_prepared


prepare_simulated_patterns_functions = {
    # Signal dimension
    "1": {
        # Data type
        "4": _prepare_simulated_patterns1d_float32,
        "8": _prepare_simulated_patterns1d_float64,
    },
    "2": {
        "4": _prepare_simulated_patterns2d_float32,
        "8": _prepare_simulated_patterns2d_float64,
    },
}
