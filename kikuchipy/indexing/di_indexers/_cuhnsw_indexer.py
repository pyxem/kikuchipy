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

"""Private tools for custom dictionary indexing of experimental
 patterns to a dictionary of simulated patterns with known orientations.
"""

from typing import List, Union, Optional
import numpy as np

# try to import CuHNSW and notify user if not available
try:
    from cuhnsw import CuHNSW
except ImportError:
    CuHNSW = None
    print("cuhnsw is not installed. " "CUHNSWlibIndexer will not be available for use.")

from kikuchipy.indexing.di_indexers._di_indexer import DIIndexer


class CUHNSWlibIndexer(DIIndexer):
    """Dictionary indexing using HNSWlib.

    Summary
    -------
    This class implements a function to match experimental and simulated
    EBSD patterns in a dictionary using HNSWlib. The dictionary must be
    able to fit in memory. The dictionary is indexed using the HNSWlib
    library, which is a fast approximate nearest neighbor search library
    for high dimensional data. More information can be found at the HNSWlib
    GitHub repository:

    https://github.com/nmslib/hnswlib

    This is the CPU implementation of the HNSWlib indexer. The GPU implementation
    can be found in the kikuchipy.indexing.di_indexers._hnswlib_indexer_gpu module.

    """

    _allowed_dtypes: List[type] = [np.float32, np.float64]

    def __init__(
        self,
        dtype: Union[str, np.dtype, type],
        normalize: bool = True,
        zero_mean: bool = True,
        space: str = "cosine",
        ef_construction: int = 150,
        ef_search: int = 150,
        max_m: int = 16,
        max_m0: int = 32,
        load_filename: Optional[str] = None,
        save_filename: Optional[str] = None,
    ):
        """Initialize the HNSWlib indexer.

        Parameters
        ----------
        dtype : numpy.dtype
            Data type of the dictionary and experimental patterns.
        kwargs
            Additional keyword arguments to pass to the HNSWlib library.
            See the HNSWlib GitHub repository for more information.

        """
        super().__init__(dtype=dtype)
        self._normalize = normalize
        self._zero_mean = zero_mean
        self.space = space
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.max_m = max_m
        self.max_m0 = max_m0
        self.load_filename = load_filename
        self.save_filename = save_filename
        self._graph = None

    def prepare_dictionary(self):
        """Prepare the dictionary_patterns for indexing."""
        if self._normalize:
            self.dictionary_patterns = _normalize_patterns(self.dictionary_patterns)
        if self._zero_mean:
            self.dictionary_patterns = _zero_mean_patterns(self.dictionary_patterns)
        self.dictionary_patterns = self.dictionary_patterns.astype(self.dtype)
        self._graph = _build_index(
            self.dictionary_patterns,
            self.space,
            self.ef_construction,
            self.max_m,
            self.max_m0,
            self.load_filename,
            self.save_filename,
        )

    def prepare_experimental(self, experimental_patterns: np.ndarray) -> np.ndarray:
        """Prepare the experimental_patterns for indexing."""
        experimental_patterns = experimental_patterns.astype(self.dtype)
        if self._normalize:
            experimental_patterns = _normalize_patterns(experimental_patterns)
        if self._zero_mean:
            experimental_patterns = _zero_mean_patterns(experimental_patterns)
        return experimental_patterns

    def query(self, experimental_patterns: np.ndarray):
        """Query the dictionary.

        Parameters
        ----------
        experimental_patterns : numpy.ndarray
            Experimental patterns to match to the dictionary.

        Returns
        -------
        numpy.ndarray
            The indices of the dictionary patterns that match the experimental
            patterns.

        """
        experimental_patterns = self.prepare_experimental(experimental_patterns)
        return _search_index(
            self._graph,
            self.keep_n,
            experimental_patterns,
            self.ef_search,
        )


def _build_index(
    dictionary_patterns,
    space: str,
    ef_construction: int,
    max_m: int,
    max_m0: int,
    load_filename: Optional[str] = None,
    save_filename: Optional[str] = None,
):
    """Build the index using the HNSWlib library.

    Parameters
    ----------
    dictionary_patterns : numpy.ndarray
        Dictionary patterns to index.
    kwargs
        Additional keyword arguments to pass to the HNSWlib library.
        See the HNSWlib GitHub repository for more information.

    Returns
    -------
    hnswlib.Index
        The index built using the HNSWlib library.

    """
    hnsw_graph = CuHNSW(
        opt={
            "distType": space,
            "maxM": max_m,
            "maxM0": max_m0,
            "efConstruction": ef_construction,
        }
    )
    if load_filename is not None:
        hnsw_graph.load_index(load_filename)
    else:
        hnsw_graph.set_data(dictionary_patterns)
        hnsw_graph.build()
        if save_filename is not None:
            hnsw_graph.save_index(save_filename)
    return hnsw_graph


def _search_index(
    index: CuHNSW, keep_n: int, experimental_patterns: np.ndarray, ef_search: int
):
    """Search the index using the HNSWlib library.

    Parameters
    ----------
    index : hnswlib.Index
        The index built using the HNSWlib library.
    keep_n : int
        The number of dictionary patterns to return for each experimental
        pattern.
    experimental_patterns : numpy.ndarray
        Experimental patterns to search.

    Returns
    -------
    numpy.ndarray
        The indices of the dictionary patterns that match the
        experimental patterns.
    numpy.ndarray
        The distances of the dictionary patterns that match the
        experimental patterns.

    """
    indices, distances, _ = index.search_knn(
        experimental_patterns,
        topk=keep_n,
        ef_search=ef_search,
    )
    return indices, distances


def _normalize_patterns(patterns: np.ndarray) -> np.ndarray:
    """Normalize the patterns."""
    patterns /= np.linalg.norm(patterns, axis=1)[:, None]
    return patterns


def _zero_mean_patterns(patterns: np.ndarray) -> np.ndarray:
    """Zero mean the patterns."""
    patterns -= np.mean(patterns, axis=1)[:, None]
    return patterns
