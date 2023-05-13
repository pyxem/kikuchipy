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

from typing import List, Union
import numpy as np

# try to import cuml and notify user if not available
try:
    from cuml import PCA, NearestNeighbors
    from cuml.preprocessing import StandardScaler
except ImportError:
    cuml = None
    print(
        "cuml is not installed and will not be available for use."
    )

from kikuchipy.indexing.di_indexers._di_indexer import DIIndexer


class PCAIndexerCuml(DIIndexer):
    """Dictionary indexing using cuml.

    Summary
    -------
    This class implements a function to match experimental and simulated
    EBSD patterns in a dictionary using cuml. The dictionary must be able
    to fit in memory. The dictionary is indexed using the cuml library,
    specifically the NearestNeighbors class. More information can be found
    at the cuml documentation:

    https://docs.rapids.ai/api/cuml/stable/api.html#nearest-neighbors

    """

    _allowed_dtypes: List[type] = [np.float32, np.float64]

    def __init__(self,
                 zero_mean: bool = True,
                 unit_var: bool = True,
                 space: str = "cosine",
                 n_components: int = 700,
                 whiten: bool = True,
                 datatype: Union[str, np.dtype, type] = np.float32
                 ):
        """Initialize the HNSWlib indexer.

        Parameters
        ----------
        zero_mean
            Whether to zero mean the patterns before indexing.
        unit_var
            Whether to normalize the patterns to unit variance before indexing.
        space
            The space to use for indexing. See cuml documentation for more
            information.
        n_components
            The number of components to use for PCA.
        whiten
            Whether to whiten the patterns before indexing.
        datatype
            The datatype to cast the patterns to before indexing.

        """
        super().__init__(datatype=datatype)
        self._unit_var = unit_var
        self._zero_mean = zero_mean
        self._preprocessor = StandardScaler(with_mean=zero_mean, with_std=unit_var)
        self._space = space
        self._n_components = n_components
        self._whiten = whiten
        self._pca = None
        self._dictionary_pca_components = None
        self._knn_lookup_object = None

    def prepare_dictionary(self, dictionary_patterns: np.ndarray):
        """Prepare the dictionary_patterns for indexing.

        """
        dictionary_patterns = self._preprocessor.fit_transform(dictionary_patterns)

        # change dictionary to datatype
        dictionary_patterns = dictionary_patterns.astype(self.datatype)

        # make incremental PCA object
        self._pca = PCA(n_components=self._n_components, whiten=self._whiten)

        # make the cuml nearest neighbors object
        self._dictionary_pca_components = self._pca.fit_transform(dictionary_patterns)

        # we don't need the dictionary patterns anymore
        del dictionary_patterns

        # fit the knn lookup object
        self._knn_lookup_object = NearestNeighbors(n_neighbors=self.keep_n, metric=self._space)
        self._knn_lookup_object.fit(self._dictionary_pca_components)

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
        experimental_patterns = self._preprocessor.transform(experimental_patterns.astype(self.datatype))
        experimental_pca_components = self._pca.transform(experimental_patterns)
        distances, indices = self._knn_lookup_object.kneighbors(experimental_pca_components)
        return indices, distances
