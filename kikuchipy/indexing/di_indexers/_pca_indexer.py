import numpy as np
from numba import njit, prange
import heapq
from scipy.linalg import blas, lapack
from typing import Tuple

from kikuchipy.indexing.di_indexers import DIIndexer


@njit
def _fast_l2(query, dictionary_transposed, dictionary_half_norms):
    """
    Compute the L2 squared distance between a query and a dictionary of vectors.

    Parameters
    ----------
    query : np.ndarray
    dictionary_transposed : np.ndarray
    dictionary_half_norms : np.ndarray

    Returns
    -------
    np.ndarray

    """
    pseudo_distances = dictionary_half_norms - np.dot(query, dictionary_transposed)
    return pseudo_distances


@njit
def pseudo_to_real_l2_norm(pseudo_distances, query_pca):
    """
    Convert pseudo L2 distances to real L2 distances.

    Parameters
    ----------
    pseudo_distances : np.ndarray
    query_pca : np.ndarray

    Returns
    -------
    np.ndarray

    """
    return pseudo_distances + np.sum(query_pca ** 2, axis=1)[:, np.newaxis]


@njit(parallel=True)
def find_smallest_indices_values(matrix, k):
    """
    Find the smallest k values in each row of a matrix.

    Parameters
    ----------
    matrix
    k

    Returns
    -------
    smallest_indices : np.ndarray
    smallest_values : np.ndarray

    """
    n_rows = matrix.shape[0]
    smallest_indices = np.zeros((n_rows, k), dtype=np.int32)
    smallest_values = np.zeros((n_rows, k), dtype=matrix.dtype)

    for i in prange(n_rows):
        row = -1.0 * matrix[i]
        heap = [(val, idx) for idx, val in enumerate(row[:k])]
        heapq.heapify(heap)

        for idx in range(k, len(row)):
            heapq.heappushpop(heap, (row[idx], idx))

        sorted_heap = sorted(heap)

        for j in range(k):
            smallest_values[i, j] = sorted_heap[j][0]
            smallest_indices[i, j] = sorted_heap[j][1]

    return smallest_indices, smallest_values


def _eigendecomposition(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the eigenvector decomposition of a matrix.
    Uses LAPACK routines `ssyrk` followed by `dsyevd`.
    Parameters
    ----------
    matrix

    Returns
    -------
    eigenvalues : np.ndarray
    eigenvectors : np.ndarray

    """
    """
    Compute the eigenvector decomposition of a matrix.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to compute the eigenvector decomposition of.

    Returns
    -------
    np.ndarray
        The eigenvector decomposition of the matrix.

    """
    if matrix.dtype == np.float32:
        # Note: ssyrk computes alpha*A*A.T + beta*C, so alpha and beta should be set to 1 and 0 respectively
        cov_matrix = blas.ssyrk(a=matrix.T, alpha=1.0)

    elif matrix.dtype == np.float64:
        # Note: dsyrk computes alpha*A*A.T + beta*C, so alpha and beta should be set to 1 and 0 respectively
        cov_matrix = blas.dsyrk(a=matrix.T, alpha=1.0)
        # Compute eigenvector decomposition with dsyevd
    else:
        raise TypeError(f"Unsupported dtype {matrix.dtype}")

    w, v, info = lapack.dsyevd(cov_matrix.astype(np.float64), compute_v=True)

    # if the info is not 0, then something went wrong
    if info != 0:
        raise RuntimeError("dsyevd (symmetric eigenvalue decomposition call to LAPACK) failed with info=%d" % info)

    return w, v


class PCAKNNSearch:
    """
    A PCA KNN search object. Look up nearest neighbors in a PCA component space.

    """

    def __init__(self,
                 n_components: int = 100,
                 n_neighbors: int = 1,
                 whiten: bool = True,
                 datatype: str = 'float32'):
        """

        Parameters
        ----------
        n_components : int
            The number of components to keep in the PCA. The default is 100.
        n_neighbors : int
            The number of nearest neighbors to return. The default is 1.
        whiten : bool
            Whether to whiten the patterns. The default is True.
        datatype : str
            The dtype to use for matching. The default is 'float32'.

        """

        self._n_components = n_components
        self._n_neighbors = n_neighbors
        self._whiten = whiten
        self._datatype = datatype

        self._n_features = None
        self._singular_values_ = None
        self._explained_variance_ = None
        self._explained_variance_ratio_ = None

        self._transformation_matrix = None
        self._inverse_transformation_matrix = None

        self._dictionary_pca_T = None
        self._dictionary_pca_half_norms = None

    def fit(self, dictionary: np.ndarray):
        """
        Fit the PCA KNN search object to a dictionary of patterns.

        Parameters
        ----------
        dictionary : np.ndarray
            The dictionary of patterns to fit the PCA KNN search object to.

        """
        # check the data dimensons
        if dictionary.ndim != 2:
            raise ValueError("A must be a 2D array but got %d dimensions" % dictionary.ndim)

        # set the data to contiguous and C order as the correct dtype
        data = np.ascontiguousarray(dictionary.astype(self._datatype))

        # get the number of samples and features
        n_entries, n_features = data.shape
        self._n_features = n_features

        # check the n_components is less than the number of features
        if self._n_components > n_features:
            raise ValueError(f"n_components, {self._n_components} given, exceeds features in A, {n_features}")

        # compute the eigenvector decomposition
        eigenvalues, eigenvectors = _eigendecomposition(data)

        # sort the eigenvalues and eigenvectors
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # keep only the first n_components and make sure the arrays are contiguous in C order
        eigenvalues = np.ascontiguousarray(eigenvalues[:self._n_components]).astype(self._datatype)
        eigenvectors = np.ascontiguousarray(eigenvectors[:, :self._n_components]).astype(self._datatype)

        # calculate the explained variance
        self._explained_variance_ = eigenvalues / (n_entries - 1)
        self._explained_variance_ratio_ = self._explained_variance_ / np.sum(self._explained_variance_)

        # calculate the singular values
        self._singular_values_ = np.sqrt(eigenvalues)

        # components is an alias for the un-whitened eigenvectors
        # calculate the transformation matrix
        if self._whiten:
            self._transformation_matrix = eigenvectors * np.sqrt(n_entries - 1) / self._singular_values_[
                                                                                  :self._n_components]
            self._inverse_transformation_matrix = np.linalg.pinv(self._transformation_matrix)
        else:
            self._transformation_matrix = eigenvectors
            self._inverse_transformation_matrix = np.linalg.pinv(self._transformation_matrix)

        # make sure the transformation matrix is contiguous in C order
        self._transformation_matrix = np.ascontiguousarray(self._transformation_matrix)
        self._inverse_transformation_matrix = np.ascontiguousarray(self._inverse_transformation_matrix)

        # calculate the PCA of the dictionary using calls to BLAS
        dictionary_pca = blas.dgemm(a=data, b=self._transformation_matrix, alpha=1.0)

        # calculate the norms of the PCA of the dictionary
        self._dictionary_pca_half_norms = (np.sum(dictionary_pca ** 2, axis=1) / 2.0).astype(self._datatype)
        # make sure the transpose of the dictionary PCA is contiguous in C order
        self._dictionary_pca_T = np.ascontiguousarray(dictionary_pca.T).astype(self._datatype)

    def query(self, query: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query the dictionary.

        Parameters
        ----------
        query : np.ndarray
            The query to perform.

        Returns
        -------
        np.ndarray
            The indices of the nearest neighbors.
        np.ndarray
            The distances to the nearest neighbors.

        """
        query_pca = np.dot(query, self._transformation_matrix)
        # calculate distance matrix
        pseudo_distances = _fast_l2(query_pca, self._dictionary_pca_T, self._dictionary_pca_half_norms)
        # get the indices of the nearest neighbors
        indices, pseudo_distances = find_smallest_indices_values(pseudo_distances, self._n_neighbors)
        # make the distances true distances by adding the query norms
        distances = pseudo_distances + np.sum(query_pca ** 2, axis=1)[:, np.newaxis]
        return indices, distances


class PCAIndexer(DIIndexer):
    """
    Class for performing PCA KNN search using JAX. All nearest neighbor
    lookups are performed in the truncated PCA space. This class is a
    wrapper around the PCAKNNSearch class that allows for the use of
    """

    def __init__(self,
                 zero_mean: bool = True,
                 unit_var: bool = False,
                 n_components: int = 750,
                 n_neighbors: int = 1,
                 whiten: bool = False,
                 datatype: str = 'float32'):
        """
        Parameters
        ----------
        n_components : int, optional
            The number of components to keep in the PCA. The default is 750.
        n_neighbors : int, optional
            The number of nearest neighbors to return. The default is 1.
        whiten : bool, optional
            Whether to whiten the patterns. The default is False.
        datatype : str, optional
            The datatype to use for matching. The default is 'float32'.
        """
        super().__init__(datatype=datatype)
        self._pca_knn_search = PCAKNNSearch(n_components=n_components,
                                            n_neighbors=n_neighbors,
                                            whiten=whiten,
                                            datatype=datatype)
        self._zero_mean = zero_mean
        self._unit_var = unit_var

    def prepare_dictionary(self, dictionary: np.ndarray):
        """
        Fit the PCA and KNN lookup objects.

        Parameters
        ----------
        dictionary : numpy.ndarray
            The dictionary to fit.

        Returns
        -------
        None.

        """
        dictionary = dictionary.astype(self.datatype)
        # preprocess the dictionary
        if self._zero_mean:
            dictionary -= np.mean(dictionary, axis=0)[None, :]
        if self._unit_var:
            dictionary /= np.std(dictionary, axis=0)[None, :]
        self._pca_knn_search.fit(dictionary)

    def query(self, query: np.ndarray):
        """
        Query the dictionary.

        Parameters
        ----------
        query : np.ndarray
            The query to perform. Must be of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            The indices of the nearest neighbors.
        np.ndarray
            The distances to the nearest neighbors.

        """
        query = query.astype(self.datatype)
        # preprocess the query
        if self._zero_mean:
            query -= np.mean(query, axis=0)[None, :]
        if self._unit_var:
            query /= np.std(query, axis=0)[None, :]
        return self._pca_knn_search.query(query)
