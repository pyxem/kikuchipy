import numpy as np
from numba import njit, prange
import heapq
import pyopencl as cl
from pyopencl.array import Array, arange, empty, zeros
import pyclblast
from scipy.linalg.lapack import dsyevd
from kikuchipy.indexing.di_indexers._di_indexer import DIIndexer
from typing import List, Union

KERNEL_SUM_SQ = """
__kernel void row_sum_sq(__global const float *matrix, __global float *result, const int cols)
{
    int row = get_global_id(0);
    float sum = 0.0f;
    for (int col = 0; col < cols; col++)
    {
        float element = matrix[row * cols + col];
        sum = fma(element, element, sum);
    }
    result[row] = sum;
}
"""


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
        # this -1.0 makes it find mins
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


def blast_covariance_matrix(queue, data, datatype: Union[str, np.dtype]):
    """Compute the covariance matrix of A.

    Parameters
    ----------
    queue : pyopencl.CommandQueue
        The OpenCL command queue to use.
    data : numpy.ndarray
        The array to compute the covariance matrix of.
    datatype : str or numpy.dtype
        The dtype of the arrays.

    Returns
    -------
    numpy.ndarray
        The covariance matrix of A.

    """
    # Set up NumPy arrays
    n_samples, n_features = data.shape
    data = data.astype(datatype)
    # Set up OpenCL array
    cl_a = Array(queue, data.shape, data.dtype)
    cl_a.set(data)
    # Prepare an empty OpenCL array for the result
    cl_cov = Array(queue, (n_features, n_features), datatype)
    # Perform the dsyrk operation
    pyclblast.syrk(queue, n_features, n_samples, cl_a, cl_cov, n_features, n_features, alpha=1.0, beta=0.0,
                   lower_triangle=False, a_transp=True)
    # Transfer result from device to host
    covariance_matrix = cl_cov.get()
    return covariance_matrix


def blast_transform(queue: cl.CommandQueue,
                    cl_data: cl.array.Array,
                    cl_transform: cl.array.Array,
                    cl_out: cl.array.Array,
                    n_samples: int,
                    n_features: int,
                    n_components: int):
    """ Transform A using the transformation matrix T.

    Parameters
    ----------
    queue : pyopencl.CommandQueue
        The OpenCL queue to use.
    cl_data : pyopencl.array.Array
        The data array. Shape (n_samples, n_features).
    cl_transform : pyopencl.array.Array
        The transformation matrix. Shape (n_features, n_components).
    cl_out : pyopencl.array.Array
        The output array. Shape (n_samples, n_components).
    n_samples : int
        The number of samples.
    n_features : int
        The number of features.
    n_components : int
        The number of components.

    Returns
    -------
    numpy.ndarray
        The transformed array.

    """
    # Perform the gemm operation
    pyclblast.gemm(queue, n_samples, n_components, n_features, cl_data, cl_transform, cl_out,
                   a_ld=n_features,
                   b_ld=n_components,
                   c_ld=n_components)


class BlastNN:
    """Nearest neighbor search using pyclblast.

    """

    def __init__(self, queue: cl.CommandQueue, n_neighbors: int = 1, metric: str = 'l2',
                 datatype: Union[str, np.dtype] = 'float32'):
        """Initialize the BlastNN object.

                Parameters
                ----------
                queue : pyopencl.CommandQueue
                    The OpenCL queue to use.
                metric : str, optional
                    The metric to use. The default is 'l2'.
                datatype : str or numpy.dtype, optional
                    The dtype of the data. The default is 'float32'.
                """
        # check the datatype
        if isinstance(datatype, str):
            assert datatype in ['float64', 'float32']
        elif isinstance(datatype, np.dtype):
            assert datatype in [np.float64, np.float32]
        else:
            raise TypeError("dtype must be a string or a numpy dtype but got %s" % type(datatype))

        self.datatype = datatype
        self.n_neighbors = n_neighbors
        self.metric = metric

        # Set up data attributes
        self.n_data = None

        # set the queue
        self.queue = queue

        # set up the three OpenCL arrays
        self.cl_data_T = None
        self.cl_dists = None
        self.cl_data_norms = None

        # make programs for norms of rows
        self.norms_program = cl.Program(self.queue.context, KERNEL_SUM_SQ).build().row_sum_sq

    def fit(self, cl_data: cl.array.Array):
        """Initialize the BlastNN object.

        Parameters
        ----------
        cl_data : pyopencl.array.Array
            The data array.
        """
        # save the number of data entries
        self.n_data, n_features = cl_data.shape

        # compute the norms of the data
        self.cl_data_norms = zeros(self.queue, (self.n_data,), self.datatype)
        self.norms_program(self.queue, (self.n_data,), None, cl_data.data, self.cl_data_norms.data, np.int32(n_features))

        # Either data or query has to be transposed for GEMM. We choose to transpose data.
        # This will silently fail if you don't make the memory contiguous.
        self.cl_data_T = Array(self.queue, (n_features, self.n_data), self.datatype)
        self.cl_data_T.set(np.ascontiguousarray(cl_data.get().T).astype(self.datatype))

    def kneighbors(self, cl_q: cl.array.Array):
        """Query the data array.

        Parameters
        ----------
        cl_q : pyopencl.array.Array
            The query array as a PyOpenCL array.
        Returns
        -------
        numpy.ndarray
            The indices of the nearest neighbors.

        """
        n_queries, n_features = cl_q.shape

        # check if the dists array has the correct shape and exists
        if (self.cl_dists is None) or (self.cl_dists.shape != (n_queries, self.n_data)):
            self.cl_dists = empty(self.queue, (n_queries, self.n_data), self.datatype)

        # Perform the GEMM operation with beta=0.0
        pyclblast.gemm(self.queue, n_queries, self.n_data, n_features, cl_q, self.cl_data_T,
                       self.cl_dists,
                       a_ld=n_features,
                       b_ld=self.n_data,
                       c_ld=self.n_data,
                       alpha=-2.0,
                       beta=0.0)

        # get the inner product matrix
        ip_matrix_neg_two = self.cl_dists.get()

        # compose a pseudo l2 distance matrix (doesn't have the dictionary norms subtracted)
        ip_matrix_neg_two += self.cl_data_norms.get().reshape((1, self.n_data))

        # lookup the row min k values (numpy arg-partition doesn't use OpenMP, so it's really slow)
        # decided to use numba to compile a fast version of this using heapq
        indices, pseudo_distances = find_smallest_indices_values(ip_matrix_neg_two, self.n_neighbors)

        # add the query norms to make the distances true l2 distances squared
        distances = pseudo_distances + (cl_q**2).get().sum(axis=1).reshape((-1, 1))

        return indices, distances


class BlastPCA:
    """PCA using pyclblast to compute the covariance matrix.

    """

    def __init__(self, queue: cl.CommandQueue, n_components: int, whiten: bool = True,
                 datatype: Union[str, np.dtype] = 'float32'):
        """Initialize the BlastPCA object.

        Parameters
        ----------
        queue : pyopencl.CommandQueue
            The OpenCL queue to use.
        n_components : int, optional
            The number of components to keep. The default is None.
        datatype : str or numpy.dtype, optional
            The dtype of the data. The default is 'float32'.
        """
        self.n_components = n_components
        self.n_features = None
        self.whiten = whiten
        # check the datatype
        if isinstance(datatype, str):
            assert datatype in ['float64', 'float32']
        elif isinstance(datatype, np.dtype):
            assert datatype in [np.float64, np.float32]
        else:
            raise TypeError("dtype must be a string or a numpy dtype but got %s" % type(datatype))
        self.datatype = datatype
        self._eigenvectors = None
        self._components = None
        self._eigenvalues = None
        self._singular_values_ = None
        self._explained_variance_ = None
        self._explained_variance_ratio_ = None
        self._transformation_matrix = None
        self._inverse_transformation_matrix = None

        # set up OpenCL arrays
        self.queue = queue
        self.cl_data = None
        self.cl_T = None
        self.cl_T_inv = None
        self.cl_out = None

    def fit(self, data: np.ndarray):
        """Fit the PCA to the data.

        Parameters
        ----------
        data : numpy.ndarray
            The data to fit the PCA to.

        """
        # check the data dimensons
        if data.ndim != 2:
            raise ValueError("A must be a 2D array but got %d dimensions" % data.ndim)
        # set the data to contiguous and C order
        data = np.ascontiguousarray(data.astype(self.datatype))
        # get the number of samples and features
        n_entries, n_features = data.shape
        self.n_features = n_features
        # check the n_components is less than the number of features
        if self.n_components > n_features:
            raise ValueError(f"n_components, {self.n_components} given, exceeds features in A, {n_features}")
        # compute the covariance matrix
        cov = blast_covariance_matrix(self.queue, data, self.datatype)
        # compute the eigenvalues and eigenvectors
        self._eigenvalues, self._eigenvectors, info = dsyevd(cov.astype(np.float64))
        # recast the eigenvalues and eigenvectors to the correct datatype
        self._eigenvalues = self._eigenvalues.astype(self.datatype)
        self._eigenvectors = self._eigenvectors.astype(self.datatype)
        # if the info is not 0, then something went wrong
        if info != 0:
            raise RuntimeError("dsyevd (symmetric eigenvalue decomposition call to LAPACK) failed with info=%d" % info)
        # sort the eigenvalues and eigenvectors
        idx = np.argsort(self._eigenvalues)[::-1]
        self._eigenvalues = self._eigenvalues[idx]
        self._eigenvectors = self._eigenvectors[:, idx]
        # keep only the first n_components and make sure the arrays are contiguous in C order
        self._eigenvalues = np.ascontiguousarray(self._eigenvalues[:self.n_components]).astype(self.datatype)
        self._eigenvectors = np.ascontiguousarray(self._eigenvectors[:, :self.n_components]).astype(self.datatype)
        # calculate the explained variance
        self._explained_variance_ = self._eigenvalues / (n_entries - 1)
        self._explained_variance_ratio_ = self._explained_variance_ / np.sum(self._explained_variance_)
        # components is an alias for the un-whitened eigenvectors
        # calculate the transformation matrix
        if self.whiten:
            self._singular_values_ = np.sqrt(self._eigenvalues)[:self.n_components]
            self._transformation_matrix = self._eigenvectors * np.sqrt(n_entries - 1) / self._singular_values_[
                                                                                        np.newaxis, :]
            self._inverse_transformation_matrix = np.linalg.pinv(self._transformation_matrix)
        else:
            self._transformation_matrix = self._eigenvectors
            self._inverse_transformation_matrix = np.linalg.pinv(self._transformation_matrix)
        # make sure the transformation matrix is contiguous in C order
        self._transformation_matrix = np.ascontiguousarray(self._transformation_matrix)

        # set up the OpenCL arrays
        self.cl_data = Array(self.queue, data.shape, data.dtype)
        self.cl_T = Array(self.queue, self._transformation_matrix.shape, self._transformation_matrix.dtype)
        self.cl_T.set(self._transformation_matrix)
        self.cl_T_inv = Array(self.queue, self._inverse_transformation_matrix.shape,
                              self._inverse_transformation_matrix.dtype)
        self.cl_T_inv.set(self._inverse_transformation_matrix)

    def fit_transform(self, data: np.ndarray) -> Array:
        """Fit the PCA to the data and transform the data.

        Parameters
        ----------
        data : numpy.ndarray
            The data to fit the PCA to.

        Returns
        -------
        numpy.ndarray
            The transformed data.

        """
        self.fit(data)
        return self.transform(data)

    def transform(self, data: np.ndarray, mean_center: bool = False) -> Array:
        """Apply the dimensionality reduction on data.

        Parameters
        ----------
        data : numpy.ndarray
            The input data.
        mean_center : bool, optional
            Whether to mean center the data. The default is False.

        Returns
        -------
        numpy.ndarray
            The transformed data.

        """
        # assert the data is 2D and has the correct number of features
        if data.ndim != 2:
            raise ValueError("transforming data must be a 2D array but got %d dimensions" % data.ndim)
        if data.shape[1] != self.n_features:
            raise ValueError("transforming data must have %d features but got %d" % (self.n_features, data.shape[1]))
        # set the data to contiguous and C order
        data = np.ascontiguousarray(data.astype(self.datatype))
        # Mean centering
        if mean_center:
            data = data - np.mean(data, axis=0)
        # set up the input array if it is not already set up or is the wrong shape
        if (self.cl_data is None) or (self.cl_data.shape != data.shape):
            self.cl_data = Array(self.queue, (data.shape[0], self.n_features), data.dtype)
        # set up the output array if it is not already set up or is the wrong shape
        if (self.cl_out is None) or (self.cl_out.shape != (data.shape[0], self.n_components)):
            self.cl_out = Array(self.queue, (data.shape[0], self.n_components), self.datatype)
        # set the data
        self.cl_data.set(data)
        # Project the data onto the principal components
        blast_transform(self.queue,
                        self.cl_data,
                        self.cl_T,
                        self.cl_out,
                        data.shape[0],
                        self.n_features,
                        self.n_components)
        # return the transformed data OpenCL array
        return self.cl_out

    def inverse_transform(self, data: np.array):
        """Transform data back to its original space.

        Parameters
        ----------
        data : numpy.ndarray
            The transformed data.

        Returns
        -------
        numpy.ndarray
            The original data.

        """
        # assert the data is 2D and has the correct number of features
        if data.ndim != 2:
            raise ValueError("transforming data must be a 2D array but got %d dimensions" % data.ndim)
        if data.shape[1] != self.n_components:
            raise ValueError(
                "transforming data must have %d components but got %d" % (self.n_components, data.shape[1]))
        # set up the input array if it is not already set up
        if self.cl_data is None:
            self.cl_data = Array(self.queue, data.shape, data.dtype)
        # if a previous array was set up, check the shape against the new data
        if self.cl_data.shape != data.shape:
            self.cl_data = Array(self.queue, (data.shape[0], self.n_features), data.dtype)
        # set up the output array if it is not already set up
        if self.cl_out is None:
            self.cl_out = Array(self.queue, (data.shape[0], self.n_components), self.datatype)
        # set the data
        self.cl_data.set(data)
        # Project the data onto the principal components
        blast_transform(self.queue,
                        self.cl_data,
                        self.cl_T,
                        self.cl_out,
                        data.shape[0],
                        self.n_features,
                        self.n_components)
        # return the transformed data OpenCL array
        return self.cl_out

    @property
    def components_(self):
        """Return the principal components (eigenvectors).

        """
        return self._eigenvectors.T

    @property
    def explained_variance_(self):
        """Return the explained variance (eigenvalues).

        """
        return self._explained_variance_

    @property
    def singular_values_(self):
        """Return the singular values.

        """
        return self._singular_values_


class PCAIndexerBLAST(DIIndexer):
    """Dictionary indexing using BLAST.

    Summary
    -------

    BLAST is a tuned OpenCL implementation of BLAS (Basic Linear Algebra Subprograms).
    It is used for covariance matrix calculation and PCA transformation.

    """

    _allowed_dtypes: List[type] = [np.float32, np.float64]

    def __init__(self,
                 platform_id: int,
                 device_id: int,
                 datatype: Union[str, np.dtype, type],
                 normalize: bool = False,
                 zero_mean: bool = True,
                 space: str = "l2",
                 n_components: int = 500,
                 whiten: bool = True,
                 ):
        """Initialize the HNSWlib indexer.

        Parameters
        ----------
        datatype : numpy.dtype
            Data type of the dictionary and experimental patterns.
        platform_id : int, optional
            The platform id to use for the OpenCL context.
        device_id : int, optional
            The device id to use for the OpenCL context.
        normalize : bool, optional
            Whether to normalize the patterns before indexing. The default is True.
        zero_mean : bool, optional
            Whether to zero mean the patterns before indexing. The default is True.
        space : str, optional
            The metric to use for the nearest neighbors search. The default is "cosine".
        n_components : int, optional
            The number of components to keep in the PCA. The default is 700.
        whiten : bool, optional
            Whether to whiten the PCA. The default is True.
        """
        super().__init__(datatype=datatype)
        self._normalize = normalize
        self._zero_mean = zero_mean
        self.space = space
        self.n_components = n_components
        self.whiten = whiten
        self._pca = None
        self._knn_lookup_object = None
        # make the OpenCL context
        platforms = cl.get_platforms()
        self.ctx = cl.Context(devices=[platforms[platform_id].get_devices()[device_id]])
        self.queue = cl.CommandQueue(self.ctx)

    def __del__(self):
        """Clean up the OpenCL context.

        """
        self.queue.flush()
        self.queue.finish()
        self.ctx = None
        self.queue = None

    def prepare_dictionary(self, dictionary_patterns: np.ndarray):
        """Prepare the dictionary_patterns for indexing.

        """
        if self._normalize:
            dictionary_patterns = _normalize_patterns(dictionary_patterns)
        if self._zero_mean:
            dictionary_patterns = _zero_mean_patterns(dictionary_patterns)

        # make incremental PCA object
        self._pca = BlastPCA(self.queue, n_components=self.n_components, whiten=self.whiten, datatype=self.datatype)

        # make the cuml nearest neighbors object
        dictionary_pca_components = self._pca.fit_transform(dictionary_patterns)

        # fit the knn lookup object
        self._knn_lookup_object = BlastNN(self.queue, n_neighbors=self.keep_n, metric=self.space)
        self._knn_lookup_object.fit(dictionary_pca_components.astype(self.datatype))

    def prepare_experimental(self, experimental_patterns: np.ndarray) -> np.ndarray:
        """Prepare the experimental_patterns for indexing.

        """
        experimental_patterns = experimental_patterns.astype(self.datatype)
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
        indices, distances = self._knn_lookup_object.kneighbors(self._pca.transform(experimental_patterns))
        return indices, distances


def _normalize_patterns(patterns: np.ndarray) -> np.ndarray:
    """Normalize the patterns.
    """
    patterns /= np.linalg.norm(patterns, axis=1)[:, None]
    return patterns


def _zero_mean_patterns(patterns: np.ndarray) -> np.ndarray:
    """Zero mean the patterns.
    """
    patterns -= np.mean(patterns, axis=1)[:, None]
    return patterns
