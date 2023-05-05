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
import numpy as np

# try to import pyopencl and notify user if not available
try:
    import pyopencl as cl
    from pyopencl import array as cl_array
except ImportError:
    cl = None
    print(
        "pyopencl is not installed. "
        "Will not be available for use."
    )

from kikuchipy.indexing.di_indexers._di_indexer import DIIndexer

KERNEL_CODE_L2 = """
__kernel void nearest_neighbor(__global const float *data,
                               __global const float *query,
                               const uint n_data,
                               const uint n_query,
                               const uint k,
                               const uint n_features,
                               __global float *distances,
                               __global int *indices)
{
    const uint gid = get_global_id(0);

    const uint query_idx = gid * n_features;
    float min_dist = INFINITY;
    int min_idx = -1;

    for (uint j = 0; j < n_data; j++) {
        const uint data_idx = j * n_features;
        float dist = 0.0f;

        for (uint l = 0; l < n_features; l += 4) {
            const float4 a = vload4(l, &data[data_idx]);
            const float4 b = vload4(l, &query[query_idx]);
            dist = fma(a.x - b.x, a.x - b.x, dist);
            dist = fma(a.y - b.y, a.y - b.y, dist);
            dist = fma(a.z - b.z, a.z - b.z, dist);
            dist = fma(a.w - b.w, a.w - b.w, dist);
        }

        if (dist < min_dist) {
            min_dist = dist;
            min_idx = j;
        }
    }

    distances[gid * k] = min_dist;
    indices[gid * k] = min_idx;
}
"""

KERNEL_CODE_L2 = """
__kernel void nearest_neighbor(__global const float *data,
                               __global const float *query,
                               const uint n_data,
                               const uint n_query,
                               const uint k,
                               const uint n_features,
                               __global float *distances,
                               __global int *indices)
{
    const uint gid = get_global_id(0);

    const uint query_idx = gid * n_features;
    float min_dist = INFINITY;
    int min_idx = -1;

    for (uint j = 0; j < n_data; j++) {
        const uint data_idx = j * n_features;
        float dist = 0.0f;

        for (uint l = 0; l < n_features; l += 16) {
            const float4 a1 = vload4(l, &data[data_idx]);
            const float4 a2 = vload4(l+4, &data[data_idx]);
            const float4 a3 = vload4(l+8, &data[data_idx]);
            const float4 a4 = vload4(l+12, &data[data_idx]);
            const float4 b1 = vload4(l, &query[query_idx]);
            const float4 b2 = vload4(l+4, &query[query_idx]);
            const float4 b3 = vload4(l+8, &query[query_idx]);
            const float4 b4 = vload4(l+12, &query[query_idx]);
            const float4 diff1 = a1 - b1;
            const float4 diff2 = a2 - b2;
            const float4 diff3 = a3 - b3;
            const float4 diff4 = a4 - b4;
            dist = fma(diff1.x, diff1.x, dist);
            dist = fma(diff1.y, diff1.y, dist);
            dist = fma(diff1.z, diff1.z, dist);
            dist = fma(diff1.w, diff1.w, dist);
            dist = fma(diff2.x, diff2.x, dist);
            dist = fma(diff2.y, diff2.y, dist);
            dist = fma(diff2.z, diff2.z, dist);
            dist = fma(diff2.w, diff2.w, dist);
            dist = fma(diff3.x, diff3.x, dist);
            dist = fma(diff3.y, diff3.y, dist);
            dist = fma(diff3.z, diff3.z, dist);
            dist = fma(diff3.w, diff3.w, dist);
            dist = fma(diff4.x, diff4.x, dist);
            dist = fma(diff4.y, diff4.y, dist);
            dist = fma(diff4.z, diff4.z, dist);
            dist = fma(diff4.w, diff4.w, dist);
        }

        if (dist < min_dist) {
            min_dist = dist;
            min_idx = j;
        }
    }

    distances[gid * k] = min_dist;
    indices[gid * k] = min_idx;
}
"""

class GPUNearestNeighbors:
    """
    Class that finds the k nearest neighbors on the GPU and keeps the training data on the GPU for faster predictions.
    """

    def __init__(self, k: int = 1, distance_metric: str = "l2", batch_size_factor: int = 4):
        """
        Initializes the GPUNearestNeighbors object.

        Parameters
        ----------
        k : int, optional (default=1)
            The number of neighbors to find.
        distance_metric : str, optional (default="l2")
            The distance metric to use: "l2" for L2 distance and "cosine" for cosine distance.
        """
        self.k = k
        self.distance_metric = distance_metric
        self.batch_size_factor = batch_size_factor

        self.platform = cl.get_platforms()[0]
        self.device = self.platform.get_devices()[0]

        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)

        self.nearest_neighbor_kernel = None
        self.data_buf = None
        self.data_len = None
        self.n_features = None

        if self.distance_metric == "l2":
            self.kernel_code = KERNEL_CODE_L2
        elif self.distance_metric == "cosine":
            self.kernel_code = KERNEL_CODE_L2

    def fit(self, data):
        data = data.astype(np.float32)
        self.data_len = data.shape[0]
        self.n_features = data.shape[1]
        self.data_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)

        self.nearest_neighbor_kernel = cl.Program(self.context, self.kernel_code).build().nearest_neighbor

    def predict(self, query):
        query = query.astype(np.float32)
        query_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=query)

        distances_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY,
                                  query.shape[0] * self.k * np.dtype(np.float32).itemsize)

        indices_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY,
                                query.shape[0] * self.k * np.dtype(np.int32).itemsize)

        if self.distance_metric == "l2":
            distance_metric_val = 0
        elif self.distance_metric == "cosine":
            distance_metric_val = 1
        else:
            raise ValueError("Invalid distance metric specified. Must be 'l2' or 'cosine'.")

        self.nearest_neighbor_kernel.set_args(self.data_buf, query_buf, np.uint32(self.data_len),
                                              np.uint32(query.shape[0]), np.uint32(self.k),
                                              np.uint32(self.n_features), distances_buf, indices_buf)
        cl.enqueue_nd_range_kernel(self.queue, self.nearest_neighbor_kernel, (query.shape[0],), None)

        distances = np.empty((query.shape[0], self.k), dtype=np.float32)
        indices = np.empty((query.shape[0], self.k), dtype=np.int32)

        cl.enqueue_copy(self.queue, distances, distances_buf)
        cl.enqueue_copy(self.queue, indices, indices_buf)

        query_buf.release()
        distances_buf.release()
        indices_buf.release()

        return distances, indices


class PyOpenCLIndexer(DIIndexer):
    """Dictionary indexing using cuml.

    Summary
    -------
    This class implements a function to match experimental and simulated
    EBSD patterns in a dictionary using pyopencl.

    """

    _allowed_dtypes: List[type] = [np.float32, np.float64]

    def __init__(self,
                 dtype: Union[str, np.dtype, type],
                 normalize: bool = False,
                 zero_mean: bool = True,
                 space: str = "l2",
                 ):
        """
        Parameters
        ----------
        dtype : numpy.dtype
            Data type of the dictionary and experimental patterns.
        normalize : bool, optional (default=False)
            Whether to normalize the patterns before indexing.
        zero_mean : bool, optional (default=True)
            Whether to zero mean the patterns before indexing.


        """
        super().__init__(dtype=dtype)
        self._normalize = normalize
        self._zero_mean = zero_mean
        self._space = space
        self._knn_search = None

    def prepare_dictionary(self, dictionary_patterns: np.ndarray):
        """Prepare the dictionary_patterns for indexing.

        """
        if self._normalize:
            dictionary_patterns = _normalize_patterns(dictionary_patterns)
        if self._zero_mean:
            dictionary_patterns = _zero_mean_patterns(dictionary_patterns)
        self._knn_search = None
        self._knn_search = GPUNearestNeighbors(k=self.keep_n, distance_metric=self._space)
        self._knn_search.fit(dictionary_patterns)

    def prepare_experimental(self, experimental_patterns: np.ndarray) -> np.ndarray:
        """Prepare the experimental_patterns for indexing.

        """
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
        distances, indices = self._knn_search.predict(experimental_patterns)
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
