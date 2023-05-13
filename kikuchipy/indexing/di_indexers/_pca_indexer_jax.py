from typing import Tuple
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
import functools
from kikuchipy.indexing.di_indexers._di_indexer import DIIndexer


@jit
def _eigendecomposition(matrix: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    Compute the eigenvector decomposition of a matrix.

    Parameters
    ----------
    matrix : jnp.ndarray
        The matrix to compute the eigenvector decomposition of.

    Returns
    -------
    jnp.ndarray
        The eigenvector decomposition of the matrix.

    """
    covar = jnp.cov(matrix, rowvar=False, ddof=0)
    v, w = jax.lax.linalg.eigh(covar)
    return v, w


@jit
def _transform(raw_data: jax.Array, transformation) -> jax.Array:
    """
    Transform the data using the transformation matrix.

    Parameters
    ----------
    raw_data : jnp.ndarray
        The raw data to transform.
    transformation : jnp.ndarray
        The transformation matrix.

    Returns
    -------
    jnp.ndarray
        The transformed data.

    """
    return jnp.dot(raw_data, transformation)


@functools.partial(jax.jit, static_argnames=["n_neighbors"])
def bot_min_k_exact(query: jax.Array, dictionary: jax.Array, n_neighbors: int) -> Tuple[jax.Array, jax.Array]:
    """
    Perform an exact nearest neighbor lookup.

    Parameters
    ----------
    query : jnp.ndarray
        The query to perform. Shape is (n_query, n_features).
    dictionary : jnp.ndarray
        The dictionary to perform the lookup in. Shape is (n_dictionary, n_features).
    n_neighbors : int
        The number of nearest neighbors to return.

    Returns
    -------
    jnp.ndarray
        The indices of the nearest neighbors.
    jnp.ndarray
        The distances to the nearest neighbors.

    """
    neg_l2_sq_pseudo_distances = (2.0 * jnp.dot(query, dictionary.T)) - jnp.sum(dictionary ** 2, axis=1)
    pseudo_distances, indices = jax.lax.top_k(neg_l2_sq_pseudo_distances, n_neighbors)
    distances = -pseudo_distances + jnp.sum(query ** 2, axis=1)[:, None]
    return distances, indices


@functools.partial(jax.jit, static_argnames=["k", "recall_target"])
def top_k_min_l2_ann(qy, db, k, recall_target):
    half_db_norms = jnp.linalg.norm(db, axis=1) ** 2 / 2
    dists = half_db_norms - jax.lax.dot(qy, db.transpose())
    return jax.lax.approx_min_k(dists, k=k, recall_target=recall_target)


class PCAKNNSearch:
    """
    Class for performing PCA KNN search using JAX. All nearest neighbor
    lookups are performed in the truncated PCA space.
    """

    def __init__(self,
                 n_components: int,
                 n_neighbors: int,
                 whiten: bool,
                 target_recall,
                 dtype: str):
        """
        Parameters
        ----------
        n_components : int, optional
            The number of components to keep in the PCA.
        n_neighbors : int, optional
            The number of nearest neighbors to return.
        whiten : bool
            Whether to whiten the patterns.
        dtype : str, optional
            The dtype to use for the patterns.

        Returns
        -------
        None.

        """
        self.target_recall = target_recall
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.whiten = whiten
        self.dtype = dtype
        self._pca_dictionary = None
        self._transformation = None
        self._inverse_transformation = None

    def fit(self, dictionary: jax.Array):
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
        v, w = _eigendecomposition(dictionary)
        self._transformation = v[:, -self.n_components:]
        self._inverse_transformation = jnp.linalg.pinv(self._transformation)
        self._pca_dictionary = _transform(dictionary, self._transformation)

    def query(self, query: jnp.ndarray):
        """
        Query the dictionary.

        Parameters
        ----------
        query : jnp.ndarray
            The query to perform.

        Returns
        -------
        jnp.ndarray
            The indices of the nearest neighbors.
        jnp.ndarray
            The distances to the nearest neighbors.

        """
        query = _transform(query, self._transformation)
        if self.target_recall < 1.0:
            distances, indices = top_k_min_l2_ann(query, self._pca_dictionary, self.n_neighbors, self.target_recall)
        else:
            distances, indices = bot_min_k_exact(query, self._pca_dictionary, self.n_neighbors)
        return indices, distances


class PCAIndexerJAX(DIIndexer):
    """
    Class for performing PCA KNN search using JAX. All nearest neighbor
    lookups are performed in the truncated PCA space. This class is a
    wrapper around the PCAKNNSearch class that allows for the use of
    """

    def __init__(self,
                 zero_mean: bool = True,
                 unit_var: bool = True,
                 n_components: int = 750,
                 n_neighbors: int = 1,
                 whiten: bool = False,
                 target_recall: float = 1.0,
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
            The dtype to use for matching. The default is 'float32'.
        """
        super().__init__(datatype=datatype)
        self._pca_knn_search = PCAKNNSearch(n_components=n_components,
                                            n_neighbors=n_neighbors,
                                            whiten=whiten,
                                            target_recall=target_recall,
                                            dtype=datatype)
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
        self._pca_knn_search.fit(jnp.array(dictionary))

    def query(self, query: np.ndarray):
        """
        Query the dictionary.

        Parameters
        ----------
        query : jnp.ndarray
            The query to perform. Must be of shape (n_samples, n_features).

        Returns
        -------
        jnp.ndarray
            The indices of the nearest neighbors.
        jnp.ndarray
            The distances to the nearest neighbors.

        """
        query = jnp.array(query)
        if self._zero_mean:
            query = query - jnp.mean(query, axis=1, keepdims=True)
        if self._unit_var:
            query = query / jnp.std(query, axis=1, keepdims=True)
        return self._pca_knn_search.query(jnp.array(query))
