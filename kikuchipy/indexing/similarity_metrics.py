# -*- coding: utf-8 -*-
# Copyright 2019-2020 The kikuchipy developers
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

"""Similarity metrics for comparing gray-tone images."""

from enum import Enum
from typing import Callable, Tuple, Union

import dask.array as da
import numpy as np


class MetricScope(Enum):
    """Describes the input parameters for a similarity metric. See
    :func:`make_similarity_metric`.
    """

    MANY_TO_MANY = "many_to_many"
    ONE_TO_MANY = "one_to_many"
    ONE_TO_ONE = "one_to_one"
    MANY_TO_ONE = "many_to_one"


def make_similarity_metric(
    metric_func: Callable,
    greater_is_better: bool = True,
    scope: Enum = MetricScope.MANY_TO_MANY,
    flat: bool = False,
    make_compatible_to_lower_scopes: bool = False,
    dtype_out: np.dtype = np.float32,
):
    """Make a similarity metric for comparing gray-tone images of equal
    size.

    This factory function wraps metric functions for use in
    `template_match`, which again is used by
    :class:`~kikuchipy.indexing.StaticDictionary` and
    :class:`~kikuchipy.indexing.DynamicDictionary`.

    Parameters
    ----------
    metric_func
        Metric function with signature
        `metric_func(patterns, templates)`, which computes the
        similarity or a distance matrix between (experimental)
        pattern(s) and (simulated) template(s).
    greater_is_better
        Whether greater values correspond to more similar images, by
        default True. Used for choosing `n_largest` metric results in
        `template_match`.
    scope
        Describes the shapes of the `metric_func`'s input parameters
        `patterns` and `templates`, by default
        `MetricScope.MANY_TO_MANY`.
    flat
        Whether patterns and templates are to be flattened before sent
        to `metric_func` when the similarity metric is called, by
        default False.
    make_compatible_to_lower_scopes
        Whether to reshape patterns and templates by adding single
        dimensions to match the given scope, by default False.
    dtype_out
        The data type used and returned by the metric, by default
        :class:`np.float32`.

    Returns
    -------
    metric_class : SimilarityMetric or FlatSimilarityMetric
        A callable class instance computing a similarity matrix with
        signature `metric_func(patterns, templates)`.

    Notes
    -----
    The metric function must take the arrays `patterns` and `templates`
    as arguments, in that order. The scope and whether the metric is
    flat defines the intended data shapes. In the following table,
    (ny, nx) and (sy, sx) correspond to the navigation and signal
    shapes (row, column), respectively, with N number of templates:

    ============ ============= ========= ========= ============= ========= =========
    MetricScope  flat = False                      flat = True
    ------------ --------------------------------- ---------------------------------
    \-           patterns      templates returns   patterns      templates returns
    ============ ============= ========= ========= ============= ========= =========
    MANY_TO_MANY (ny,nx,sy,sx) (N,sy,sx) (ny,nx,N) (ny*nx,sy*sx) (N,sy*sx) (ny*nx,N)
    ONE_TO_MANY  (sy,sx)       (N,sy,sx) (N,)      (sy*sx,)      (N,sy*sx) (N,)
    MANY_TO_ONE  (ny,nx,sy,sx) (sy,sx)   (ny,nx)   (ny*nx,sy*sx) (sy*sx,)  (ny*nx)
    ONE_TO_ONE   (sy,sx)       (sy,sx)   (1,)      (sy*sx,)      (sy*sx,)  (1,)
    ============ ============= ========= ========= ============= ========= =========
    """
    sign = 1 if greater_is_better else -1
    if flat:
        return FlatSimilarityMetric(
            metric_func,
            sign,
            scope,
            flat,
            make_compatible_to_lower_scopes,
            dtype_out,
        )
    else:
        return SimilarityMetric(
            metric_func,
            sign,
            scope,
            flat,
            make_compatible_to_lower_scopes,
            dtype_out,
        )


class SimilarityMetric:
    """Similarity metric between 2D gray-tone images."""

    # See table in docstring of `make_similarity_metric`
    # TODO: Support for 1D navigation shape
    _P_T_NDIM_TO_SCOPE = {
        (4, 3): MetricScope.MANY_TO_MANY,
        (2, 3): MetricScope.ONE_TO_MANY,
        (4, 2): MetricScope.MANY_TO_ONE,
        (2, 2): MetricScope.ONE_TO_ONE,
    }

    _SCOPE_TO_P_T_NDIM = {
        MetricScope.MANY_TO_MANY: (4, 3),
        MetricScope.ONE_TO_MANY: (2, 3),
        MetricScope.MANY_TO_ONE: (4, 2),
        MetricScope.ONE_TO_ONE: (2, 2),
    }

    _SCOPE_TO_LOWER_SCOPES = {
        MetricScope.MANY_TO_MANY: (
            MetricScope.MANY_TO_ONE,
            MetricScope.ONE_TO_MANY,
            MetricScope.ONE_TO_ONE,
        ),
        MetricScope.ONE_TO_MANY: (
            MetricScope.ONE_TO_MANY,
            MetricScope.ONE_TO_ONE,
        ),
        MetricScope.ONE_TO_ONE: (),
        MetricScope.MANY_TO_ONE: (
            MetricScope.MANY_TO_ONE,
            MetricScope.ONE_TO_ONE,
        ),
    }

    def __init__(
        self,
        metric_func: Callable,
        sign: int,
        scope: Enum,
        flat: bool,
        make_compatible_to_lower_scopes: bool,
        dtype_out: np.dtype = np.float32,
    ):
        self._metric_func = metric_func
        self._make_compatible_to_lower_scopes = make_compatible_to_lower_scopes
        self._dtype_out = dtype_out
        self.sign = sign
        self.flat = flat
        self.scope = scope

    def __call__(
        self,
        patterns: Union[np.ndarray, da.Array],
        templates: Union[np.ndarray, da.Array],
    ) -> Union[np.ndarray, da.Array]:
        dtype = self._dtype_out
        patterns = patterns.astype(dtype)
        templates = templates.astype(dtype)
        if isinstance(patterns, da.Array):
            patterns = patterns.rechunk()
        if isinstance(templates, da.Array):
            templates = templates.rechunk()
        if self._make_compatible_to_lower_scopes:
            patterns, templates = self._expand_dims_to_match_scope(
                patterns, templates
            )
        return self._measure(patterns, templates).squeeze()

    def __repr__(self):
        return (
            f"{self.__class__.__name__} {self._metric_func.__name__}, "
            f"scope: {self.scope.value}"
        )

    def _measure(
        self,
        patterns: Union[np.ndarray, da.Array],
        templates: Union[np.ndarray, da.Array],
    ) -> Union[np.ndarray, da.Array]:
        """Measure the similarities and return the results."""
        return self._metric_func(patterns, templates)

    def _expand_dims_to_match_scope(
        self, p: Union[np.ndarray, da.Array], t: Union[np.ndarray, da.Array],
    ) -> Tuple[Union[np.ndarray, da.Array], Union[np.ndarray, da.Array]]:
        """Return patterns and templates with added axes corresponding
        to the scope of the metric.
        """
        p_scope_ndim, t_scope_ndim = self._SCOPE_TO_P_T_NDIM[self.scope]
        p = p[(np.newaxis,) * (p_scope_ndim - p.ndim)]
        t = t[(np.newaxis,) * (t_scope_ndim - t.ndim)]
        return p, t

    def _is_compatible(self, p_ndim: int, t_ndim: int) -> bool:
        """Return whether shapes of patterns and templates are
        compatible with the metric's scope.
        """
        if self.flat:
            p_ndim = p_ndim // 2  # 4 -> 2 or 2 -> 1
            t_ndim -= 1
        inferred_scope = self._P_T_NDIM_TO_SCOPE.get((p_ndim, t_ndim), False)
        if not inferred_scope:
            return False
        if inferred_scope == self.scope:
            return True
        else:
            if self._make_compatible_to_lower_scopes:
                return inferred_scope in self._SCOPE_TO_LOWER_SCOPES[self.scope]
            else:
                return False


class FlatSimilarityMetric(SimilarityMetric):
    """Similarity metric between 2D gray-tone images where the images
    are flattened before sent to `metric_func`.
    """

    # See table in docstring of `make_similarity_metric`
    _P_T_NDIM_TO_SCOPE = {
        (2, 2): MetricScope.MANY_TO_MANY,
        (1, 2): MetricScope.ONE_TO_MANY,
        (3, 1): MetricScope.MANY_TO_ONE,
        (1, 1): MetricScope.ONE_TO_ONE,
    }

    _SCOPE_TO_P_T_NDIM = {
        MetricScope.MANY_TO_MANY: (2, 2),
        MetricScope.ONE_TO_MANY: (1, 2),
        MetricScope.MANY_TO_ONE: (3, 1),
        MetricScope.ONE_TO_ONE: (1, 1),
    }

    _PARTIAL_SCOPE_TO_NDIM = {
        "MANY_": 2,  # Many patterns
        "_MANY": 2,  # Many templates
        "ONE_": 1,  # One pattern
        "_ONE": 1,  # One template
    }

    def _measure(
        self,
        patterns: Union[np.ndarray, da.Array],
        templates: Union[np.ndarray, da.Array],
    ) -> Union[np.ndarray, da.Array]:
        """Flatten images, measure the similarities and return the
        results.
        """
        nav_shape = _get_nav_shape(patterns)
        sig_shape = _get_sig_shape(patterns)
        nav_flat_size, sig_flat_size = np.prod(nav_shape), np.prod(sig_shape)
        patterns = patterns.reshape((nav_flat_size, sig_flat_size))
        templates = templates.reshape((-1, sig_flat_size))
        return self._metric_func(patterns, templates)


def _get_nav_shape(p):
    return {2: (), 3: (p.shape[0],), 4: (p.shape[:2])}[p.ndim]


def _get_sig_shape(p):
    return p.shape[-2:]


def _get_number_of_templates(t):
    return t.shape[0] if t.ndim == 3 else 1


def _expand_dims_to_many_to_many(
    p: Union[np.ndarray, da.Array], t: Union[np.ndarray, da.Array], flat: bool,
) -> Tuple[Union[np.ndarray, da.Array], Union[np.ndarray, da.Array]]:
    """Expand the dims of patterns and templates to match
    `MetricScope.MANY_TO_MANY`.

    Parameters
    ----------
    p
        Patterns.
    t
        Templates.
    flat
        Whether `p` and `t` are flattened.

    Returns
    -------
    p
        Patterns with their dimension expanded to match
        `MetricScope.MANY_TO_MANY`.
    t
        Templates with their dimension expanded to match
        `MetricScope.MANY_TO_MANY`.
    """
    metric_cls = FlatSimilarityMetric if flat else SimilarityMetric
    p_scope_ndim, t_scope_ndim = metric_cls._SCOPE_TO_P_T_NDIM[
        MetricScope.MANY_TO_MANY
    ]
    p = p[(np.newaxis,) * (p_scope_ndim - p.ndim)]
    t = t[(np.newaxis,) * (t_scope_ndim - t.ndim)]
    return p, t


def _zero_mean(
    p: Union[np.ndarray, da.Array],
    t: Union[np.ndarray, da.Array],
    flat: bool = False,
) -> Tuple[Union[np.ndarray, da.Array], Union[np.ndarray, da.Array]]:
    """Subtract the mean from patterns and templates of any scope.

    Parameters
    ----------
    p : np.ndarray or da.Array
        Patterns.
    t : np.ndarray or da.Array
        Templates.
    flat : bool, optional
        Whether `p` and `t` are flattened, by default False.

    Returns
    -------
    p
        Patterns with their mean subtracted.
    t
        Templates with their mean subtracted.
    """
    squeeze = 1 not in p.shape + t.shape
    p, t = _expand_dims_to_many_to_many(p, t, flat)
    p_mean_axis = 1 if flat else (2, 3)
    t_mean_axis = 1 if flat else (1, 2)
    p -= p.mean(axis=p_mean_axis, keepdims=True)
    t -= t.mean(axis=t_mean_axis, keepdims=True)

    if squeeze:
        return p.squeeze(), t.squeeze()
    else:
        return p, t


def _normalize(
    p: Union[np.ndarray, da.Array],
    t: Union[np.ndarray, da.Array],
    flat: bool = False,
) -> Tuple[Union[np.ndarray, da.Array], Union[np.ndarray, da.Array]]:
    """Normalize patterns and templates of any scope.

    Parameters
    ----------
    p : np.ndarray or da.Array
        Patterns.
    t : np.ndarray or da.Array
        Templates.
    flat : bool, optional
        Whether `p` and `t` are flattened, by default False.

    Returns
    -------
    p
        Patterns divided by their L2 norms.
    t
        Templates divided by their L2 norms.
    """
    squeeze = 1 not in p.shape + t.shape
    p, t = _expand_dims_to_many_to_many(p, t, flat)
    p_sum_axis = 1 if flat else (2, 3)
    t_sum_axis = 1 if flat else (1, 2)
    p /= (p ** 2).sum(axis=p_sum_axis, keepdims=True) ** 0.5
    t /= (t ** 2).sum(axis=t_sum_axis, keepdims=True) ** 0.5

    if squeeze:
        return p.squeeze(), t.squeeze()
    else:
        return p, t


def _zncc_einsum(
    patterns: Union[da.Array, np.ndarray],
    templates: Union[da.Array, np.ndarray],
) -> Union[np.ndarray, da.Array]:
    """Compute (lazily) the zero-mean normalized cross-correlation
    coefficient between patterns and templates.

    Parameters
    ----------
    patterns
        Experimental patterns to compare to the templates.
    templates
        Templates, typically simulated patterns.

    Returns
    -------
    zncc
        Correlation coefficients in range [-1, 1] for all comparisons,
        as :class:`np.ndarray` if both `patterns` and `templates` are
        :class:`np.ndarray`, else :class:`da.Array`.

    Notes
    -----
    Equivalent results are obtained with :func:`dask.Array.tensordot`
    with the `axes` argument `axes=([2, 3], [1, 2]))`.
    """
    patterns, templates = _zero_mean(patterns, templates)
    patterns, templates = _normalize(patterns, templates)
    zncc = da.einsum("ijkl,mkl->ijm", patterns, templates, optimize=True)
    if isinstance(patterns, np.ndarray) and isinstance(templates, np.ndarray):
        return zncc.compute()
    else:
        return zncc


def _ndp_einsum(
    patterns: Union[da.Array, np.ndarray],
    templates: Union[da.Array, np.ndarray],
) -> Union[np.ndarray, da.Array]:
    """Compute the normalized dot product between patterns and
    templates.

    Parameters
    ----------
    patterns
        Experimental patterns to compare to the templates.
    templates
        Templates, typically simulated patterns.

    Returns
    -------
    ndp
        Normalized dot products in range [0, 1] for all comparisons,
        as :class:`np.ndarray` if both `patterns` and `templates` are
        :class:`np.ndarray`, else :class:`da.Array`.
    """
    patterns, templates = _normalize(patterns, templates)
    ndp = da.einsum("ijkl,mkl->ijm", patterns, templates, optimize=True)
    if isinstance(patterns, np.ndarray) and isinstance(templates, np.ndarray):
        return ndp.compute()
    else:
        return ndp


SIMILARITY_METRICS = {
    "zncc": make_similarity_metric(
        metric_func=_zncc_einsum,
        scope=MetricScope.MANY_TO_MANY,
        make_compatible_to_lower_scopes=True,
    ),
    "ndp": make_similarity_metric(
        metric_func=_ndp_einsum,
        scope=MetricScope.MANY_TO_MANY,
        make_compatible_to_lower_scopes=True,
    ),
}
