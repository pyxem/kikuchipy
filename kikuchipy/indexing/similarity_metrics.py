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
from typing import Callable, Dict, Tuple, Union

import dask.array as da
import numpy as np

from kikuchipy.indexing._util import _get_nav_shape, _get_sig_shape


class MetricScope(Enum):
    """Describes the input parameters for a similarity metric. See
    :func:`make_similarity_metric`."""

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
        Describes how `metric_func`'s input parameters are structured,
        by default `MetricScope.MANY_TO_MANY`.
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
    metric_class : Union[SimilarityMetric, FlatSimilarityMetric]
        A callable class instance computing a similarity matrix with
        signature `metric(patterns, templates)`.

    Notes
    -----
    The metric function must take the arrays `patterns` and `templates`
    as arguments, in that order. The scope and whether the metric is
    flat defines the intended data shapes. In the following table,
    (m,n) and (x,y) correspond to navigation and signal shape,
    respectively:

    ============ ========= ========= ======== ======== ========= =======
    MetricScope  flat = False                 flat = True
    ------------ ---------------------------- --------------------------
    \-           patterns  templates returns  patterns templates returns
    ============ ========= ========= ======== ======== ========= =======
    MANY_TO_MANY (n,m,y,x) (N,y,x)   (n,m,N)  (nm,yx)  (N,yx)    (nm,N)
    ONE_TO_MANY  (y,x)     (N,y,x)   (N,)     (yx,)    (N,yx)    (N,)
    MANY_TO_ONE  (n,m,y,x) (y,x)     (n,m)    (nm,yx)  (yx,)     (nm,)
    ONE_TO_ONE   (y,x)     (y,x)     (1,)     (yx,)    (yx,)     (1,)
    ============ ========= ========= ======== ======== ========= =======
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
    ):
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

    def _measure(
        self,
        patterns: Union[np.ndarray, da.Array],
        templates: Union[np.ndarray, da.Array],
    ) -> Union[np.ndarray, da.Array]:
        return self._metric_func(patterns, templates)

    def _expand_dims_to_match_scope(
        self,
        p: Union[np.ndarray, da.Array],
        t: Union[np.ndarray, da.Array],
    ) -> Tuple[Union[np.ndarray, da.Array], Union[np.ndarray, da.Array]]:
        p_scope_ndim, t_scope_ndim = self._SCOPE_TO_P_T_NDIM[self.scope]
        p = p[(np.newaxis,) * (p_scope_ndim - p.ndim)]
        t = t[(np.newaxis,) * (t_scope_ndim - t.ndim)]
        return p, t

    def _is_compatible(
        self, p: Union[np.ndarray, da.Array], t: Union[np.ndarray, da.Array]
    ) -> bool:
        p_ndim, t_ndim = p.ndim, t.ndim
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
        nav_shape = _get_nav_shape(patterns)
        sig_shape = _get_sig_shape(patterns)
        nav_flat_size, sig_flat_size = np.prod(nav_shape), np.prod(sig_shape)
        patterns = patterns.reshape((nav_flat_size, sig_flat_size))
        templates = templates.reshape((-1, sig_flat_size))
        return self._metric_func(patterns, templates)


def _expand_dims_to_many_to_many(
    p: Union[np.ndarray, da.Array],
    t: Union[np.ndarray, da.Array],
    flat: bool,
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
    p : Union[np.ndarray, da.Array]
        Patterns.
    t : Union[np.ndarray, da.Array]
        Templates.
    flat : bool, optional
        Whether `p` and `t` are flattened, by default False.

    Returns
    -------
    p, t
        Tuple of `p` and `t` with their mean subtracted.
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
    p : Union[np.ndarray, da.Array]
        Patterns.
    t : Union[np.ndarray, da.Array]
        Templates.
    flat : bool, optional
        Whether `p` and `t` are flattened, by default False.

    Returns
    -------
    p, t
        Tuple of `p` and `t` divided respectively by their L2-norms.
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


SIMILARITY_METRICS: Dict[str, Callable] = {}


def _zncc_einsum(
    patterns: Union[da.Array, np.ndarray],
    templates: Union[da.Array, np.ndarray],
) -> Union[np.ndarray, da.Array]:
    """Compute (lazily) ZNCC between patterns and templates.

    Parameters
    ----------
    patterns : Union[da.Array, np.ndarray]
        [description]
    templates : Union[da.Array, np.ndarray]
        [description]

    Returns
    -------
    Union[da.Array, np.ndarray]
        [description]
    """
    patterns, templates = _zero_mean(patterns, templates)
    patterns, templates = _normalize(patterns, templates)

    zncc = da.einsum("ijkl,mkl->ijm", patterns, templates, optimize=True)

    # Alternative with equivalent results:
    # zncc = da.tensordot(p_da, t_da, axes=([2,3], [1, 2]))
    return zncc


SIMILARITY_METRICS["zncc"] = make_similarity_metric(
    _zncc_einsum,
    scope=MetricScope.MANY_TO_MANY,
    make_compatible_to_lower_scopes=True,
)


def _ndp_einsum(
    patterns: Union[da.Array, np.ndarray],
    templates: Union[da.Array, np.ndarray],
) -> Union[np.ndarray, da.Array]:
    patterns, templates = _normalize(patterns, templates)
    ndp = da.einsum("ijkl,mkl->ijm", patterns, templates, optimize=True)
    return ndp


SIMILARITY_METRICS["ndp"] = make_similarity_metric(
    _ndp_einsum,
    scope=MetricScope.MANY_TO_MANY,
    make_compatible_to_lower_scopes=True,
)
