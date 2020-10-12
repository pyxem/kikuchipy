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

from typing import Union, Dict, Callable
import dask.array as da
import numpy as np

from enum import Enum
from kikuchipy.indexation._util import _get_nav_shape, _get_sig_shape


class MetricScope(Enum):
    """Describes the input parameters for a similarity metric. See `make_similarity_metric`"""

    MANY_TO_MANY = "many_to_many"
    ONE_TO_MANY = "one_to_many"
    ONE_TO_ONE = "one_to_one"
    MANY_TO_ONE = "many_to_one"
    ANY_TO_ANY = "*_to_*"
    ANY_TO_MANY = "*_to_many"
    MANY_TO_ANY = "many_to_*"
    ONE_TO_ANY = "one_to_*"
    ANY_TO_ONE = "*_to_one"


def make_similarity_metric(
    metric_func: Callable,
    greater_is_better=True,
    scope=MetricScope.MANY_TO_MANY,
    flat=False,
    make_compatible_to_lower_scopes=False,
    dtype_out=np.float32,
):
    """Make a similarity metric for comparing gray-tone images of equal size.

    This factory function wraps metric functions for use in `template_match`,
    which again is used by :class:`~kikuchipy.indexation.StaticDictionary` and :class:`~kikuchipy.indexation.DynamicDictionary`.

    The metric function must take the arrays; patterns and templates as arguments.
    The scope and wheter the metric is flat defines the intended data shapes:
    +--------------+-----------------------+----------------------+
    | MetricScope  | flat = False          | flat = True          |
    +==============+===========+===========+==========+===========+
    |              | patterns    templates | patterns   templates |
    |              |        returns        |       returns        |
    +--------------+-----------+-----------+----------+-----------+
    | MANY_TO_MANY | (n,m,y,x)    (N,y,x)  | (nm,yx)      (N,yx)  |
    |              |        (N,n,m)        |        (N,nm)        |
    +--------------+-----------+-----------+----------+-----------+
    | ONE_TO_MANY  |   (y,x)      (N,y,x)  |   (yx,)      (N,yx)  |
    |              |         (N,)          |        (N,)          |
    +--------------+-----------+-----------+----------+-----------+
    | MANY_TO_ONE  | (n,m,y,x)     (y,x)   | (nm,yx)      (yx,)   |
    |              |         (n,m)         |        (nm,)         |
    +--------------+-----------+-----------+----------+-----------+
    | ONE_TO_ONE   |   (y,x)       (y,x)   |   (yx,)      (yx,)   |
    |              |         scalar        |        scalar        |
    +--------------+-----------+-----------+----------+-----------+
    where (m,n) and (x,y) correspond to navigation and signal shape in HyperSpy, respectively.

    Parameters
    ----------
    metric_func : Callable
        Metric function with signature
        `metric_func(patterns,templates)`

        Computes similarity or distance matrix
        between (experimental) pattern(s) and (simulated) template(s).

    greater_is_better : bool, optional
        Whether greater values correspond to more similar images, by default True.
        Used for choosing `n_largest` metric results in `template_match`.
    scope : MetricScope, optional
        Describes how `metric_func`'s input parameters is structured,
        by default MetricScope.MANY_TO_MANY.
    flat : bool, optional
        Whether patterns and templates are flattened before sent to `metric_func`
        when the similarity metric is called, by default False.
    make_compatible_to_lower_scopes : bool, optional
        Whether to pad patterns and templates with single dimensions
        to match given scope, by default False.
    dtype_out : data-type, optional
        The data type used and returned by the metric, by default np.float32

    Returns
    -------
    Union[SimilarityMetric, FlatSimilarityMetric]
        A callable class instance computing a similarity matrix with signature
        `metric(patterns,templates)`.
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

    _P_T_NDIM_TO_SCOPE = {
        (4, 3): MetricScope.MANY_TO_MANY,
        (2, 3): MetricScope.ONE_TO_MANY,
        (4, 2): MetricScope.MANY_TO_ONE,
        (2, 2): MetricScope.ONE_TO_ONE,
    }

    _PARTIAL_SCOPE_TO_NDIM = {
        "MANY_": 4,  #  Many patterns
        "_MANY": 3,  # Many templates
        "ONE_": 2,  # One pattern
        "_ONE": 2,  # One template￼
    }

    _SCOPE_INCLUDING_LOWER_SCOPES = {
        MetricScope.MANY_TO_MANY: MetricScope.ANY_TO_ANY,
        MetricScope.ONE_TO_MANY: MetricScope.ONE_TO_ANY,
        MetricScope.ONE_TO_ONE: MetricScope.ONE_TO_ONE,
        MetricScope.MANY_TO_ONE: MetricScope.ANY_TO_ONE,
    }

    def __init__(
        self,
        metric_func,
        sign,
        scope,
        flat,
        make_compatible_to_lower_scopes,
        dtype_out=np.float32,
    ):
        self._metric_func = metric_func
        self._make_compatible_to_lower_scopes = make_compatible_to_lower_scopes
        self._dtype_out = dtype_out
        self.sign = sign
        self.flat = flat
        self.scope = scope

    def __call__(self, patterns, templates):
        dtype = self._dtype_out
        patterns = patterns.astype(dtype)
        templates = templates.astype(dtype)
        if self._make_compatible_to_lower_scopes:
            patterns, templates = self._expand_dims_to_match_scope(
                patterns, templates
            )
        return self._measure(patterns, templates)

    def _measure(self, patterns, templates):
        return self._metric_func(patterns, templates)

    def _expand_dims_to_match_scope(self, p, t):
        p_sl, t_sl = len(p.shape), len(t.shape)
        p_scope, t_scope = self.scope.name.split("TO")
        p_ssl = self._PARTIAL_SCOPE_TO_NDIM.get(p_scope, p_sl)
        t_ssl = self._PARTIAL_SCOPE_TO_NDIM.get(t_scope, t_sl)
        p = p[(np.newaxis,) * (p_ssl - p_sl)]
        t = t[(np.newaxis,) * (t_ssl - t_sl)]
        return p, t

    def _is_compatible(self, p, t):
        p_ndim, t_ndim = p.ndim, t.ndim
        if self.flat:
            p_ndim -= 2
            t_ndim -= 1
        scope = self._P_T_NDIM_TO_SCOPE.get((p_ndim, t_ndim), False)
        if not scope:
            return False
        if scope == self.scope:
            return True
        else:
            if self._make_compatible_to_lower_scopes:
                scope = self._SCOPE_INCLUDING_LOWER_SCOPES[scope]
            p_scope, t_scope = scope.name.split("TO")
            return (
                self._PARTIAL_SCOPE_TO_NDIM.get(p_scope, p_ndim) == p_ndim
                and self._PARTIAL_SCOPE_TO_NDIM.get(t_scope, t_ndim) == t_ndim
            )


class FlatSimilarityMetric(SimilarityMetric):
    """Similarity metric between 2D gray-tone images where the images are flattened before sent to `metric_func`"""

    _P_T_NDIM_TO_SCOPE = {
        (3, 2): MetricScope.MANY_TO_MANY,
        (1, 2): MetricScope.ONE_TO_MANY,
        (3, 1): MetricScope.MANY_TO_ONE,
        (1, 1): MetricScope.ONE_TO_ONE,
    }

    _PARTIAL_SCOPE_TO_NDIM = {
        "MANY_": 2,  #  Many patterns
        "_MANY": 2,  # Many templates
        "ONE_": 1,  # One pattern
        "_ONE": 1,  # One template
    }

    def _measure(
        self,
        patterns: Union[np.ndarray, da.Array],
        templates: Union[np.ndarray, da.Array],
    ):
        nav_shape = _get_nav_shape(patterns)
        sig_shape = _get_sig_shape(patterns)
        nav_flat_size, sig_flat_size = np.prod(nav_shape), np.prod(sig_shape)
        patterns = patterns.reshape((nav_flat_size, sig_flat_size))
        templates = templates.reshape((-1, sig_flat_size))
        return self._metric_func(patterns, templates)


def expand_dims_to_many_to_many(p, t, flat):
    """Expand the dims of patterns and templates to match MetricScope.MANY_TO_MANY.

    Parameters
    ----------
    p : Union[np.ndarray, da.Array]
        Patterns
    t : Union[np.ndarray, da.Array]
        Templates
    flat : bool, optional
        Wheter `p` and `t` are flattened
    Returns
    -------
    (p,t)
        Tuple of p and t
        with their dimensions expanded to match MetricScope.MANY_TO_MANY.
    """
    p_sl, t_sl = len(p.shape), len(t.shape)
    metric_class = FlatSimilarityMetric if flat else SimilarityMetric
    p_ssl = metric_class._PARTIAL_SCOPE_TO_NDIM["MANY_"]
    t_ssl = metric_class._PARTIAL_SCOPE_TO_NDIM["_MANY"]
    p = p[(np.newaxis,) * (p_ssl - p_sl)]
    t = t[(np.newaxis,) * (t_ssl - t_sl)]
    return p, t


def zero_mean(p, t, flat=False):
    """Subtract the mean from patterns and templates of any scope.

    Parameters
    ----------
    p : Union[np.ndarray, da.Array]
        Patterns
    t : Union[np.ndarray, da.Array]
        Templates
    flat : bool, optional
        Wheter `p` and `t` are flattened, by default False

    Returns
    -------
    (p,t)
        Tuple of p and t with their mean subtracted.
    """
    squeeze = 1 not in p.shape + t.shape
    p, t = expand_dims_to_many_to_many(p, t, flat)
    p_mean_axis = 1 if flat else (2, 3)
    t_mean_axis = 1 if flat else (1, 2)
    p -= p.mean(axis=p_mean_axis, keepdims=True)
    t -= t.mean(axis=t_mean_axis, keepdims=True)

    if squeeze:
        return (
            p.squeeze(),
            t.squeeze(),
        )
    else:
        return p, t


def normalize(p, t, flat=False):
    """Normalize patterns and templates of any scope.

    Parameters
    ----------
    p : Union[np.ndarray, da.Array]
        Patterns
    t : Union[np.ndarray, da.Array]
        Templates
    flat : bool, optional
        Wheter `p` and `t` are flattened, by default False

    Returns
    -------
    (p,t)
        Tuple of p and t divided respectively by their L2-norms.
    """
    squeeze = 1 not in p.shape + t.shape
    p, t = expand_dims_to_many_to_many(p, t, flat)
    p_sum_axis = 1 if flat else (2, 3)
    t_sum_axis = 1 if flat else (1, 2)
    p /= (p ** 2).sum(axis=p_sum_axis, keepdims=True) ** 0.5
    t /= (t ** 2).sum(axis=t_sum_axis, keepdims=True) ** 0.5

    if squeeze:
        return (
            p.squeeze(),
            t.squeeze(),
        )
    else:
        return (p, t)


SIMILARITY_METRICS: Dict[str, Callable] = {}


def _zncc_einsum(
    patterns: Union[da.Array, np.ndarray],
    templates: Union[da.Array, np.ndarray],
):
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
    patterns, templates = zero_mean(patterns, templates)
    patterns, templates = normalize(patterns, templates)

    zncc = da.einsum(
        "ijk,lmjk->ilm", templates, patterns, optimize=True
    )  # TODO: Will fail if np.ndarray!!!

    # Alternative with equivalent results:
    #   zncc = da.tensordot(templates, patterns, axes=([1, 2], [2, 3]))
    return zncc


SIMILARITY_METRICS["zncc"] = make_similarity_metric(
    _zncc_einsum,
    scope=MetricScope.MANY_TO_MANY,
    make_compatible_to_lower_scopes=True,
)


def _ndp_einsum(
    patterns: Union[da.Array, np.ndarray],
    templates: Union[da.Array, np.ndarray],
):
    patterns, templates = normalize(patterns, templates)
    ndp = da.einsum("ijk,lmjk->ilm", templates, patterns, optimize=True)
    return ndp


SIMILARITY_METRICS["ndp"] = make_similarity_metric(
    _zncc_einsum,
    scope=MetricScope.MANY_TO_MANY,
    make_compatible_to_lower_scopes=True,
)
