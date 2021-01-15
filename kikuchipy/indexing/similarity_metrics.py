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

"""Similarity metrics for comparing gray-tone patterns."""

from enum import Enum
from typing import Callable, Tuple, Union

import dask.array as da
import numpy as np

from kikuchipy.pattern._pattern import _normalize, _zero_mean


class MetricScope(Enum):
    """Describes the input parameters for a similarity metric. See
    :func:`make_similarity_metric`.
    """

    MANY_TO_MANY = "many_to_many"
    SOME_TO_MANY = "some_to_many"
    ONE_TO_MANY = "one_to_many"
    ONE_TO_ONE = "one_to_one"
    MANY_TO_ONE = "many_to_one"
    SOME_TO_ONE = "some_to_one"


def make_similarity_metric(
    metric_func: Callable,
    greater_is_better: bool = True,
    scope: Enum = MetricScope.MANY_TO_MANY,
    flat: bool = False,
    make_compatible_to_lower_scopes: bool = False,
    dtype_out: np.dtype = np.float32,
):
    """Make a similarity metric for comparing gray-tone patterns of
    equal size.

    This factory function wraps metric functions for use in
    :meth:`~kikuchipy.signals.EBSD.match_patterns` (which uses
    :class:`~kikuchipy.indexing.StaticPatternMatching`).

    Parameters
    ----------
    metric_func : Callable
        Metric function with signature
        `metric_func(experimental, simulated)`, which computes the
        similarity or a distance matrix between experimental
        and simulated pattern(s)).
    greater_is_better : bool, optional
        Whether greater values correspond to more similar patterns, by
        default True. Used for choosing `keep_n` metric results in
        `pattern_match`.
    scope : MetricScope, optional
        Describes the shapes of the `metric_func`'s input parameters
        `experimental` and `simulated`, by default
        `MetricScope.MANY_TO_MANY`.
    flat : bool, optional
        Whether experimental and simulated patterns are to be flattened
        before sent to `metric_func` when the similarity metric is
        called, by default False.
    make_compatible_to_lower_scopes : bool, optional
        Whether to reshape experimental and simulated data by adding
        single dimensions to match the given scope, by default False.
    dtype_out : numpy.dtype, optional
        The data type used and returned by the metric, by default
        :class:`numpy.float32`.

    Returns
    -------
    metric_class : SimilarityMetric or FlatSimilarityMetric
        A callable class instance computing a similarity matrix with
        signature `metric_func(experimental, simulated)`.

    Notes
    -----
    The metric function must take the arrays `experimental` and
    `simulated` as arguments, in that order. The scope and whether the
    metric is flat defines the intended data shapes. In the following
    table, (ny, nx) and (sy, sx) correspond to the navigation and signal
    shapes (row, column), respectively, with N number of simulations:

    ============ ============= ========= ========= ============= ========= =========
    MetricScope  flat = False                      flat = True
    ------------ --------------------------------- ---------------------------------
    -            experimental  simulated returns   experimental  simulated returns
    ============ ============= ========= ========= ============= ========= =========
    MANY_TO_MANY (ny,nx,sy,sx) (N,sy,sx) (ny,nx,N) (ny*nx,sy*sx) (N,sy*sx) (ny*nx,N)
    SOME_TO_MANY (nx,sy,sx)    (N,sy,sx) (nx,N)    -             -         -
    ONE_TO_MANY  (sy,sx)       (N,sy,sx) (N,)      (sy*sx,)      (N,sy*sx) (N,)
    MANY_TO_ONE  (ny,nx,sy,sx) (sy,sx)   (ny,nx)   (ny*nx,sy*sx) (sy*sx,)  (ny*nx)
    SOME_TO_ONE  (nx,sy,sx)    (sy,sx)   (nx,)     -             -         -
    ONE_TO_ONE   (sy,sx)       (sy,sx)   (1,)      (sy*sx,)      (sy*sx,)  (1,)
    ============ ============= ========= ========= ============= ========= =========

    If a scope of `SOME_TO_MANY` or `SOME_TO_ONE` and `flat=True` is
    desired, the returned similarity metric has the scope `MANY_TO_MANY`
    or `MANY_TO_ONE`, respectively.
    """
    sign = 1 if greater_is_better else -1
    if flat:
        if "some" in scope.value:
            if "many" in scope.value:
                scope = MetricScope.MANY_TO_MANY
            else:  # "one" in scope.value
                scope = MetricScope.MANY_TO_ONE
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
    """Similarity metric between 2D gray-tone patterns."""

    # See table in docstring of `make_similarity_metric`
    _EXPT_SIM_NDIM_TO_SCOPE = {
        (4, 3): MetricScope.MANY_TO_MANY,
        (3, 3): MetricScope.SOME_TO_MANY,
        (2, 3): MetricScope.ONE_TO_MANY,
        (4, 2): MetricScope.MANY_TO_ONE,
        (3, 2): MetricScope.SOME_TO_ONE,
        (2, 2): MetricScope.ONE_TO_ONE,
    }

    _SCOPE_TO_EXPT_SIM_NDIM = {
        MetricScope.MANY_TO_MANY: (4, 3),
        MetricScope.SOME_TO_MANY: (3, 3),
        MetricScope.ONE_TO_MANY: (2, 3),
        MetricScope.MANY_TO_ONE: (4, 2),
        MetricScope.SOME_TO_ONE: (3, 2),
        MetricScope.ONE_TO_ONE: (2, 2),
    }

    _SCOPE_TO_LOWER_SCOPES = {
        MetricScope.MANY_TO_MANY: (
            MetricScope.SOME_TO_MANY,
            MetricScope.ONE_TO_MANY,
            MetricScope.MANY_TO_ONE,
            MetricScope.SOME_TO_ONE,
            MetricScope.ONE_TO_ONE,
        ),
        MetricScope.SOME_TO_MANY: (
            MetricScope.ONE_TO_MANY,
            MetricScope.SOME_TO_ONE,
            MetricScope.ONE_TO_ONE,
        ),
        MetricScope.ONE_TO_MANY: (MetricScope.ONE_TO_ONE,),
        MetricScope.MANY_TO_ONE: (
            MetricScope.SOME_TO_ONE,
            MetricScope.ONE_TO_ONE,
        ),
        MetricScope.ONE_TO_ONE: (),
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
        experimental: Union[np.ndarray, da.Array],
        simulated: Union[np.ndarray, da.Array],
    ) -> Union[np.ndarray, da.Array]:
        dtype = self._dtype_out
        experimental = experimental.astype(dtype)
        simulated = simulated.astype(dtype)
        if isinstance(experimental, da.Array):
            experimental = _rechunk(experimental)
        if isinstance(simulated, da.Array):
            simulated = _rechunk(simulated)
        if self._make_compatible_to_lower_scopes:
            experimental, simulated = self._expand_dims_to_match_scope(
                experimental, simulated
            )
        return self._measure(experimental, simulated).squeeze()

    def __repr__(self):
        return (
            f"{self.__class__.__name__} {self._metric_func.__name__}, "
            f"scope: {self.scope.value}"
        )

    def _measure(
        self,
        experimental: Union[np.ndarray, da.Array],
        simulated: Union[np.ndarray, da.Array],
    ) -> Union[np.ndarray, da.Array]:
        """Measure the similarities and return the results."""
        return self._metric_func(experimental, simulated)

    def _expand_dims_to_match_scope(
        self,
        expt: Union[np.ndarray, da.Array],
        sim: Union[np.ndarray, da.Array],
    ) -> Tuple[Union[np.ndarray, da.Array], Union[np.ndarray, da.Array]]:
        """Return experimental and simulated data with added axes
        corresponding to the scope of the metric.
        """
        expt_scope_ndim, sim_scope_ndim = self._SCOPE_TO_EXPT_SIM_NDIM[
            self.scope
        ]
        expt = expt[(np.newaxis,) * (expt_scope_ndim - expt.ndim)]
        sim = sim[(np.newaxis,) * (sim_scope_ndim - sim.ndim)]
        return expt, sim

    def _is_compatible(self, expt_ndim: int, sim_ndim: int) -> bool:
        """Return whether shapes of experimental and simulated are
        compatible with the metric's scope.
        """
        if self.flat:
            expt_ndim = expt_ndim // 2  # 4 -> 2 or 2 -> 1
            sim_ndim -= 1
        inferred_scope = self._EXPT_SIM_NDIM_TO_SCOPE.get(
            (expt_ndim, sim_ndim), False
        )
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
    """Similarity metric between 2D gray-tone images where the
    navigation and signal axes are flattened before sent to
    `metric_func`.
    """

    # See table in docstring of `make_similarity_metric`
    _EXPT_SIM_NDIM_TO_SCOPE = {
        (2, 2): MetricScope.MANY_TO_MANY,
        (1, 2): MetricScope.ONE_TO_MANY,
        (3, 1): MetricScope.MANY_TO_ONE,
        (1, 1): MetricScope.ONE_TO_ONE,
    }

    _SCOPE_TO_EXPT_SIM_NDIM = {
        MetricScope.MANY_TO_MANY: (2, 2),
        MetricScope.ONE_TO_MANY: (1, 2),
        MetricScope.MANY_TO_ONE: (3, 1),
        MetricScope.ONE_TO_ONE: (1, 1),
    }

    _PARTIAL_SCOPE_TO_NDIM = {
        "MANY_": 2,  # Many experimental
        "_MANY": 2,  # Many simulated
        "ONE_": 1,  # One pattern
        "_ONE": 1,  # One template
    }

    def _measure(
        self,
        experimental: Union[np.ndarray, da.Array],
        simulated: Union[np.ndarray, da.Array],
    ) -> Union[np.ndarray, da.Array]:
        """Flatten images, measure the similarities and return the
        results.
        """
        nav_shape = _get_nav_shape(experimental)
        sig_shape = _get_sig_shape(experimental)
        nav_flat_size, sig_flat_size = np.prod(nav_shape), np.prod(sig_shape)
        experimental = experimental.reshape((nav_flat_size, sig_flat_size))
        simulated = simulated.reshape((-1, sig_flat_size))
        return self._metric_func(experimental, simulated)


def _rechunk(dask_array: da.Array):
    ndim_to_chunks = {
        2: {0: -1, 1: -1},
        3: {0: "auto", 1: -1, 2: -1},
        4: {0: "auto", 1: "auto", 2: -1, 3: -1},
    }
    return dask_array.rechunk(ndim_to_chunks[dask_array.ndim])


def _get_nav_shape(expt):
    return {2: (), 3: (expt.shape[0],), 4: (expt.shape[:2])}[expt.ndim]


def _get_sig_shape(expt):
    return expt.shape[-2:]


def _get_number_of_simulated(sim):
    return sim.shape[0] if sim.ndim == 3 else 1


def _expand_dims_to_many_to_many(
    expt: Union[np.ndarray, da.Array],
    sim: Union[np.ndarray, da.Array],
    flat: bool,
) -> Tuple[Union[np.ndarray, da.Array], Union[np.ndarray, da.Array]]:
    """Expand the dims of experimental and simulated to match
    `MetricScope.MANY_TO_MANY`.

    Parameters
    ----------
    expt
        Experimental data.
    sim
        Simulated data.
    flat
        Whether `expt` and `sim` are flattened.

    Returns
    -------
    expt
        Experimental data dimension expanded to match
        `MetricScope.MANY_TO_MANY`.
    sim
        Simulated data dimension expanded to match
        `MetricScope.MANY_TO_MANY`.
    """
    metric_cls = FlatSimilarityMetric if flat else SimilarityMetric
    expt_scope_ndim, sim_scope_ndim = metric_cls._SCOPE_TO_EXPT_SIM_NDIM[
        MetricScope.MANY_TO_MANY
    ]
    expt = expt[(np.newaxis,) * (expt_scope_ndim - expt.ndim)]
    sim = sim[(np.newaxis,) * (sim_scope_ndim - sim.ndim)]
    return expt, sim


def _zero_mean_expt_sim(
    expt: Union[np.ndarray, da.Array],
    sim: Union[np.ndarray, da.Array],
    flat: bool = False,
) -> Tuple[Union[np.ndarray, da.Array], Union[np.ndarray, da.Array]]:
    """Subtract the mean from experimental and simulated data of any
    scope.

    Parameters
    ----------
    expt : np.ndarray or da.Array
        Experimental.
    sim : np.ndarray or da.Array
        Simulated.
    flat : bool, optional
        Whether `expt` and `sim` are flattened, by default False.

    Returns
    -------
    expt
        Experimental data with mean subtracted.
    sim
        Simulated data with mean subtracted.
    """
    expt_expanded, sim_expanded = _expand_dims_to_many_to_many(expt, sim, flat)
    mean_axis = 1 if flat else (-2, -1)
    expt_mean_sub = _zero_mean(patterns=expt_expanded, axis=mean_axis)
    sim_mean_sub = _zero_mean(patterns=sim_expanded, axis=mean_axis)
    if 1 not in expt.shape + sim.shape:
        expt_mean_sub = expt_mean_sub.squeeze()
        sim_mean_sub = sim_mean_sub.squeeze()
    return expt_mean_sub, sim_mean_sub


def _normalize_expt_sim(
    expt: Union[np.ndarray, da.Array],
    sim: Union[np.ndarray, da.Array],
    flat: bool = False,
) -> Tuple[Union[np.ndarray, da.Array], Union[np.ndarray, da.Array]]:
    """Normalize experimental and simulated patterns of any scope.

    Parameters
    ----------
    expt : numpy.ndarray or dask.array.Array
        Experimental patterns.
    sim : numpy.ndarray or dask.array.Array
        Simulated patterns.
    flat : bool, optional
        Whether `expt` and `sim` are flattened, by default False.

    Returns
    -------
    expt : numpy.ndarray or dask.array.Array
        Experimental patterns divided by their L2 norms.
    sim : numpy.ndarray or dask.array.Array
        Simulated patterns divided by their L2 norms.
    """
    expt_expanded, sim_expanded = _expand_dims_to_many_to_many(expt, sim, flat)
    sum_axis = 1 if flat else (-2, -1)
    expt_normalized = _normalize(patterns=expt_expanded, axis=sum_axis)
    sim_normalized = _normalize(patterns=sim_expanded, axis=sum_axis)
    if 1 not in expt.shape + sim.shape:
        expt_normalized = expt_normalized.squeeze()
        sim_normalized = sim_normalized.squeeze()
    return expt_normalized, sim_normalized


def _zncc_einsum(
    experimental: Union[da.Array, np.ndarray],
    simulated: Union[da.Array, np.ndarray],
) -> Union[np.ndarray, da.Array]:
    experimental, simulated = _zero_mean_expt_sim(experimental, simulated)
    experimental, simulated = _normalize_expt_sim(experimental, simulated)
    r = da.einsum("ijkl,mkl->ijm", experimental, simulated, optimize=True)
    if isinstance(experimental, np.ndarray) and isinstance(
        simulated, np.ndarray
    ):
        return r.compute()
    else:
        return r


def _ndp_einsum(
    experimental: Union[da.Array, np.ndarray],
    simulated: Union[da.Array, np.ndarray],
) -> Union[np.ndarray, da.Array]:
    experimental, simulated = _normalize_expt_sim(experimental, simulated)
    rho = da.einsum("ijkl,mkl->ijm", experimental, simulated, optimize=True)
    if isinstance(experimental, np.ndarray) and isinstance(
        simulated, np.ndarray
    ):
        return rho.compute()
    else:
        return rho


ncc = make_similarity_metric(
    metric_func=_zncc_einsum,
    scope=MetricScope.MANY_TO_MANY,
    make_compatible_to_lower_scopes=True,
)
ncc.__doc__ = r"""
    A similarity metric for calculation of the normalized
    cross-correlation coefficient (NCC) `r` :cite:`goshtasby2012image`
    between experimental and simulated patterns.

    Parameters
    ----------
    experimental : numpy.ndarray or dask.array.Array
        Experimental patterns.
    simulated : numpy.ndarray or dask.array.Array
        Simulated patterns.

    Returns
    -------
    r : numpy.ndarray or dask.array.Array
        Correlation coefficients in range [-1, 1] for all comparisons,
        as :class:`numpy.ndarray` if both `experimental` and `simulated`
        are :class:`numpy.ndarray`, else :class:`dask.array.Array`.

    Notes
    -----
    The NCC, or Pearson Correlation Coefficient, is defined as

    .. math::

        r = \frac
            {\sum^n_{i=1}(x_i - \bar{x})(y_i - \bar{y})}
            {
                \sqrt{\sum ^n _{i=1}(x_i - \bar{x})^2}
                \sqrt{\sum ^n _{i=1}(y_i - \bar{y})^2}
            },

    where experimental patterns :math:`x` and simulated patterns
    :math:`y` are centered by subtracting out the mean of each pattern,
    and the sum of cross-products of the centered patterns is
    accumulated. The denominator adjusts the scales of the patterns to
    have equal units.

    Equivalent results are obtained with :func:`dask.array.tensordot`
    with ``axes=([2, 3], [1, 2]))`` for 4D and 3D experimental and
    simulated data sets, respectively.
"""


ndp = make_similarity_metric(
    metric_func=_ndp_einsum,
    scope=MetricScope.MANY_TO_MANY,
    make_compatible_to_lower_scopes=True,
)
ndp.__doc__ = r"""
    A similarity metric for calculation of the normalized dot product
    (NDP) :math:`\rho` :cite:`chen2015dictionary` between experimental
    and simulated patterns.

    Parameters
    ----------
    experimental : numpy.ndarray or dask.array.Array
        Experimental patterns.
    simulated : numpy.ndarray or dask.array.Array
        Simulated patterns.

    Returns
    -------
    rho : numpy.ndarray or dask.array.Array
        Normalized dot products in range [0, 1] for all comparisons,
        as :class:`numpy.ndarray` if both `experimental` and `simulated`
        are :class:`numpy.ndarray`, else :class:`dask.array.Array`.

    Notes
    -----
    The NDP is defined as

    .. math::

        \rho = \frac
        {\langle \mathbf{X}, \mathbf{Y} \rangle}
        {||\mathbf{X}|| \cdot ||\mathbf{Y}||},

    where :math:`{\langle \mathbf{X}, \mathbf{Y} \rangle}` is the dot
    (inner) product of the pattern vectors :math:`\mathbf{X}` and
    :math:`\mathbf{Y}`.
"""


_SIMILARITY_METRICS = {
    "ncc": ncc,
    "ndp": ndp,
}
