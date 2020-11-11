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

import dask.array as da
import numpy as np
import pytest
from scipy.spatial.distance import cdist

from kikuchipy.indexing.similarity_metrics import (
    make_similarity_metric,
    SimilarityMetric,
    MetricScope,
    FlatSimilarityMetric,
    _SIMILARITY_METRICS,
    _get_number_of_simulated,
    _zncc_einsum,
)


class TestSimilarityMetric:
    @pytest.mark.parametrize(
        "flat, returned_class",
        [(False, SimilarityMetric), (True, FlatSimilarityMetric)],
    )
    def test_make_similarity_metric(self, flat, returned_class):
        assert (
            type(
                make_similarity_metric(
                    lambda expt, sim: np.zeros((2, 4))
                    if flat
                    else np.zeros((2, 2, 2)),
                    flat=flat,
                    scope=MetricScope.MANY_TO_MANY,
                )
            )
            is returned_class
        )

    @pytest.mark.parametrize("metric", ["ncc", "ndp"])
    def test_ncc_ndp_returns_desired_array_type(self, metric):
        metric = _SIMILARITY_METRICS[metric]
        expt = np.array(
            [
                [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                [[[9, 8], [1, 7]], [[5, 2], [2, 7]]],
            ],
            np.int8,
        )
        sim = np.array([[[5, 3], [2, 7]], [[9, 8], [1, 7]]], np.int8)

        assert isinstance(metric(expt, sim), np.ndarray)
        assert isinstance(
            metric(da.from_array(expt), da.from_array(sim)), da.Array
        )
        assert isinstance(metric(expt, da.from_array(sim)), da.Array)
        assert isinstance(metric(da.from_array(expt), sim), da.Array)

    def test_flat_metric(self):
        expt = np.array(
            [
                [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                [[[9, 8], [1, 7]], [[5, 2], [2, 7]]],
            ],
            np.int8,
        )
        sim = np.array([[[5, 3], [2, 7]], [[9, 8], [1, 7]]], np.int8)
        euclidean_metric = make_similarity_metric(
            lambda expt, sim: cdist(expt, sim, metric="euclidean"),
            greater_is_better=False,
            flat=True,
            scope=MetricScope.MANY_TO_MANY,
            make_compatible_to_lower_scopes=True,
        )
        assert euclidean_metric._is_compatible(
            expt.ndim, sim.ndim
        ) is True and np.allclose(euclidean_metric(expt, sim)[2, 1], 0)

    def test_make_compatible_to_lower_scopes(self):
        ncc_metric = _SIMILARITY_METRICS["ncc"]
        assert ncc_metric._is_compatible(
            np.zeros((2, 2)).ndim, np.zeros((2, 2)).ndim
        )

    def test_too_large_scoped_inputs(self):
        metric = make_similarity_metric(
            lambda expt, sim: 1.0, scope=MetricScope.ONE_TO_ONE
        )
        assert (
            metric._is_compatible(
                np.zeros((2, 2, 2, 2)).ndim, np.zeros((4, 2, 2)).ndim
            )
            is False
        )

    def test_not_supported_inputs(self):
        metric = make_similarity_metric(
            lambda expt, sim: 1.0,
            scope=MetricScope.MANY_TO_MANY,
            make_compatible_to_lower_scopes=True,
        )
        assert (
            metric._is_compatible(
                np.zeros((2, 2, 2, 2, 2)).ndim, np.zeros((4, 2, 2)).ndim
            )
            is False
        )

    def test_too_small_scoped_inputs(self):
        metric = make_similarity_metric(
            lambda expt, sim: np.zeros((2, 2, 2)),
            scope=MetricScope.MANY_TO_MANY,
        )
        assert (
            metric._is_compatible(np.zeros((2, 2)).ndim, np.zeros((2, 2)).ndim)
            is False
        )

    def test_get_number_of_simulated(self):
        sim = np.array([[[5, 3], [2, 7]], [[9, 8], [1, 7]]], np.int8)
        assert (
            _get_number_of_simulated(sim) == 2
            and _get_number_of_simulated(sim[0]) == 1
        )

    def test_similarity_metric_representation(self):
        metrics = [
            make_similarity_metric(
                metric_func=lambda expt, sim: np.zeros((2, 2, 2)),
                scope=MetricScope.MANY_TO_MANY,
            ),
            make_similarity_metric(
                metric_func=lambda expt, sim: np.zeros((2, 2, 2)),
                scope=MetricScope.ONE_TO_MANY,
                flat=True,
            ),
            _SIMILARITY_METRICS["ncc"],
            _SIMILARITY_METRICS["ndp"],
        ]
        desired_repr = [
            "SimilarityMetric <lambda>, scope: many_to_many",
            "FlatSimilarityMetric <lambda>, scope: one_to_many",
            "SimilarityMetric _zncc_einsum, scope: many_to_many",
            "SimilarityMetric _ndp_einsum, scope: many_to_many",
        ]

        for i in range(len(desired_repr)):
            assert repr(metrics[i]) == desired_repr[i]

    def test_some_to_many(self, dummy_signal):
        scope = MetricScope.SOME_TO_MANY
        assert scope.name == "SOME_TO_MANY"
        assert scope.value == "some_to_many"

        sig_shape = dummy_signal.axes_manager.signal_shape
        expt = dummy_signal.data.reshape((-1,) + sig_shape)
        sim = expt[:3]
        dims = (expt.ndim, sim.ndim)
        assert dims == (3, 3)

        # Expansion of dimensions works
        ncc_metric = _SIMILARITY_METRICS["ncc"]
        ncc = ncc_metric(expt, sim)
        assert ncc.shape == (9, 3)
        assert np.allclose(np.diagonal(ncc), 1)

        def dot_product(a, b):
            norm_a = np.linalg.norm(a, axis=(1, 2))[:, np.newaxis, np.newaxis]
            norm_b = np.linalg.norm(b, axis=(1, 2))[:, np.newaxis, np.newaxis]
            return np.tensordot(a / norm_a, b / norm_b, axes=([1, 2], [2, 1]))

        metric = make_similarity_metric(metric_func=dot_product, scope=scope)
        assert metric._EXPT_SIM_NDIM_TO_SCOPE[dims] == scope
        assert metric._SCOPE_TO_EXPT_SIM_NDIM[scope] == dims

        ndp = metric(expt, sim)
        assert ndp.shape == (9, 3)
        assert np.allclose(np.sum(ndp), 19.92476)

    def test_some_to_many_flat(self, dummy_signal):
        scope_in = MetricScope.SOME_TO_MANY
        metric = make_similarity_metric(
            metric_func=_zncc_einsum, scope=scope_in, flat=True
        )
        scope_out = metric.scope

        assert metric.flat
        assert scope_out.name == "MANY_TO_MANY"

    def test_some_to_one(self, dummy_signal):
        scope = MetricScope.SOME_TO_ONE
        assert scope.name == "SOME_TO_ONE"
        assert scope.value == "some_to_one"

        sig_shape = dummy_signal.axes_manager.signal_shape
        expt = dummy_signal.data.reshape((-1,) + sig_shape)
        sim = expt[0]
        dims = (expt.ndim, sim.ndim)
        assert dims == (3, 2)

        # Expansion of dimensions works
        ndp_metric = _SIMILARITY_METRICS["ndp"]
        ndp = ndp_metric(expt, sim)
        assert ndp.shape == (9,)
        assert np.allclose(ndp[0], 1)

        def dot_product(a, b):
            norm_a = np.linalg.norm(a, axis=(1, 2))[:, np.newaxis, np.newaxis]
            norm_b = np.linalg.norm(b)
            return np.tensordot(a / norm_a, b / norm_b, axes=([1, 2], [1, 0]))

        metric = make_similarity_metric(metric_func=dot_product, scope=scope)
        assert metric._EXPT_SIM_NDIM_TO_SCOPE[dims] == scope
        assert metric._SCOPE_TO_EXPT_SIM_NDIM[scope] == dims

        ndp = metric(expt, sim)
        assert ndp.shape == (9,)
        assert np.allclose(np.sum(ndp), 6.9578266)

    def test_some_to_one_flat(self, dummy_signal):
        scope_in = MetricScope.SOME_TO_ONE
        metric = make_similarity_metric(
            metric_func=_zncc_einsum, scope=scope_in, flat=True
        )
        scope_out = metric.scope

        assert metric.flat
        assert scope_out.name == "MANY_TO_ONE"


class TestNCC:
    def test_zncc(self):
        ncc_metric = _SIMILARITY_METRICS["ncc"]
        # Four experimental data
        expt = np.array(
            [
                [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                [[[9, 8], [1, 7]], [[5, 2], [2, 7]]],
            ],
            np.int8,
        )
        expt_da = da.from_array(expt)

        # One perfect match, at [1,0,1] in results, and one close match
        # Two simulated
        sim = np.array([[[5, 3], [2, 7]], [[9, 8], [1, 7]]], np.int8)
        sim_da = da.from_array(sim)

        # many to many
        assert np.allclose(ncc_metric(expt_da, sim_da).compute()[1, 0, 1], 1)

        # Working with lower scopes, here one to many:
        assert np.allclose(ncc_metric(expt_da[1, 0], sim_da).compute()[1], 1)


class TestNDP:
    def test_ndp(self):
        ndp_metric = _SIMILARITY_METRICS["ndp"]
        expt = np.array(
            [
                [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                [[[9, 8], [1, 7]], [[5, 2], [2, 7]]],
            ],
            np.int8,
        )
        expt_da = da.from_array(expt)

        # One perfect match and one close match
        sim = np.array([[[5, 3], [2, 7]], [[9, 8], [1, 7]]], np.int8)
        sim_da = da.from_array(sim)

        # many to many
        assert (
            pytest.approx(ndp_metric(expt_da, sim_da).compute()[1, 0, 1]) == 1
        )
