# -*- coding: utf-8 -*-
# Copyright 2019-2020 The KikuchiPy developers
#
# This file is part of KikuchiPy.
#
# KikuchiPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# KikuchiPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with KikuchiPy. If not, see <http://www.gnu.org/licenses/>.

import os

import dask.array as da
import numpy as np
import pytest

import kikuchipy as kp


class TestDask:
    @pytest.mark.parametrize("mbytes_chunk", (0.5, 50, 100))
    def test_get_chunks(self, mbytes_chunk):
        data_shape = (200, 298, 60, 60)
        data_type = np.dtype("uint8")
        chunks = kp.util.dask._get_chunks(
            data_shape=data_shape, dtype=data_type, mbytes_chunk=mbytes_chunk
        )

        # Determine chunking of longest axis
        if mbytes_chunk == 0.5:
            nx, ny = (12, 12)
        elif mbytes_chunk == 50:
            nx, ny = (66, 200)
        else:  # mbytes_chunk == 100
            nx, ny = (136, 200)
        assert chunks == [ny, nx, 60, 60]

    def test_get_dask_array(self):
        s = kp.signals.EBSD(
            (255 * np.random.rand(10, 10, 120, 120)).astype(np.uint8)
        )
        dask_array = kp.util.dask._get_dask_array(s)
        assert dask_array.chunksize == (8, 8, 120, 120)

        # Make data lazy
        s.data = dask_array.rechunk((5, 5, 120, 120))
        dask_array = kp.util.dask._get_dask_array(s)
        assert dask_array.chunksize == (5, 5, 120, 120)

    def test_rechunk_learning_results(self):
        data = da.from_array(np.random.rand(10, 100, 100, 5).astype(np.float32))
        lazy_signal = kp.signals.LazyEBSD(data)

        # Decomposition
        lazy_signal.decomposition(algorithm="PCA", output_dimension=10)
        factors = lazy_signal.learning_results.factors
        loadings = lazy_signal.learning_results.loadings

        # Raise error when last dimension in factors/loadings are not identical
        with pytest.raises(ValueError, match="The last dimensions in factors"):
            kp.util.dask._rechunk_learning_results(
                factors=factors, loadings=loadings.T
            )

        # Only chunk first axis in loadings
        chunks = kp.util.dask._rechunk_learning_results(
            factors=factors, loadings=loadings, mbytes_chunk=0.02
        )
        assert chunks == [(-1, -1), (200, -1)]

        # Chunk first axis in both loadings and factors
        chunks = kp.util.dask._rechunk_learning_results(
            factors=factors, loadings=loadings, mbytes_chunk=0.01
        )
        assert chunks == [(125, -1), (62, -1)]
