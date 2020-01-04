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

import dask.array as da

import kikuchipy as kp


def _update_learning_results(learning_results, components, dtype_out):
    """Update learning results before calling
    :meth:`hyperspy.learn.mva.MVA.get_decomposition_model` by
    changing data type, keeping only desired components and rechunking
    them into suitable chunks if they are lazy.

    Parameters
    ----------
    learning_results : hyperspy.learn.mva.LearningResults
        Learning results with component patterns and loadings.
    components : None, int or list of ints
        If ``None``, rebuilds the signal from all ``components``. If
        ``int``, rebuilds signal from ``components`` in range 0-given
        ``int``. If list of ``int``, rebuilds signal from only
        ``components`` in given list.
    dtype_out : numpy.float16, numpy.float32 or numpy.float64
        Data type to cast learning results to.

    Returns
    -------
    factors : :attr:`hyperspy.learn.mva.LearningResults.factors`
        Updated component patterns in learning results.
    loadings : :attr:`hyperspy.learn.mva.LearningResults.loadings`
        Updated component loadings in learning results.
    """

    # Change data type
    factors = learning_results.factors.astype(dtype_out)
    loadings = learning_results.loadings.astype(dtype_out)

    # Keep desired components
    if hasattr(components, "__iter__"):  # components is a list of ints
        factors = factors[:, components]
        loadings = loadings[:, components]
    else:  # components is an int
        factors = factors[:, :components]
        loadings = loadings[:, :components]

    # Rechunk if learning results are lazy
    if isinstance(factors, da.Array) and isinstance(loadings, da.Array):
        chunks = kp.util.dask._rechunk_learning_results(
            factors=factors, loadings=loadings
        )
        factors = factors.rechunk(chunks=chunks[0])
        loadings = loadings.rechunk(chunks=chunks[1])

    return factors, loadings
