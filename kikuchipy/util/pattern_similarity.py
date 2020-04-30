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

"""
Utilities for computing similarities between EBSD patterns.
"""

import numpy as np


def normalized_correlation_coefficient(pattern, template, zero_normalised=True):
    """Calculate the normalised or zero-normalised correlation coefficient
    between a image and a template following [Gonzalez2017]_.

    Parameters
    ----------
    pattern : numpy.ndarray or dask.array.Array
        Pattern to compare the template to.
    template : numpy.ndarray or dask.array.Array
        Template image.
    zero_normalised : bool, optional
        Subtract local mean value of intensities (default is ``True``).

    Returns
    -------
    coefficient : float
        Correlation coefficient in range [-1, 1] if zero normalised,
        otherwise [0, 1].

    References
    ----------
    .. [Gonzalez2017] R. C. Gonzalez, R. E. Woods, "Digital Image\
        Processing," 4th edition, Pearson Education Limited, 2017.

    """

    pattern = pattern.astype(np.float32)
    template = template.astype(np.float32)
    if zero_normalised:
        pattern = pattern - pattern.mean()
        template = template - template.mean()
    coefficient = np.sum(pattern * template) / np.sqrt(
        np.sum(pattern ** 2) * np.sum(template ** 2)
    )

    return coefficient
