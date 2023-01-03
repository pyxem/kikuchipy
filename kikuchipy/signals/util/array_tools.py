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

from typing import Tuple, Union

import numpy as np


def grid_indices(
    grid_shape: Union[Tuple[int, int], int],
    nav_shape: Union[Tuple[int, int], int],
    return_spacing: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Return indices of a grid evenly spaced in a larger grid of max.
    two dimensions.

    Parameters
    ----------
    grid_shape
        Tuple of integers or just an integer signifying the number of
        grid indices in each dimension. If 2D, the shape is (n rows,
        n columns).
    nav_shape
        Tuple of integers or just an integer of giving the shape of the
        larger grid. If 2D, the shape is (n rows, n columns).
    return_spacing
        Whether to return the spacing in each dimension. Default is
        ``False``.

    Returns
    -------
    indices
        Array of indices of shape ``(2,) + grid_shape`` or
        ``(1,) + grid_shape`` into the larger grid spanned by
        ``nav_shape``.
    spacing
        The spacing in each dimension. Only returned if
        ``return_spacing=True``.

    Examples
    --------
    >>> import kikuchipy as kp
    >>> kp.signals.util.grid_indices((4, 5), (55, 75))
    array([[[11, 11, 11, 11, 11],
            [22, 22, 22, 22, 22],
            [33, 33, 33, 33, 33],
            [44, 44, 44, 44, 44]],
           [[12, 25, 38, 51, 64],
            [12, 25, 38, 51, 64],
            [12, 25, 38, 51, 64],
            [12, 25, 38, 51, 64]]])
    >>> idx, spacing = kp.signals.util.grid_indices(10, 105, return_spacing=True)
    >>> idx
    array([[ 8, 18, 28, 38, 48, 58, 68, 78, 88, 98]])
    >>> spacing
    array([10])
    """
    if isinstance(grid_shape, int):
        grid_shape = (grid_shape,)
    if isinstance(nav_shape, int):
        nav_shape = (nav_shape,)

    ndim = len(nav_shape)
    if not ndim == len(grid_shape):
        raise ValueError(
            "`grid_shape` and `nav_shape` must both signify either a 1D or 2D grid"
        )

    # Get spacing in each dimension
    spacing = np.ceil(np.array(nav_shape) / (np.array(grid_shape) + 1))
    spacing = spacing.astype(int)

    idx_1d_all = np.arange(np.prod(nav_shape)).reshape(nav_shape)
    if ndim == 2:
        idx_1d = idx_1d_all[:: spacing[0], :: spacing[1]][1:, 1:]
    else:
        idx_1d = idx_1d_all[:: spacing[0]][1:]

    idx_2d = np.stack(np.unravel_index(idx_1d, nav_shape))
    if ndim == 2:
        subtract_to_center = (idx_2d[:, 0, 0] - (nav_shape - idx_2d[:, -1, -1])) // 2
    else:
        subtract_to_center = (idx_2d[:, 0] - (nav_shape - idx_2d[:, -1])) // 2

    if ndim == 2:
        idx_2d[0] -= subtract_to_center[0]
        idx_2d[1] -= subtract_to_center[1]
    else:
        idx_2d[0] -= subtract_to_center[0]

    out = idx_2d
    if return_spacing:
        out = (out, spacing)

    return out
