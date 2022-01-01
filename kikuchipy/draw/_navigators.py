# Copyright 2019-2022 The kikuchipy developers
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

"""Convenience functions for creating HyperSpy signals to use as
navigators with :meth:`~hyperspy.signals.Signal2D.plot`.
"""

from typing import Union

import numpy as np
from skimage.exposure import rescale_intensity

import hyperspy.api as hs


def get_rgb_navigator(image, dtype: Union[type, np.dtype] = np.uint16):
    """Create an RGB navigator signal which is suitable to pass to
    :meth:`~hyperspy.signals.Signal2D.plot` as the `navigator`
    parameter.

    Parameters
    ----------
    image : numpy.ndarray
        RGB color image of shape (n rows, n columns, 3).
    dtype : numpy.dtype
        Which data type to cast the signal data to, either uint16
        (default) or uint8.

    Returns
    -------
    signal : hyperspy.signals.Signal2D
        Signal with an (n columns, n rows) signal shape and no
        navigation shape, of data type either rgb8 or rgb16.
    """
    image_rescaled = rescale_intensity(image, out_range=dtype).astype(dtype)
    s = hs.signals.Signal2D(image_rescaled)
    s = s.transpose(signal_axes=1)
    s.change_dtype({"uint8": "rgb8", "uint16": "rgb16"}[np.dtype(dtype).name])
    return s
