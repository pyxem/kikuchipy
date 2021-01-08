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

import sys
from typing import Union, Tuple, Optional

from dask.diagnostics import ProgressBar
from hyperspy._signals.signal2d import Signal2D
from hyperspy.misc.rgb_tools import rgb_dtypes
import numpy as np
from skimage.util.dtype import dtype_range

from kikuchipy.signals.util._dask import get_dask_array
from kikuchipy.pattern import chunk


class CommonImage(Signal2D):
    """A class extending HyperSpy's Signal2D class with some common
    intensity manipulation methods.

    Methods inherited from HyperSpy can be found in the HyperSpy user
    guide.

    See the docstring of :class:`hyperspy.signal.BaseSignal` for a list
    of attributes.
    """

    def rescale_intensity(
        self,
        relative: bool = False,
        in_range: Union[None, Tuple[int, int], Tuple[float, float]] = None,
        out_range: Union[None, Tuple[int, int], Tuple[float, float]] = None,
        dtype_out: Union[
            None, np.dtype, Tuple[int, int], Tuple[float, float]
        ] = None,
        percentiles: Union[None, Tuple[int, int], Tuple[float, float]] = None,
    ):
        """Rescale image intensities inplace.

        Output min./max. intensity is determined from `out_range` or the
        data type range of the :class:`numpy.dtype` passed to
        `dtype_out` if `out_range` is None.

        This method is based on
        :func:`skimage.exposure.rescale_intensity`.

        Parameters
        ----------
        relative
            Whether to keep relative intensities between images (default
            is False). If True, `in_range` must be None, because
            `in_range` is in this case set to the global min./max.
            intensity.
        in_range
            Min./max. intensity of input images. If None (default),
            `in_range` is set to pattern min./max intensity. Contrast
            stretching is performed when `in_range` is set to a narrower
            intensity range than the input patterns. Must be None if
            `relative` is True or `percentiles` are passed.
        out_range
            Min./max. intensity of output images. If None (default),
            `out_range` is set to `dtype_out` min./max according to
            `skimage.util.dtype.dtype_range`.
        dtype_out
            Data type of rescaled images, default is input images' data
            type.
        percentiles
            Disregard intensities outside these percentiles. Calculated
            per image. Must be None if `in_range` or `relative` is
            passed. Default is None.

        See Also
        --------
        kikuchipy.pattern.rescale_intensity,
        :func:`skimage.exposure.rescale_intensity`

        Examples
        --------
        Image intensities are stretched to fill the available grey
        levels in the input images' data type range or any
        :class:`numpy.dtype` range passed to `dtype_out`, either
        keeping relative intensities between images or not:

        >>> print(s.data.dtype_out, s.data.min(), s.data.max(),
        ...       s.inav[0, 0].data.min(), s.inav[0, 0].data.max())
        uint8 20 254 24 233
        >>> s2 = s.deepcopy()
        >>> s.rescale_intensity(dtype_out=np.uint16)
        >>> print(s.data.dtype_out, s.data.min(), s.data.max(),
        ...       s.inav[0, 0].data.min(), s.inav[0, 0].data.max())
        uint16 0 65535 0 65535
        >>> s2.rescale_intensity(relative=True)
        >>> print(s2.data.dtype_out, s2.data.min(), s2.data.max(),
        ...       s2.inav[0, 0].data.min(), s2.inav[0, 0].data.max())
        uint8 0 255 4 232

        Contrast stretching can be performed by passing percentiles:

        >>> s.rescale_intensity(percentiles=(1, 99))

        Here, the darkest and brightest pixels within the 1% percentile
        are set to the ends of the data type range, e.g. 0 and 255
        respectively for images of ``uint8`` data type.

        Notes
        -----
        Rescaling RGB images is not possible. Use RGB channel
        normalization when creating the image instead.
        """
        if self.data.dtype in rgb_dtypes.values():
            raise NotImplementedError(
                "Use RGB channel normalization when creating the image instead."
            )

        # Determine min./max. intensity of input image to rescale to
        if in_range is not None and percentiles is not None:
            raise ValueError(
                "'percentiles' must be None if 'in_range' is not None."
            )
        elif relative is True and in_range is not None:
            raise ValueError("'in_range' must be None if 'relative' is True.")
        elif relative:  # Scale relative to min./max. intensity in images
            in_range = (self.data.min(), self.data.max())

        if dtype_out is None:
            dtype_out = self.data.dtype.type

        if out_range is None:
            dtype_out_pass = dtype_out
            if isinstance(dtype_out, np.dtype):
                dtype_out_pass = dtype_out.type
            out_range = dtype_range[dtype_out_pass]

        # Create dask array of signal images and do processing on this
        dask_array = get_dask_array(signal=self)

        # Rescale images
        rescaled_images = dask_array.map_blocks(
            func=chunk.rescale_intensity,
            in_range=in_range,
            out_range=out_range,
            dtype_out=dtype_out,
            percentiles=percentiles,
            dtype=dtype_out,
        )

        # Overwrite signal images
        if not self._lazy:
            with ProgressBar():
                if self.data.dtype != rescaled_images.dtype:
                    self.change_dtype(dtype_out)
                print("Rescaling the image intensities:", file=sys.stdout)
                rescaled_images.store(self.data, compute=True)
        else:
            self.data = rescaled_images

    def normalize_intensity(
        self,
        num_std: int = 1,
        divide_by_square_root: bool = False,
        dtype_out: Optional[np.dtype] = None,
    ):
        """Normalize image intensities in inplace to a mean of zero with
        a given standard deviation.

        Parameters
        ----------
        num_std
            Number of standard deviations of the output intensities.
            Default is 1.
        divide_by_square_root
            Whether to divide output intensities by the square root of
            the signal dimension size. Default is False.
        dtype_out
            Data type of normalized images. If None (default), the input
            images' data type is used.

        Notes
        -----
        Data type should always be changed to floating point, e.g.
        ``np.float32`` with
        :meth:`~hyperspy.signal.BaseSignal.change_dtype`, before
        normalizing the intensities.

        Examples
        --------
        >>> np.mean(s.data)
        146.0670987654321
        >>> s.change_dtype(np.float32)  # Or passing dtype_out=np.float32
        >>> s.normalize_intensity()
        >>> np.mean(s.data)
        2.6373216e-08

        Notes
        -----
        Rescaling RGB images is not possible. Use RGB channel
        normalization when creating the image instead.
        """
        if self.data.dtype in rgb_dtypes.values():
            raise NotImplementedError(
                "Use RGB channel normalization when creating the image instead."
            )

        if dtype_out is None:
            dtype_out = self.data.dtype

        dask_array = get_dask_array(self, dtype=np.float32)

        normalized_images = dask_array.map_blocks(
            func=chunk.normalize_intensity,
            num_std=num_std,
            divide_by_square_root=divide_by_square_root,
            dtype_out=dtype_out,
            dtype=dtype_out,
        )

        # Change data type if requested
        if dtype_out != self.data.dtype:
            self.change_dtype(dtype_out)

        # Overwrite signal patterns
        if not self._lazy:
            with ProgressBar():
                print("Normalizing the image intensities:", file=sys.stdout)
                normalized_images.store(self.data, compute=True)
        else:
            self.data = normalized_images
