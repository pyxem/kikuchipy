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

from typing import Union, List, Optional, Tuple

import numpy as np

from kikuchipy.pattern import rescale_intensity


def normalize_image(
    image: np.ndarray,
    add_bright: int = 0,
    contrast: int = 1.0,
    dtype_out: Union[np.uint8, np.uint16] = np.uint8,
) -> np.ndarray:
    """Normalize an image's intensities to a mean of 0 and a standard
    deviation of 1, with the possibility to also scale by a contrast
    factor and shift the brightness values.

    Clips intensities to uint8 data type range, [0, 255].

    Adapted from the aloe/xcdskd package.

    Parameters
    ----------
    image
        Image to normalize.
    add_bright
        Brightness offset. Default is 0.
    contrast
        Contrast factor. Default is 1.0.
    dtype_out
        Output data type, either np.uint16 or np.uint8 (default).

    Returns
    -------
    image_out : np.ndarray
    """
    dtype_max = np.iinfo(dtype_out).max

    offset = (dtype_max // 2) + add_bright
    contrast *= dtype_max * 0.3125
    median = np.median(image)
    std = np.std(image)
    normalized_image = offset + ((contrast * (image - median)) / std)

    return np.clip(normalized_image, 0, dtype_max)


def get_rgb_image(
    channels: List[np.ndarray],
    percentiles: Optional[Tuple] = None,
    normalize: bool = True,
    alpha: Optional[np.ndarray] = None,
    dtype_out: Union[np.uint8, np.uint16] = np.uint8,
    **kwargs,
) -> np.ndarray:
    """Return an RGB image from three numpy arrays, with a potential
    alpha channel.

    Parameters
    ----------
    channels
        A list of np.ndarray for the red, green and blue channel,
        respectively.
    normalize
        Whether to normalize the individual `channels` before
        RGB image creation.
    alpha
        Potential alpha channel. If None (default), no alpha channel
        is added to the image.
    percentiles
        Whether to apply contrast stretching with a given percentile
        tuple with percentages, e.g. (0.5, 99.5), after creating the
        RGB image. If None (default), no contrast stretching is
        performed.
    dtype_out
        Output data type, either np.uint16 or np.uint8 (default).
    kwargs :
        Keyword arguments passed to
        :func:` ~kikuchipy.generators.util.virtual_bse.normalize_image`.

    Returns
    -------
    rgb_image : np.ndarray
        RGB image.

    See Also
    --------
    kikuchipy.generators.VirtualBSEGenerator.get_rgb_image
    """
    n_channels = 3
    rgb_image = np.zeros(channels[0].shape + (n_channels,), np.float32)
    for i, channel in enumerate(channels):
        if normalize:
            channel = normalize_image(
                channel.astype(np.float32), dtype_out=dtype_out, **kwargs
            )
        rgb_image[..., i] = channel

    # Apply alpha channel if desired
    if alpha is not None:
        alpha_min = np.nanmin(alpha)
        rescaled_alpha = (alpha - alpha_min) / (np.nanmax(alpha) - alpha_min)
        for i in range(n_channels):
            rgb_image[..., i] *= rescaled_alpha

    # Rescale to fit data type range
    if percentiles is not None:
        in_range = tuple(np.percentile(rgb_image, q=percentiles))
    else:
        in_range = None
    rgb_image = rescale_intensity(
        rgb_image, in_range=in_range, dtype_out=dtype_out
    )

    return rgb_image.astype(dtype_out)
