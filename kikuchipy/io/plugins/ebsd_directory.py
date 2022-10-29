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

"""Reader of EBSD patterns from a dictionary of images."""

import glob
import logging
import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Union
import warnings

import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import imageio.v3 as iio
import numpy as np


__all__ = ["file_reader"]


_logger = logging.getLogger(__name__)


# Plugin characteristics
# ----------------------
format_name = "Directory of EBSD patterns"
description = "Read support for patterns in image files in a directory"
full_support = False
# Recognised file extension
file_extensions = ["tif", "tiff", "bmp", "png"]
default_extension = 0
# Writing capabilities (signal dimensions, navigation dimensions)
writes = False


def file_reader(
    filename: Union[str, Path],
    xy_pattern: Optional[str] = None,
    show_progressbar: Optional[bool] = None,
    lazy: bool = False,
) -> List[Dict]:
    r"""Read all images in a directory, assuming they are electron
    backscatter diffraction (EBSD) patterns of equal shape and data
    type.

    Not meant to be used directly; use :func:`~kikuchipy.load`.

    Parameters
    ----------
    filename
        Name of directory with patterns.
    xy_pattern
        Regular expression to extract map coordinates from the
        filenames. If not given, two regular expressions will be tried:
        assuming (x, y) = (5, 10), "_x5y10.tif" or "-5-10.bmp".
        Valid ``xy_pattern`` equal to these are ``r"_x(\d+)y(\d+).tif"``
        and ``r"-(\d+)-(\d+).bmp"``, respectively. If none of these
        expressions match the first file's name in the directory, a
        warning is printed and the returned signal will have only one
        navigation dimension.
    show_progressbar
        Whether to show a progressbar when reading the signal into
        memory when ``lazy=False``.
    lazy
        Read the patterns lazily without actually reading them from disk
        until required. Default is ``False``.

    Returns
    -------
    scan
        Data, axes, metadata and original metadata.

    Warns
    -----
    UserWarning
        If navigation coordinates can not be read from the filenames.
    UserWarning
        If there are more detected patterns in the directory than the
        navigation shape determined from the filenames suggest.

    Notes
    -----
    Adapted from https://blog.dask.org/2019/06/20/load-image-data.
    """
    # Read all filenames
    filenames = glob.glob(filename)
    n_patterns = len(filenames)
    _logger.info(f"{n_patterns} patterns found in directory")
    ext = os.path.splitext(filename)[1]

    # Get regex pattern
    if xy_pattern is None:
        for p in [rf"_x(\d+)y(\d+){ext}", rf"-(\d+)-(\d+){ext}"]:
            match = re.search(p, filenames[0])
            if match is not None:
                xy_pattern = p
                break

    if xy_pattern is None:
        warnings.warn(
            "Returned signal will have one navigation dimension as coordinates could "
            "not be read from the file names"
        )
        nav_shape = (n_patterns,)
    else:
        # Read coordinates of each file
        fn_idx_sets = dict()
        xy_coords = np.zeros((n_patterns, 2), dtype=int)
        for j, fn in enumerate(filenames):
            for i, idx in enumerate(re.search(xy_pattern, fn).groups()):
                fn_idx_sets.setdefault(i, set())
                fn_idx_sets[i].add(int(idx))
                xy_coords[j, i] = idx

        # Sort indices and determine navigation shape
        fn_idx_sets = list(map(sorted, fn_idx_sets.values()))
        nav_shape = tuple(map(len, fn_idx_sets))[::-1]

        if n_patterns != int(np.prod(nav_shape)):
            warnings.warn(
                "Returned signal will have one navigation dimension as the number of "
                f"patterns found in the directory, {n_patterns}, does not match the "
                f"navigation shape determined from the filenames, {nav_shape}."
            )
            nav_shape = (n_patterns,)

    _logger.info(f"Navigation shape is {nav_shape}")

    # Read one pattern
    sample = iio.imread(filenames[0])
    sig_shape = sample.shape
    dtype = sample.dtype
    _logger.info(f"Sample pattern has shape {sig_shape} and dtype {dtype}")

    # Read all patterns lazily
    lazy_patterns = [dask.delayed(iio.imread)(fn) for fn in filenames]
    lazy_patterns = [
        da.from_delayed(x, shape=sig_shape, dtype=dtype) for x in lazy_patterns
    ]

    # Concatenate patterns into full dataset
    if xy_pattern is None or nav_shape == (n_patterns,):
        data = da.stack(lazy_patterns)
    else:
        data = np.empty(nav_shape + (1, 1), dtype=object)
        for (x, y), pat in zip(xy_coords, lazy_patterns):
            data[fn_idx_sets[1].index(y), fn_idx_sets[0].index(x), 0, 0] = pat
        data = da.block(data.tolist())

    if not lazy:
        pbar = ProgressBar()
        if show_progressbar:
            pbar.register()
        data_np = np.empty(nav_shape + sig_shape, dtype=dtype)
        data.store(data_np, compute=True)
        data = data_np
        try:
            pbar.unregister()
        except KeyError:
            pass

    nav_dim = len(nav_shape)
    ndim = nav_dim + 2

    units = ["um"] * ndim
    scales = [1] * ndim
    axes_names = ["y", "x"][-nav_dim:] + ["dy", "dx"]
    axes = [
        {
            "size": data.shape[i],
            "index_in_array": i,
            "name": axes_names[i],
            "scale": scales[i],
            "offset": 0.0,
            "units": units[i],
        }
        for i in range(ndim)
    ]
    metadata = dict(Signal=dict(signal_type="EBSD", record_by="image"))

    scan = dict(axes=axes, data=data, metadata=metadata)

    return [scan]
