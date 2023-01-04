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

"""Reader of EBSD calibration patterns from NORDIF files."""

import os
from pathlib import Path
from typing import List, Tuple, Union
import warnings

from matplotlib.pyplot import imread
import numpy as np

from kikuchipy.detectors import EBSDDetector
from kikuchipy.io.plugins.nordif import _get_settings_from_file


__all__ = ["file_reader"]


# Plugin characteristics
# ----------------------
format_name = "NORDIF calibration patterns"
description = "Read support for NORDIF's calibration patterns"
full_support = False
# Recognised file extension
file_extensions = ["txt"]
default_extension = 0
# Writing capabilities (signal dimensions, navigation dimensions)
writes = False


def file_reader(filename: Union[str, Path], lazy: bool = False) -> List[dict]:
    """Reader electron backscatter patterns from .bmp files stored in a
    NORDIF project directory, their filenames listed in a text file.

    Not meant to be used directly; use :func:`~kikuchipy.load`.

    Parameters
    ----------
    filename
        File path to the NORDIF settings text file.
    lazy
        This parameter is not used in this reader.

    Returns
    -------
    scan
        Data, axes, metadata and original metadata.
    """
    # Get metadata from setting file
    md, omd, _, detector = _get_settings_from_file(filename, pattern_type="calibration")
    dirname = os.path.dirname(filename)

    scan = {}
    # Read static background pattern, to be passed to EBSD.__init__() to
    # set the EBSD.static_background property
    static_bg_file = os.path.join(dirname, "Background calibration pattern.bmp")
    try:
        scan["static_background"] = imread(static_bg_file)
    except FileNotFoundError:
        scan["static_background"] = None
        warnings.warn(
            f"Could not read static background pattern '{static_bg_file}', however it "
            "can be set as 'EBSD.static_background'"
        )

    # Set required and other parameters in metadata
    md.update(
        {
            "General": {
                "original_filename": filename,
                "title": "Calibration patterns",
            },
            "Signal": {"signal_type": "EBSD", "record_by": "image"},
        }
    )
    scan["metadata"] = md
    scan["original_metadata"] = omd

    scan["detector"] = EBSDDetector(**detector)

    yx = omd["calibration_patterns"]["indices"]

    data = _get_patterns(dirname=dirname, coordinates=yx)
    scan["data"] = data

    units = ["um"] * 3
    names = ["x", "dy", "dx"]
    scales = np.ones(3)
    scan["axes"] = [
        {
            "size": data.shape[i],
            "index_in_array": i,
            "name": names[i],
            "scale": scales[i],
            "offset": 0,
            "units": units[i],
        }
        for i in range(data.ndim)
    ]

    return [scan]


def _get_patterns(dirname: str, coordinates: List[Tuple[int]]) -> np.ndarray:
    patterns = []
    for y, x in coordinates:
        fname_pattern = f"Calibration ({x},{y}).bmp"
        file_pattern = os.path.join(dirname, fname_pattern)
        pattern = imread(file_pattern)
        patterns.append(pattern)
    return np.asarray(patterns)
