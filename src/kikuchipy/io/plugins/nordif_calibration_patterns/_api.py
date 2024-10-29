# Copyright 2019-2024 The kikuchipy developers
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
import warnings

from matplotlib.pyplot import imread
import numpy as np

from kikuchipy.detectors.ebsd_detector import EBSDDetector
from kikuchipy.io.plugins.nordif._api import _get_settings_from_file


def file_reader(filename: str | Path, lazy: bool = False) -> list[dict]:
    """Return NORDIF calibration electron backscatter diffraction
    patterns in a directory with a settings text file.

    Not meant to be used directly; use :func:`~kikuchipy.load` instead.

    Parameters
    ----------
    filename
        File path to the NORDIF settings text file.
    lazy
        This parameter is not used in this reader.

    Returns
    -------
    scan
        Data, axes, metadata, and original metadata.
    """
    # Get metadata from setting file
    metadata, orig_metadata, _, detector = _get_settings_from_file(
        str(filename), pattern_type="calibration"
    )
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
            f"Could not read static background pattern {static_bg_file!r}, however it "
            "can be set as 'EBSD.static_background'"
        )

    # Set required and other parameters in metadata
    metadata.update(
        {
            "General": {
                "original_filename": filename,
                "title": "Calibration patterns",
            },
            "Signal": {"signal_type": "EBSD", "record_by": "image"},
        }
    )
    scan["metadata"] = metadata
    scan["original_metadata"] = orig_metadata

    scan["detector"] = EBSDDetector(**detector)

    yx = orig_metadata["calibration_patterns"]["indices"]

    data = _get_patterns(dirname=dirname, coordinates=yx)
    scan["data"] = data

    units = ["um"] * 3
    names = ["x", "dy", "dx"]
    scales = np.ones(3)
    axes = []
    for i in range(data.ndim):
        axes.append(
            {
                "size": data.shape[i],
                "index_in_array": i,
                "name": names[i],
                "scale": scales[i],
                "offset": 0,
                "units": units[i],
            }
        )
    scan["axes"] = axes

    return [scan]


def _get_patterns(dirname: str, coordinates: list[tuple[int, int]]) -> np.ndarray:
    patterns = []
    for y, x in coordinates:
        fname_pattern = f"Calibration ({x},{y}).bmp"
        file_pattern = os.path.join(dirname, fname_pattern)
        pattern = imread(file_pattern)
        patterns.append(pattern)
    return np.asarray(patterns)
