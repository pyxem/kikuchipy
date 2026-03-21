#
# Copyright 2019-2026 the kikuchipy developers
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
#

"""IO functions for the EBSD detector."""

from datetime import datetime
from pathlib import Path
import re
from typing import Any

import numpy as np

from kikuchipy import __version__
from kikuchipy.detectors._ebsd_detector import PC_CONVENTIONS, EBSDDetector


def read_ebsd_detector_from_file(fname: Path | str) -> dict[str, Any]:
    """See the docstring of the EBSD detector method using this function
    for details.
    """
    pc = np.loadtxt(fname)

    keys = [
        "shape",
        "px_size",
        "binning",
        "tilt",
        "azimuthal",
        "twist",
        "sample_tilt",
        "convention",
        "navigation_shape",
    ]

    detector_kw: dict[str, Any] = dict(zip(keys, [None] * len(keys)))
    with open(fname, mode="r") as f:
        header = []
        for line in f.readlines():
            if line[0] == "#":
                line = line[2:-1].lstrip(" ")
                if len(line) > 0:
                    header.append(line)
                    match = re.match(r"^(\w+|\w+\s\w+): (.*)", line)
                    if match:
                        groups = match.groups()
                        if groups[0] in detector_kw and len(groups) > 1:
                            detector_kw[groups[0]] = groups[1]
            else:
                break

    for k in ["shape", "navigation_shape"]:
        shape = detector_kw[k]
        try:
            detector_kw[k] = tuple(int(i) for i in shape[1:-1].split(","))
        except ValueError:  # pragma: no cover
            detector_kw[k] = None
    for k in ["px_size", "binning", "tilt", "azimuthal", "twist", "sample_tilt"]:
        value = detector_kw[k].rstrip(" deg")
        try:
            detector_kw[k] = float(value)
        except Exception:  # pragma: no cover
            detector_kw[k] = None

    nav_shape = detector_kw.pop("navigation_shape")

    if isinstance(nav_shape, tuple):
        pc = pc.reshape(nav_shape + (3,))

    detector_kw["pc"] = pc

    return detector_kw


def write_ebsd_detector_to_file(
    detector: EBSDDetector,
    filename: str | Path,
    convention: PC_CONVENTIONS = "bruker",
    **kwargs,
) -> None:
    """See the docstring of the EBSD detector method using this function
    for details.
    """
    pc = detector._get_pc_in_convention(convention)
    pc = pc.reshape(-1, 3)

    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    kwargs.setdefault(
        "header",
        (
            f"EBSDDetector\n"
            f"  shape: {detector.shape}\n"
            f"  px_size: {detector.px_size}\n"
            f"  binning: {detector._binning}\n"
            f"  tilt: {detector.tilt} deg\n"
            f"  azimuthal: {detector.azimuthal} deg\n"
            f"  twist: {detector.twist} deg\n"
            f"  sample_tilt: {detector.sample_tilt} deg\n"
            f"  convention: {convention}\n"
            f"  navigation_shape: {detector.navigation_shape}\n\n"
            f"kikuchipy version: {__version__}\n"
            f"Time: {time_now}\n\n"
            "Column names: PCx, PCy, PCz"
        ),
    )
    kwargs.setdefault("fmt", "%.7f")
    np.savetxt(fname=filename, X=pc, **kwargs)
