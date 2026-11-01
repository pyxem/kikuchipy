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
import logging
from pathlib import Path
import re
from typing import Any

import numpy as np
from packaging.version import Version

from kikuchipy import __version__
from kikuchipy.detectors._ebsd_detector import PC_CONVENTIONS, EBSDDetector

_logger = logging.getLogger(__name__)


VERSION_KEY = "kikuchipy version"


def read_ebsd_detector_from_file(fname: Path | str) -> dict[str, Any]:
    """See the docstring of the EBSD detector method using this function
    for details.
    """
    pc = np.loadtxt(fname)

    keys = {
        "shape",
        "px_size",
        "binning",
        "tilt",
        "azimuthal",
        "twist",  # New in v0.12.0
        "sample_tilt",
        "convention",
        "navigation_shape",
        VERSION_KEY,
    }

    # Parse header
    header: dict[str, str] = {}
    with open(fname, mode="r") as f:
        for line in f.readlines():
            if line[0] == "#":
                line = line[2:-1].lstrip(" ")
                if len(line) > 0:
                    match = re.match(r"^(\w+|\w+\s\w+): (.*)", line)
                    if match:
                        groups = match.groups()
                        if groups[0] in keys and len(groups) > 1:
                            header[groups[0]] = groups[1]
            else:
                break

    # Fail if not working as expected
    version = Version(header.pop(VERSION_KEY))

    missing_keys = keys - set(header.keys())
    for k in missing_keys:
        _logger.debug(
            f"Detector parameter {k!r} missing from file {str(fname)}, using default"
        )

    detector_kw: dict[str, tuple[int, int] | float | str | np.ndarray] = {}

    # Fail if not working as expected
    nrows, ncols = tuple(map(int, header["shape"][1:-1].split(",")))
    detector_kw["shape"] = (nrows, ncols)
    nav_shape_str_list = header["navigation_shape"][1:-1].split(",")
    nav_shape = []
    for i in nav_shape_str_list:
        if len(i.strip()) > 0:
            nav_shape.append(int(i))
    nav_shape = tuple(nav_shape)
    pc = pc.reshape(nav_shape + (3,))
    detector_kw["pc"] = pc

    # Allowed to fail silently (with log message)
    for k, v in header.items():
        if k in {"shape", "navigation_shape"}:
            continue
        if k in {"convention"}:
            detector_kw[k] = v
            continue
        v = v.removesuffix(" deg")
        try:
            detector_kw[k] = float(v)
        except ValueError:
            _logger.debug(
                f"Cannot cast detector parameter {k!r}: {v} to a float, using default"
            )

    # Sign of azimuthal angle was changed in v0.12.0
    if version < Version("0.12.0") and "azimuthal" in detector_kw:
        detector_kw["azimuthal"] = -detector_kw["azimuthal"]
        _logger.debug(
            f"Inverted azimuthal angle read from file {str(fname)} written with "
            f"kikuchipy {version} < 0.12.0"
        )

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
            f"{VERSION_KEY}: {__version__}\n"
            f"Time: {time_now}\n\n"
            "Column names: PCx, PCy, PCz"
        ),
    )
    kwargs.setdefault("fmt", "%.7f")
    np.savetxt(fname=filename, X=pc, **kwargs)
