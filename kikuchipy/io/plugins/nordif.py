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

"""Reader and writer of EBSD patterns from a NORDIF binary file."""

import logging
import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
from matplotlib.pyplot import imread
from orix.crystal_map import CrystalMap

from kikuchipy.detectors import EBSDDetector


__all__ = ["file_reader", "file_writer"]

_logger = logging.getLogger(__name__)

# Plugin characteristics
# ----------------------
format_name = "NORDIF"
description = "Read/write support for NORDIF pattern and setting files"
full_support = False
# Recognised file extension
file_extensions = ["dat"]
default_extension = 0
# Writing capabilities (signal dimensions, navigation dimensions)
writes = [(2, 2), (2, 1), (2, 0)]


def file_reader(
    filename: Union[str, Path],
    mmap_mode: Optional[str] = None,
    scan_size: Union[None, int, Tuple[int, ...]] = None,
    pattern_size: Optional[Tuple[int, ...]] = None,
    setting_file: Optional[str] = None,
    lazy: bool = False,
) -> List[Dict]:
    """Read electron backscatter patterns from a NORDIF data file.

    Not meant to be used directly; use :func:`~kikuchipy.load`.

    Parameters
    ----------
    filename
        File path to NORDIF data file.
    mmap_mode
        Memory map mode. If not given, ``"r"`` is used unless
        ``lazy=True``, in which case ``"c"`` is used.
    scan_size
        Scan size in number of patterns in width and height.
    pattern_size
        Pattern size in detector pixels in width and height.
    setting_file
        File path to NORDIF setting file (default is `Setting.txt` in
        same directory as ``filename``).
    lazy
        Open the data lazily without actually reading the data from disk
        until required. Allows opening arbitrary sized datasets. Default
        is ``False``.

    Returns
    -------
    scan
        Data, axes, metadata and original metadata.
    """
    if mmap_mode is None:
        mmap_mode = "r" if lazy else "c"

    scan = {}

    # Make sure we open in correct mode
    if "+" in mmap_mode or ("write" in mmap_mode and "copyonwrite" != mmap_mode):
        if lazy:
            raise ValueError("Lazy loading does not support in-place writing")
        f = open(filename, mode="r+b")
    else:
        f = open(filename, mode="rb")

    # Get metadata from setting file
    folder, _ = os.path.split(filename)
    if setting_file is None:
        setting_file = os.path.join(folder, "Setting.txt")
    setting_file_exists = os.path.isfile(setting_file)
    if setting_file_exists:
        md, omd, scan_size_file, detector_dict = _get_settings_from_file(setting_file)
        if not scan_size:
            scan_size = (scan_size_file["nx"], scan_size_file["ny"])
        if not pattern_size:
            pattern_size = (scan_size_file["sx"], scan_size_file["sy"])
    else:
        if scan_size is None or pattern_size is None:
            raise ValueError(
                "No setting file found and no scan_size or pattern_size detected in "
                "input arguments. These must be set if no setting file is provided"
            )
        warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
        md = {}
        omd = {}
        detector_dict = None

    # --- Static background
    static_bg_file = os.path.join(folder, "Background acquisition pattern.bmp")
    try:
        scan["static_background"] = imread(static_bg_file)
    except FileNotFoundError:
        scan["static_background"] = None
        warnings.warn(
            f"Could not read static background pattern '{static_bg_file}', however it "
            "can be set as 'EBSD.static_background'"
        )

    # --- Metadata
    md.update(
        {
            "General": {
                "original_filename": filename,
                "title": os.path.splitext(os.path.split(filename)[1])[0],
            },
            "Signal": {"signal_type": "EBSD", "record_by": "image"},
        }
    )
    scan["metadata"] = md
    scan["original_metadata"] = omd

    # Set scan size and image size
    if isinstance(scan_size, int):
        nx = scan_size
        ny = 1
    else:
        nx, ny = scan_size
    sx, sy = pattern_size

    # Read data from file
    data_size = ny * nx * sy * sx
    if not lazy:
        f.seek(0)
        data = np.fromfile(f, dtype="uint8", count=data_size)
    else:
        data = np.memmap(f.name, mode=mmap_mode, dtype="uint8")

    try:
        data = data.reshape((ny, nx, sy, sx)).squeeze()
    except ValueError:
        warnings.warn(
            "Pattern size and scan size larger than file size! Will attempt to "
            "load by zero padding incomplete frames."
        )
        # Data is stored image by image
        pw = [(0, ny * nx * sy * sx - data.size)]
        data = np.pad(data, pw)
        data = data.reshape((ny, nx, sy, sx))
    scan["data"] = data

    units = ["um"] * 4
    names = ["y", "x", "dy", "dx"]
    scales = np.ones(4)

    # Calibrate scan dimension
    dy = dx = 1
    try:
        dy = dx = scan_size_file["step_x"]
        scales[:2] = scales[:2] * dx
    except (TypeError, UnboundLocalError):
        warnings.warn(
            "Could not calibrate scan dimensions, this can be done using "
            "set_scan_calibration()"
        )

    # Create axis instances for each axis
    axes = [
        {
            "size": data.shape[i],
            "index_in_array": i,
            "name": names[i],
            "scale": scales[i],
            "offset": 0.0,
            "units": units[i],
        }
        for i in range(data.ndim)
    ]
    scan["axes"] = axes

    # --- Detector
    if detector_dict is not None:
        scan["detector"] = EBSDDetector(**detector_dict)

    # --- Crystal map
    scan["xmap"] = CrystalMap.empty(shape=(ny, nx), step_sizes=(dy, dx))

    f.close()

    return [scan]


def _get_settings_from_file(
    filename: str, pattern_type: str = "acquisition"
) -> Tuple[dict, dict, dict, dict]:
    """Return metadata with parameters from NORDIF setting file.

    Parameters
    ----------
    filename
        File path of NORDIF setting file.
    pattern_type
        Whether to read the ``"acquisition"`` (default) or
        ``"calibration"`` settings.

    Returns
    -------
    md
        Metadata complying with HyperSpy's metadata structure.
    omd
        Metadata that does not fit into HyperSpy's metadata structure.
    scan_size
        Information on image size, scan size and scan steps.
    detector
        Dictionary for creating an EBSD detector.
    """
    f = open(filename, "r", encoding="latin-1")  # Avoid byte strings
    content = f.read().splitlines()

    # Get line numbers of setting blocks
    blocks = {
        "[Microscope]": -1,
        "[Electron image]": -1,
        "[Detector angles]": -1,
        f"[{pattern_type.capitalize()} settings]": -1,
        "[Area]": -1,
        "[Calibration patterns]": -1,
    }
    for i, line in enumerate(content):
        for block in blocks:
            if block in line:
                blocks[block] = i
    l_mic = blocks["[Microscope]"]
    l_ang = blocks["[Detector angles]"]
    l_acq = blocks[f"[{pattern_type.capitalize()} settings]"]
    l_area = blocks["[Area]"]

    # Create metadata and original metadata structures
    beam_energy = _get_string(content, "Accelerating voltage\t(.*)\tkV", l_mic + 5, f)
    mic_manufacturer = _get_string(content, "Manufacturer\t(.*)\t", l_mic + 1, f)
    mic_model = _get_string(content, "Model\t(.*)\t", l_mic + 2, f)
    mag = _get_string(content, "Magnification\t(.*)\t#", l_mic + 3, f)
    wd = _get_string(content, "Working distance\t(.*)\tmm", l_mic + 6, f)
    md = {
        "Acquisition_instrument": {
            "SEM": {
                "beam_energy": float(beam_energy),
                "magnification": int(mag),
                "microscope": mic_manufacturer + " " + mic_model,
                "working_distance": float(wd),
            }
        }
    }
    omd = {"nordif_header": content}

    # Get scan size values
    num_samp = _get_string(content, "Number of samples\t(.*)\t#", l_area + 6, f)
    ny, nx = [int(i) for i in num_samp.split("x")]
    pattern_size = _get_string(content, "Resolution\t(.*)\tpx", l_acq + 2, f)
    sx, sy = [int(i) for i in pattern_size.split("x")]
    step_size = float(_get_string(content, "Step size\t(.*)\t", l_area + 5, f))
    scan_size = {
        "ny": ny,
        "nx": nx,
        "sy": sy,
        "sx": sx,
        "step_y": step_size,
        "step_x": step_size,
    }

    # Detector
    detector = {
        "shape": (sy, sx),
        "sample_tilt": float(_get_string(content, "Tilt angle\t(.*)\t", l_mic + 7, f)),
        "tilt": -float(_get_string(content, "Elevation\t(.*)\t", l_ang + 5, f)),
        "azimuthal": float(_get_string(content, "Azimuthal\t(.*)\t", l_ang + 4, f)),
    }
    if np.isclose(detector["tilt"], 0):  # Avoid -0.0
        detector["tilt"] = -detector["tilt"]

    if pattern_type == "calibration":
        omd = _get_calibration_pattern_settings(
            filename=filename,
            content=content,
            blocks=blocks,
            omd=omd,
            l_area=l_area,
            step_size=step_size,
            ny=ny,
            nx=nx,
        )

    return md, omd, scan_size, detector


def _get_string(content: list, expression: str, line_no: int, file) -> str:
    """Get relevant part of string using regular expression.

    Parameters
    ----------
    content
        File content to search in for the regular expression.
    expression
        Regular expression.
    line_no
        Line number to search in.
    file : file instance
        File handle of open setting file.

    Returns
    -------
    str
        Output string with relevant value.
    """
    match = re.search(expression, content[line_no])
    if match is None:
        warnings.warn(
            f"Failed to read line {line_no - 1} in settings file '{file.name}' using "
            f"regular expression '{expression}'"
        )
        return ""
    else:
        return match.group(1)


def _get_calibration_pattern_settings(
    filename: str,
    content: list,
    blocks: dict,
    omd: dict,
    l_area: int,
    step_size: float,
    ny: int,
    nx: int,
) -> dict:
    l_cal = blocks["[Calibration patterns]"]
    err = "No calibration patterns found in settings file"
    if l_cal == -1:
        raise ValueError(err)

    # Read required calibration coordinates in the area region of
    # interest
    rc = []
    for line in content[l_cal + 1 :]:
        match = re.search(r"Calibration \((.*)\)", line)
        try:
            match = match.group(1)
            match = match.split(",")
            rc.append(tuple(map(int, match))[::-1])
        except AttributeError:
            pass
    if len(rc) == 0:
        raise ValueError(err)
    rc = np.array(rc)
    omd["calibration_patterns"] = {"indices": rc}

    # Read width and height of area and ROI in area. This is added to
    # the original metadata as it is only required if we want to know
    # the sample position of each calibration pattern, e.g. when fitting
    # a plane to projection centers found from the patterns.
    l_img = blocks["[Electron image]"]
    img_string = re.search("(?<=Resolution)(.*)", content[l_img + 2])
    if img_string is not None:
        area_shape_match = re.findall(r"\d+", img_string.group())
        if len(area_shape_match) == 2:
            omd["area"] = {"shape": tuple(map(int, area_shape_match[::-1]))}
    else:
        omd["area"] = {"shape": None}
        _logger.debug("Could not read area (electron image) shape")

    keys = ["top", "left", "width", "height", "width_scaled"]
    roi = dict(zip(keys, len(keys) * [0]))
    for i, k in enumerate(keys[:4]):
        pattern = r"(?<=" + k.capitalize() + r")(.*)"
        try:
            match = re.search(pattern, content[l_area + i + 1])
            matches = re.findall(r"\d+", match.group())
            roi[k] = int(matches[2])
            if k == "width":
                roi[k + "_scaled"] = float(f"{matches[0]}.{matches[1]}")
        except (AttributeError, IndexError):
            _logger.debug(f"Could not read area ROI '{k.capitalize()}'")

    omd["roi"] = {
        "origin": (roi["top"], roi["left"]),
        "shape": (roi["height"], roi["width"]),
    }

    if roi["width_scaled"] != 0:
        factor = step_size * roi["width"] / roi["width_scaled"]

        def scale(x, return_tuple: bool = False):
            x_out = np.round(np.asarray(x) / factor).astype(int)
            if return_tuple:
                x_out = tuple(x_out)
            return x_out

        omd["calibration_patterns"]["indices_scaled"] = scale(rc)
        omd["roi"]["origin_scaled"] = scale(omd["roi"]["origin"], True)
        omd["roi"]["shape_scaled"] = scale(omd["roi"]["shape"], True)

        if omd["area"]["shape"] is not None:
            omd["area"]["shape_scaled"] = scale(omd["area"]["shape"], True)

        if omd["roi"]["shape_scaled"] != (ny, nx):
            _logger.debug(
                f"Number of samples {(ny, nx)} differs from the one calculated from "
                f"area/ROI shapes {omd['roi']['shape_scaled']}"
            )

    # Try to read area overview image
    filename_img = os.path.join(os.path.dirname(filename), "Area.bmp")
    try:
        omd["area_image"] = imread(filename_img)
    except FileNotFoundError:
        _logger.debug("No area image found")

    return omd


def file_writer(filename: str, signal: Union["EBSD", "LazyEBSD"]):
    """Write an :class:`~kikuchipy.signals.EBSD` or
    :class:`~kikuchipy.signals.LazyEBSD` instance to a NORDIF binary
    file.

    Parameters
    ----------
    filename
        Full path of HDF file.
    signal
        Signal instance.
    """
    with open(filename, "wb") as f:
        if signal._lazy:
            for pattern in signal._iterate_signal():
                np.array(pattern.flatten()).tofile(f)
        else:
            for pattern in signal._iterate_signal():
                pattern.flatten().tofile(f)
