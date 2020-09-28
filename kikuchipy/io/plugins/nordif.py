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

"""Read/write support for EBSD patterns in NORDIF's binary format."""

import datetime
import os
import re
import time
from typing import Union, List, Dict, Optional, Tuple
import warnings

from hyperspy.misc.utils import DictionaryTreeBrowser
import numpy as np
from matplotlib.pyplot import imread

from kikuchipy.signals.util._metadata import (
    ebsd_metadata,
    metadata_nodes,
    _phase_metadata,
)


# Plugin characteristics
# ----------------------
format_name = "NORDIF"
description = "Read/write support for NORDIF pattern and setting files"
full_support = False
# Recognised file extension
file_extensions = ["dat"]
default_extension = 0
# Writing capabilities
writes = [(2, 2), (2, 1), (2, 0)]


def file_reader(
    filename: str,
    mmap_mode: Optional[str] = None,
    scan_size: Union[None, int, Tuple[int, ...]] = None,
    pattern_size: Optional[Tuple[int, ...]] = None,
    setting_file: Optional[str] = None,
    lazy: bool = False,
) -> List[Dict]:
    """Read electron backscatter patterns from a NORDIF data file.

    Parameters
    ----------
    filename
        File path to NORDIF data file.
    mmap_mode
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
        is False.

    Returns
    -------
    scan : list of dicts
        Data, axes, metadata and original metadata.
    """
    if mmap_mode is None:
        mmap_mode = "r" if lazy else "c"

    scan = {}

    # Make sure we open in correct mode
    if "+" in mmap_mode or (
        "write" in mmap_mode and "copyonwrite" != mmap_mode
    ):
        if lazy:
            raise ValueError("Lazy loading does not support in-place writing")
        f = open(filename, mode="r+b")
    else:
        f = open(filename, mode="rb")

    # Get metadata from setting file
    sem_node, ebsd_node = metadata_nodes(["sem", "ebsd"])
    folder, fname = os.path.split(filename)
    if setting_file is None:
        setting_file = os.path.join(folder, "Setting.txt")
    setting_file_exists = os.path.isfile(setting_file)
    if setting_file_exists:
        md, omd, scan_size_file = get_settings_from_file(setting_file)
        if not scan_size:
            scan_size = (scan_size_file.nx, scan_size_file.ny)
        if not pattern_size:
            pattern_size = (scan_size_file.sx, scan_size_file.sy)
    else:
        if scan_size is None and pattern_size is None:
            raise ValueError(
                "No setting file found and no scan_size or pattern_size "
                "detected in input arguments. These must be set if no setting "
                "file is provided."
            )
        md = ebsd_metadata()
        omd = DictionaryTreeBrowser()

    # Read static background image into metadata
    static_bg_file = os.path.join(folder, "Background acquisition pattern.bmp")
    try:
        md.set_item(ebsd_node + ".static_background", imread(static_bg_file))
    except FileNotFoundError:
        warnings.warn(
            f"Could not read static background pattern '{static_bg_file}', "
            "however it can be added using set_experimental_parameters()."
        )

    # Set required and other parameters in metadata
    md.set_item("General.original_filename", filename)
    md.set_item(
        "General.title", os.path.splitext(os.path.split(filename)[1])[0]
    )
    md.set_item("Signal.signal_type", "EBSD")
    md.set_item("Signal.record_by", "image")
    scan["metadata"] = md.as_dictionary()
    scan["original_metadata"] = omd.as_dictionary()

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
        data = np.memmap(f, mode=mmap_mode, dtype="uint8")

    try:
        data = data.reshape((ny, nx, sy, sx), order="C").squeeze()
    except ValueError:
        warnings.warn(
            "Pattern size and scan size larger than file size! Will attempt to "
            "load by zero padding incomplete frames."
        )
        # Data is stored image by image
        pw = [(0, ny * nx * sy * sx - data.size)]
        data = np.pad(data, pw, mode="constant")
        data = data.reshape((ny, nx, sy, sx))
    scan["data"] = data

    units = ["um"] * 4
    names = ["y", "x", "dy", "dx"]
    scales = np.ones(4)

    # Calibrate scan dimension
    try:
        scales[:2] = scales[:2] * scan_size_file.step_x
    except (TypeError, UnboundLocalError):
        warnings.warn(
            "Could not calibrate scan dimensions, this can be done using "
            "set_scan_calibration()"
        )

    # Create axis objects for each axis
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

    f.close()

    return [
        scan,
    ]


def get_settings_from_file(
    filename: str,
) -> Tuple[DictionaryTreeBrowser, DictionaryTreeBrowser, DictionaryTreeBrowser]:
    """Return metadata with parameters from NORDIF setting file.

    Parameters
    ----------
    filename
        File path of NORDIF setting file.

    Returns
    -------
    md
        Metadata complying with HyperSpy's metadata structure.
    omd
        Metadata that does not fit into HyperSpy's metadata structure.
    scan_size
        Information on image size, scan size and scan steps.
    """
    f = open(filename, "r", encoding="latin-1")  # Avoid byte strings
    content = f.read().splitlines()

    # Get line numbers of setting blocks
    blocks = {
        "[NORDIF]": -1,
        "[Microscope]": -1,
        "[EBSD detector]": -1,
        "[Detector angles]": -1,
        "[Acquisition settings]": -1,
        "[Area]": -1,
        "[Specimen]": -1,
    }
    for i, line in enumerate(content):
        for block in blocks:
            if block in line:
                blocks[block] = i
    l_nordif = blocks["[NORDIF]"]
    l_mic = blocks["[Microscope]"]
    l_det = blocks["[EBSD detector]"]
    l_ang = blocks["[Detector angles]"]
    l_acq = blocks["[Acquisition settings]"]
    l_area = blocks["[Area]"]
    l_specimen = blocks["[Specimen]"]

    # Create metadata and original metadata structures
    md = ebsd_metadata()
    sem_node, ebsd_node = metadata_nodes(["sem", "ebsd"])
    omd = DictionaryTreeBrowser()
    omd.set_item("nordif_header", content)

    # Get metadata values from settings file using regular expressions
    azimuth_angle = get_string(content, "Azimuthal\t(.*)\t", l_ang + 4, f)
    md.set_item(ebsd_node + ".azimuth_angle", float(azimuth_angle))
    beam_energy = get_string(
        content, "Accelerating voltage\t(.*)\tkV", l_mic + 5, f
    )
    md.set_item(sem_node + ".beam_energy", float(beam_energy))
    detector = get_string(content, "Model\t(.*)\t", l_det + 1, f)
    detector = re.sub("[^a-zA-Z0-9]", repl="", string=detector)
    md.set_item(ebsd_node + ".detector", "NORDIF " + detector)
    elevation_angle = get_string(content, "Elevation\t(.*)\t", l_ang + 5, f)
    md.set_item(ebsd_node + ".elevation_angle", float(elevation_angle))
    exposure_time = get_string(content, "Exposure time\t(.*)\t", l_acq + 3, f)
    md.set_item(ebsd_node + ".exposure_time", float(exposure_time) / 1e6)
    frame_rate = get_string(content, "Frame rate\t(.*)\tfps", l_acq + 1, f)
    md.set_item(ebsd_node + ".frame_rate", int(frame_rate))
    gain = get_string(content, "Gain\t(.*)\t", l_acq + 4, f)
    md.set_item(ebsd_node + ".gain", float(gain))
    magnification = get_string(content, "Magnification\t(.*)\t#", l_mic + 3, f)
    md.set_item(sem_node + ".magnification", int(magnification))
    mic_manufacturer = get_string(content, "Manufacturer\t(.*)\t", l_mic + 1, f)
    mic_model = get_string(content, "Model\t(.*)\t", l_mic + 2, f)
    md.set_item(sem_node + ".microscope", mic_manufacturer + " " + mic_model)
    sample_tilt = get_string(content, "Tilt angle\t(.*)\t", l_mic + 7, f)
    md.set_item(ebsd_node + ".sample_tilt", float(sample_tilt))
    scan_time = get_string(content, "Scan time\t(.*)\t", l_area + 7, f)
    scan_time = time.strptime(scan_time, "%H:%M:%S")
    scan_time = datetime.timedelta(
        hours=scan_time.tm_hour,
        minutes=scan_time.tm_min,
        seconds=scan_time.tm_sec,
    ).total_seconds()
    md.set_item(ebsd_node + ".scan_time", int(scan_time))
    version = get_string(content, "Software version\t(.*)\t", l_nordif + 1, f)
    md.set_item(ebsd_node + ".version", version)
    working_distance = get_string(
        content, "Working distance\t(.*)\tmm", l_mic + 6, f
    )
    md.set_item(sem_node + ".working_distance", float(working_distance))
    md.set_item(ebsd_node + ".grid_type", "square")
    md.set_item(ebsd_node + ".manufacturer", "NORDIF")
    specimen = get_string(content, "Name\t(.*)\t", l_specimen + 1, f)
    pmd = _phase_metadata()
    pmd["material_name"] = specimen
    md.set_item("Sample.Phases.1", pmd)

    # Get scan size values
    scan_size = DictionaryTreeBrowser()
    num_samp = get_string(content, "Number of samples\t(.*)\t#", l_area + 6, f)
    ny, nx = [int(i) for i in num_samp.split("x")]
    scan_size.set_item("nx", int(nx))
    scan_size.set_item("ny", int(ny))
    pattern_size = get_string(content, "Resolution\t(.*)\tpx", l_acq + 2, f)
    sx, sy = [int(i) for i in pattern_size.split("x")]
    scan_size.set_item("sx", int(sx))
    scan_size.set_item("sy", int(sy))
    step_size = get_string(content, "Step size\t(.*)\t", l_area + 5, f)
    scan_size.set_item("step_x", float(step_size))
    scan_size.set_item("step_y", float(step_size))

    return md, omd, scan_size


def get_string(content: list, expression: str, line_no: int, file) -> str:
    """Get relevant part of string using regular expression.

    Parameters
    ----------
    content
        File content to search in for the regular expression.
    expression
        Regular expression.
    line_no
        Line number to search in.
    file : file object
        File handle of open setting file.

    Returns
    -------
    str
        Output string with relevant value.
    """
    match = re.search(expression, content[line_no])
    if match is None:
        warnings.warn(
            f"Failed to read line {line_no - 1} in settings file '{file.name}' "
            f"using regular expression '{expression}'."
        )
        return 0
    else:
        return match.group(1)


def file_writer(filename: str, signal):
    """Write an :class:`~kikuchipy.signals.EBSD` or
    :class:`~kikuchipy.signals.LazyEBSD` object to a NORDIF binary
    file.

    Parameters
    ----------
    filename
        Full path of HDF file.
    signal : kikuchipy.signals.EBSD or kikuchipy.signals.LazyEBSD
        Signal instance.
    """
    with open(filename, "wb") as f:
        if signal._lazy:
            for pattern in signal._iterate_signal():
                np.array(pattern.flatten()).tofile(f)
        else:
            for pattern in signal._iterate_signal():
                pattern.flatten().tofile(f)
