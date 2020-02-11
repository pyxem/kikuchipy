# -*- coding: utf-8 -*-
# Copyright 2019-2020 The KikuchiPy developers
#
# This file is part of KikuchiPy.
#
# KikuchiPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# KikuchiPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with KikuchiPy. If not, see <http://www.gnu.org/licenses/>.

import logging
import os
import warnings

import dask.array as da
from hyperspy.misc.utils import DictionaryTreeBrowser
from hyperspy.io_plugins.hspy import overwrite_dataset
import h5py
import numpy as np

from kikuchipy.util.phase import _phase_metadata, _update_phase_info
from kikuchipy.util.io import (
    kikuchipy_metadata,
    metadata_nodes,
    _get_input_variable,
)
from kikuchipy.util.general import (
    _delete_from_nested_dictionary,
    _get_nested_dictionary,
)

_logger = logging.getLogger(__name__)

# Plugin characteristics
# ----------------------
format_name = "h5ebsd"
description = (
    "Read/write support for electron backscatter patterns stored in "
    "an HDF5 file formatted in the h5ebsd format introduced in "
    "Jackson et al.: h5ebsd: an archival data format for electron"
    "back-scatter diffraction data sets. Integrating Materials and"
    "Manufacturing Innovation 2014 3:4, doi: "
    "https://dx.doi.org/10.1186/2193-9772-3-4."
)
full_support = False
# Recognised file extension
file_extensions = ["h5", "hdf5", "h5ebsd"]
default_extension = 1
# Writing capabilities
writes = [(2, 2), (2, 1), (1, 1)]


def file_reader(filename, scans=None, lazy=False, **kwargs):
    """Read electron backscatter patterns from an h5ebsd file
    [Jackson2014]_. A valid h5ebsd file has at least one group with the
    name '/Scan x/EBSD' with the groups 'Data' (patterns etc.) and
    'Header' (``metadata`` etc.) , where 'x' is the scan_number.

    Parameters
    ----------
    filename : str
        Full file path of the HDF file.
    scans : int or list of ints
        Integer of scan to return, or list of integers of scans to
        return. If None is passed the first scan in the file is returned.
    lazy : bool, optional
        Open the data lazily without actually reading the data from disk
        until required. Allows opening arbitrary sized datasets. Default
        is ``False``.

    Returns
    -------
    scan_dict_list : list of dicts
        Data, axes, metadata and original metadata.

    References
    ----------
    .. [Jackson2014] M. A. Jackson, M. A. Groeber, M. D. Uchic, D. J.
        Rowenhorst and M. De Graef, "h5ebsd: an archival data format for
        electron back-scatter diffraction data sets," *Integrating
        Materials and Manufacturing Innovation* **3** (2014), doi:
        https://doi.org/10.1186/2193-9772-3-4.
    """

    mode = kwargs.pop("mode", "r")
    f = h5py.File(filename, mode=mode, **kwargs)

    # Check if h5ebsd file
    check_h5ebsd(f)

    # Get manufacturer and version and check if reading the file is supported
    man, ver = manufacturer_version(f)
    man_pats = manufacturer_pattern_names()
    if any(man == s for s in man_pats.keys()) is not True:
        raise IOError(
            f"Manufacturer {man} not among recognised manufacturers "
            f"{list(man_pats.keys())}."
        )

    # Get scans to return
    scans_file = [f[k] for k in f["/"].keys() if "Scan" in k]
    scans_return = []
    if scans is None:  # Return first scan
        scans_return.append(scans_file[0])
    else:
        if isinstance(scans, int):
            scans = [
                scans,
            ]
        for scan_no in scans:  # Wanted scans
            scan_is_here = False
            for scan in scans_file:
                if scan_no == int(scan.name.split()[-1]):
                    scans_return.append(scan)
                    scan_is_here = True
                    break
            if not scan_is_here:
                scan_nos = [int(i.name.split()[-1]) for i in scans_file]
                error_str = (
                    f"Scan {scan_no} is not among the available scans "
                    f"{scan_nos} in '{filename}'."
                )
                if len(scans) == 1:
                    raise IOError(error_str)
                else:
                    warnings.warn(error_str)

    # Parse file
    scan_dict_list = []
    for scan in scans_return:
        scan_dict_list.append(h5ebsd2signaldict(scan, man, ver, lazy=lazy))

    if not lazy:
        f.close()

    return scan_dict_list


def check_h5ebsd(file):
    """Check if HDF file is an h5ebsd file by searching for datasets
    containing manufacturer, version and scans in the top group.

    Parameters
    ----------
    file : h5py:File
        File where manufacturer, version and scan datasets should
        reside in the top group.
    """

    file_keys_lower = [key.lstrip().lower() for key in file["/"].keys()]
    if not any(s in file_keys_lower for s in ["manufacturer", "version"]):
        raise IOError(
            f"'{file.filename}' is not an h5ebsd file, as manufacturer and/or"
            " version could not be read from its top group."
        )

    if not any(
        "Scan" in key
        and "EBSD/Data" in file[key]
        and "EBSD/Header" in file[key]
        for key in file["/"].keys()
    ):
        raise IOError(
            f"'{file.filename}' is not an h5ebsd file, as no scans in a group "
            "with name 'Scan <scan_number>/EBSD' with groups 'Data' and "
            "'Header' could be read."
        )


def manufacturer_version(file):
    """Get manufacturer and version from h5ebsd file.

    Parameters
    ----------
    file : h5py:File
        File with manufacturer and version datasets in the top group.

    Returns
    -------
    manufacturer : str
    version : str
    """

    manufacturer = None
    version = None
    for key, val in h5ebsdgroup2dict(file["/"]).items():
        if key.lower() == "manufacturer":
            manufacturer = val
        if key.lower() == "version":
            version = val
    return manufacturer, version


def manufacturer_pattern_names():
    """Return mapping of string of supported manufacturers to the names
    of their HDF dataset where the patterns are stored.

    Returns
    -------
    dict
    """

    return {
        "KikuchiPy": "patterns",
        "EDAX": "Pattern",
        "Bruker Nano": "RawPatterns",
    }


def h5ebsdgroup2dict(group, dictionary=None, recursive=False, lazy=False):
    """Return a dictionary with values from datasets in a group in an
    opened h5ebsd file.

    Parameters
    ----------
    group : h5py:Group
        HDF group object.
    dictionary : dict, hyperspy.misc.utils.DictionaryTreeBrowser or\
            None, optional
        To fill dataset values into.
    recursive : bool, optional
        Whether to add subgroups to dictionary.
    lazy : bool, optional
        Read dataset lazily.

    Returns
    -------
    dictionary : dict
        Dataset values in group (and subgroups if ``recursive=True``).
    """

    man_pats = manufacturer_pattern_names()
    if dictionary is None:
        dictionary = {}
    for key, val in group.items():
        # Prepare value for entry in dictionary
        if isinstance(val, h5py.Dataset):
            if key not in man_pats.values():
                val = val[()]
            if isinstance(val, np.ndarray) and len(val) == 1:
                val = val[0]
                key = key.lstrip()  # EDAX has some leading whitespaces
            if val.dtype.char == "S":
                val = val.decode()
        # Check whether to extract subgroup or write value to dictionary
        if isinstance(val, h5py.Group) and recursive:
            dictionary[key] = {}
            h5ebsdgroup2dict(
                group[key], dictionary[key], recursive=recursive, lazy=lazy
            )
        else:
            dictionary[key] = val
    return dictionary


def h5ebsd2signaldict(scan_group, manufacturer, version, lazy=False):
    """Return a dictionary with ``signal``, ``metadata`` and
    ``original_metadata`` from an h5ebsd scan.

    Parameters
    ----------
    scan_group : h5py:Group
        HDF group of scan.
    manufacturer : 'KikuchiPy', 'EDAX' or 'Bruker Nano'
        Manufacturer of file.
    version : str
        Version of manufacturer software.
    lazy : bool, optional
        Read dataset lazily.

    Returns
    -------
    scan : dict
        Dictionary with patterns, ``metadata`` and
        ``original_metadata``.
    """

    md, omd, scan_size = h5ebsdheader2dicts(
        scan_group, manufacturer, version, lazy
    )
    md.set_item("Signal.signal_type", "EBSD")
    md.set_item("Signal.record_by", "image")

    scan = {
        "metadata": md.as_dictionary(),
        "original_metadata": omd.as_dictionary(),
        "attributes": {},
    }

    # Get data dataset
    man_pats = manufacturer_pattern_names()
    data_dset = None
    for man, pats in man_pats.items():
        if manufacturer.lower() == man.lower():
            try:
                data_dset = scan_group["EBSD/Data/" + pats]
            except KeyError:
                raise KeyError(
                    "Could not find patterns in the expected dataset "
                    f"'EBSD/Data/{pats}'"
                )
            break

    # Get data from group
    if lazy:
        if data_dset.chunks is None:
            chunks = "auto"
        else:
            chunks = data_dset.chunks
        data = da.from_array(data_dset, chunks=chunks)
    else:
        data = np.asanyarray(data_dset)

    sx, sy = scan_size.sx, scan_size.sy
    nx, ny = scan_size.nx, scan_size.ny
    try:
        data = data.reshape((ny, nx, sy, sx)).squeeze()
    except ValueError:
        warnings.warn(
            f"Pattern size ({sx} x {sy}) and scan size ({nx} x {ny}) larger "
            "than file size. Will attempt to load by zero padding incomplete "
            "frames."
        )
        # Data is stored pattern by pattern
        pw = [(0, ny * nx * sy * sx - data.size)]
        if lazy:
            data = da.pad(data.flatten(), pw, mode="constant")
        else:
            data = np.pad(data.flatten(), pw, mode="constant")
        data = data.reshape((ny, nx, sy, sx))
    scan["data"] = data

    units = ["\u03BC" + "m"] * 4
    names = ["y", "x", "dy", "dx"]
    scales = np.ones(4)

    # Calibrate scan dimension and detector dimension
    step_x, step_y = scan_size.step_x, scan_size.step_y
    scales[0] = scales[0] * step_x
    scales[1] = scales[1] * step_y
    detector_pixel_size = scan_size.delta
    scales[2] = scales[2] * detector_pixel_size
    scales[3] = scales[3] * detector_pixel_size

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

    return scan


def h5ebsdheader2dicts(scan_group, manufacturer, version, lazy=False):
    """Return three dictionaries in HyperSpy's
    :class:`hyperspy.misc.utils.DictionaryTreeBrowser` format, one
    with the h5ebsd scan header parameters as KikuchiPy metadata,
    another with all datasets in the header as original metadata, and
    the last with info about scan size, pattern size and detector pixel
    size.

    Parameters
    ----------
    scan_group : h5py:Group
        HDF group of scan data and header.
    manufacturer : 'KikuchiPy', 'EDAX' or 'Bruker Nano'
        Manufacturer of file.
    version : str
        Version of manufacturer software used to create file.
    lazy : bool, optional
        Read dataset lazily.

    Returns
    -------
    md : :class:`hyperspy.misc.utils.DictionaryTreeBrowser`
        KikuchiPy ``metadata`` elements available in file.
    omd : :class:`hyperspy.misc.utils.DictionaryTreeBrowser`
        All metadata available in file.
    scan_size : :class:`hyperspy.misc.utils.DictionaryTreeBrowser`
        Scan, pattern, step and detector pixel size available in file.
    """

    md = kikuchipy_metadata()
    title = (
        scan_group.file.filename.split("/")[-1].split(".")[0]
        + " "
        + scan_group.name[1:].split("/")[0]
    )
    md.set_item("General.title", title)

    if "edax" in manufacturer.lower():
        md, omd, scan_size = edaxheader2dicts(scan_group, md)
    elif "bruker" in manufacturer.lower():
        md, omd, scan_size = brukerheader2dicts(scan_group, md)
    else:  # KikuchiPy
        md, omd, scan_size = kikuchipyheader2dicts(scan_group, md, lazy)

    ebsd_node = metadata_nodes(sem=False)
    md.set_item(ebsd_node + ".manufacturer", manufacturer)
    md.set_item(ebsd_node + ".version", version)

    return md, omd, scan_size


def kikuchipyheader2dicts(scan_group, md, lazy=False):
    """Return scan metadata dictionaries from a KikuchiPy h5ebsd file.

    Parameters
    ----------
    scan_group : h5py:Group
        HDF group of scan data and header.
    md : hyperspy.misc.utils.DictionaryTreeBrowser
        Dictionary with empty fields from KikuchiPy's metadata.
    lazy : bool, optional
        Read dataset lazily.

    Returns
    -------
    md : :class:`hyperspy.misc.utils.DictionaryTreeBrowser`
        KikuchiPy ``metadata`` elements available in KikuchiPy file.
    omd : :class:`hyperspy.misc.utils.DictionaryTreeBrowser`
        All metadata available in KikuchiPy file.
    scan_size : :class:`hyperspy.misc.utils.DictionaryTreeBrowser`
        Scan, pattern, step and detector pixel size available in
        KikuchiPy file.
    """

    omd = DictionaryTreeBrowser()
    sem_node, ebsd_node = metadata_nodes()
    md.set_item(
        ebsd_node, h5ebsdgroup2dict(scan_group["EBSD/Header"], lazy=lazy)
    )
    md = _delete_from_nested_dictionary(md, "Phases")
    phase_node = "Sample.Phases"
    md.set_item(sem_node, h5ebsdgroup2dict(scan_group["SEM/Header"], lazy=lazy))
    md.set_item(
        phase_node,
        h5ebsdgroup2dict(scan_group["EBSD/Header/Phases"], recursive=True),
    )

    # Get and remove scan info values from metadata
    mapping = {
        "sx": "pattern_width",
        "sy": "pattern_height",
        "nx": "n_columns",
        "ny": "n_rows",
        "step_x": "step_x",
        "step_y": "step_y",
        "delta": "detector_pixel_size",
    }
    scan_size = DictionaryTreeBrowser()
    for k, v in mapping.items():
        scan_size.set_item(k, _get_nested_dictionary(md, ebsd_node + "." + v))
    md = _delete_from_nested_dictionary(md, mapping.values())

    return md, omd, scan_size


def edaxheader2dicts(scan_group, md):
    """Return scan metadata dictionaries from an EDAX TSL h5ebsd file.

    Parameters
    ----------
    scan_group : h5py:Group
        HDF group of scan data and header.
    md : hyperspy.misc.utils.DictionaryTreeBrowser
        Dictionary with empty fields from KikuchiPy's metadata.

    Returns
    -------
    md : :class:`hyperspy.misc.utils.DictionaryTreeBrowser`
        KikuchiPy ``metadata`` elements available in EDAX file.
    omd : :class:`hyperspy.misc.utils.DictionaryTreeBrowser`
        All metadata available in EDAX file.
    scan_size : :class:`hyperspy.misc.utils.DictionaryTreeBrowser`
        Scan, pattern, step and detector pixel size available in EDAX
        file.
    """

    # Get header group as dictionary
    hd = h5ebsdgroup2dict(scan_group["EBSD/Header"], recursive=True)

    # Populate metadata dictionary
    sem_node, ebsd_node = metadata_nodes()
    md.set_item(ebsd_node + ".azimuth_angle", hd["Camera Azimuthal Angle"])
    md.set_item(ebsd_node + ".elevation_angle", hd["Camera Elevation Angle"])
    grid_type = hd["Grid Type"]
    if grid_type == "SqrGrid":
        md.set_item(ebsd_node + ".grid_type", "square")
    else:
        raise IOError(
            f"Only square grids are supported, however a {grid_type} grid was "
            "passed."
        )
    md.set_item(ebsd_node + ".sample_tilt", hd["Sample Tilt"])
    md.set_item("General.authors", hd["Operator"])
    md.set_item("General.notes", hd["Notes"])
    md.set_item(ebsd_node + ".xpc", hd["Pattern Center Calibration"]["x-star"])
    md.set_item(ebsd_node + ".ypc", hd["Pattern Center Calibration"]["y-star"])
    md.set_item(ebsd_node + ".zpc", hd["Pattern Center Calibration"]["z-star"])
    md.set_item(sem_node + ".working_distance", hd["Working Distance"])
    if "SEM-PRIAS Images" in scan_group.keys():
        md.set_item(
            sem_node + ".magnification",
            scan_group["SEM-PRIAS Images/Header/Mag"][0],
        )
    # Loop over phases in group and add to metadata
    for phase_no, phase in hd["Phase"].items():
        phase["material_name"] = phase["MaterialName"]
        phase["lattice_constants"] = [
            phase["Lattice Constant a"],
            phase["Lattice Constant b"],
            phase["Lattice Constant c"],
            phase["Lattice Constant alpha"],
            phase["Lattice Constant beta"],
            phase["Lattice Constant gamma"],
        ]
        md = _update_phase_info(md, phase, phase_no)

    # Populate original metadata dictionary
    omd = DictionaryTreeBrowser({"edax_header": hd})

    # Populate scan size dictionary
    scan_size = DictionaryTreeBrowser()
    scan_size.set_item("sx", hd["Pattern Width"])
    scan_size.set_item("sy", hd["Pattern Height"])
    scan_size.set_item("nx", hd["nColumns"])
    scan_size.set_item("ny", hd["nRows"])
    scan_size.set_item("step_x", hd["Step X"])
    scan_size.set_item("step_y", hd["Step Y"])
    scan_size.set_item("delta", 1.0)

    return md, omd, scan_size


def brukerheader2dicts(scan_group, md):
    """Return scan metadata dictionaries from a Bruker h5ebsd file.

    Parameters
    ----------
    scan_group : h5py:Group
        HDF group of scan data and header.
    md : hyperspy.misc.utils.DictionaryTreeBrowser
        Dictionary with empty fields from KikuchiPy's metadata.

    Returns
    -------
    md : :class:`hyperspy.misc.utils.DictionaryTreeBrowser`
        KikuchiPy ``metadata`` elements available in Bruker file.
    omd : :class:`hyperspy.misc.utils.DictionaryTreeBrowser`
        All metadata available in Bruker file.
    scan_size : :class:`hyperspy.misc.utils.DictionaryTreeBrowser`
        Scan, pattern, step and detector pixel size available in Bruker
        file.
    """

    # Get header group and data group as dictionaries
    hd = h5ebsdgroup2dict(scan_group["EBSD/Header"], recursive=True)
    dd = h5ebsdgroup2dict(scan_group["EBSD/Data"])

    # Populate metadata dictionary
    sem_node, ebsd_node = metadata_nodes()
    md.set_item(ebsd_node + ".elevation_angle", hd["CameraTilt"])
    grid_type = hd["Grid Type"]
    if grid_type == "isometric":
        md.set_item(ebsd_node + ".grid_type", "square")
    else:
        raise IOError(
            f"Only square grids are supported, however a {grid_type} grid was "
            "passed."
        )
    md.set_item(ebsd_node + ".sample_tilt", hd["SampleTilt"])
    md.set_item(ebsd_node + ".xpc", np.mean(dd["PCX"]))
    md.set_item(ebsd_node + ".ypc", np.mean(dd["PCY"]))
    md.set_item(ebsd_node + ".zpc", np.mean(dd["DD"]))
    md.set_item(ebsd_node + ".static_background", hd["StaticBackground"])
    md.set_item(sem_node + ".working_distance", hd["WD"])
    md.set_item(sem_node + ".beam_energy", hd["KV"])
    md.set_item(sem_node + ".magnification", hd["Magnification"])
    # Loop over phases
    for phase_no, phase in hd["Phases"].items():
        phase["material_name"] = phase["Name"]
        phase["space_group"] = phase["IT"]
        phase["atom_coordinates"] = {}
        for key, val in phase["AtomPositions"].items():
            atom = val.split(",")
            atom[1:] = list(map(float, atom[1:]))
            phase["atom_coordinates"][key] = {
                "atom": atom[0],
                "coordinates": atom[1:4],
                "site_occupation": atom[4],
                "debye_waller_factor": atom[5],
            }
        md = _update_phase_info(md, phase, phase_no)

    # Populate original metadata dictionary
    omd = DictionaryTreeBrowser({"bruker_header": hd})

    # Populate scan size dictionary
    scan_size = DictionaryTreeBrowser()
    scan_size.set_item("sx", hd["PatternWidth"])
    scan_size.set_item("sy", hd["PatternHeight"])
    scan_size.set_item("nx", hd["NCOLS"])
    scan_size.set_item("ny", hd["NROWS"])
    scan_size.set_item("step_x", hd["XSTEP"])
    scan_size.set_item("step_y", hd["YSTEP"])
    scan_size.set_item(
        "delta", hd["DetectorFullHeightMicrons"] / hd["UnClippedPatternHeight"]
    )

    return md, omd, scan_size


def file_writer(filename, signal, add_scan=None, scan_number=1, **kwargs):
    """Write an :class:`~kikuchipy.signals.ebsd.EBSD` or
    :class:`~kikuchipy.signals.ebsd.LazyEBSD` signal to an existing,
    but not open, or new h5ebsd file.

    Only writing to KikuchiPy's h5ebsd format is supported.

    Parameters
    ----------
    filename : str
        Full path of HDF file.
    signal : kikuchipy.signals.ebsd.EBSD or\
            kikuchipy.signals.ebsd.LazyEBSD
        Signal instance.
    add_scan : None, bool, optional
        Add signal to an existing, but not open, h5ebsd file. If it does
        not exist it is created and the signal is written to it.
    scan_number : int, optional
        Scan number in name of HDF dataset when writing to an existing,
        but not open, h5ebsd file.
    **kwargs :
        Keyword arguments passed to :meth:`h5py:Group.require_dataset`.
    """

    # Set manufacturer and version to use in file
    from kikuchipy.release import version as ver_signal

    man_ver_dict = {"manufacturer": "KikuchiPy", "version": ver_signal}

    # Open file in correct mode
    mode = "w"
    if os.path.isfile(filename) and add_scan:
        mode = "r+"
    try:
        f = h5py.File(filename, mode=mode)
    except OSError:
        raise OSError("Cannot write to an already open file.")

    if os.path.isfile(filename) and add_scan:
        check_h5ebsd(f)
        man_file, ver_file = manufacturer_version(f)
        if man_ver_dict["manufacturer"].lower() != man_file.lower():
            f.close()
            raise IOError(
                f"Only writing to KikuchiPy's (and not {man_file}'s) h5ebsd "
                "format is supported."
            )
        man_ver_dict["version"] = ver_file

        # Get valid scan number
        scans_file = [f[k] for k in f["/"].keys() if "Scan" in k]
        scan_nos = [int(i.name.split()[-1]) for i in scans_file]
        for i in scan_nos:
            if i == scan_number:
                q = f"Scan {i} already in file, enter another scan number:\n"
                scan_number = _get_input_variable(q, int)
                if scan_number is None:
                    raise IOError("Invalid scan number.")
    else:  # File did not exist
        dict2h5ebsdgroup(man_ver_dict, f["/"], **kwargs)

    scan_group = f.create_group("Scan " + str(scan_number))

    # Create scan dictionary with EBSD and SEM metadata
    # Add scan size, pattern size and detector pixel size to dictionary to write
    sx, sy = signal.axes_manager.signal_shape
    nx, ny = signal.axes_manager.navigation_shape
    nav_indices = signal.axes_manager.navigation_indices_in_array
    sig_indices = signal.axes_manager.signal_indices_in_array
    md = signal.metadata.deepcopy()
    sem_node, ebsd_node = metadata_nodes()
    md.set_item(ebsd_node + ".pattern_width", sx)
    md.set_item(ebsd_node + ".pattern_height", sy)
    md.set_item(ebsd_node + ".n_columns", nx)
    md.set_item(ebsd_node + ".n_rows", ny)
    md.set_item(
        ebsd_node + ".step_x", signal.axes_manager[nav_indices[0]].scale
    )
    md.set_item(
        ebsd_node + ".step_y", signal.axes_manager[nav_indices[1]].scale
    )
    md.set_item(
        ebsd_node + ".detector_pixel_size",
        signal.axes_manager[sig_indices[1]].scale,
    )
    # Separate EBSD and SEM metadata
    det_str, ebsd_str = ebsd_node.split(".")[-2:]  # Detector and EBSD nodes
    md_sem = md.get_item(sem_node).copy().as_dictionary()  # SEM node as dict
    md_det = md_sem.pop(det_str)  # Remove/assign detector node from SEM node
    md_ebsd = md_det.pop(ebsd_str)
    # Phases
    if md.get_item("Sample.Phases") is None:
        md = _update_phase_info(md, _phase_metadata())  # Add default phase
    md_ebsd["Phases"] = md.Sample.Phases.as_dictionary()
    for phase in md_ebsd["Phases"].keys():  # Ensure coordinates are arrays
        atom_coordinates = md_ebsd["Phases"][phase]["atom_coordinates"]
        for atom in atom_coordinates.keys():
            atom_coordinates[atom]["coordinates"] = np.array(
                atom_coordinates[atom]["coordinates"]
            )
    scan = {"EBSD": {"Header": md_ebsd}, "SEM": {"Header": md_sem}}

    # Write scan dictionary to HDF groups
    dict2h5ebsdgroup(scan, scan_group)

    # Write signal to file
    man_pats = manufacturer_pattern_names()
    dset_pattern_name = man_pats["KikuchiPy"]
    overwrite_dataset(
        scan_group.create_group("EBSD/Data"),
        signal.data.reshape(nx * ny, sy, sx),
        dset_pattern_name,
        signal_axes=(2, 1),
        **kwargs,
    )
    (
        nx_start,
        nx_stop,
        ny_start,
        ny_stop,
    ) = signal.axes_manager.navigation_extent
    sample_pos = {
        "x_sample": np.tile(np.linspace(nx_start, nx_stop, nx), ny),
        "y_sample": np.tile(np.linspace(ny_start, ny_stop, ny), nx),
    }
    dict2h5ebsdgroup(sample_pos, scan_group["EBSD/Data"])

    f.close()
    _logger.info("File closed.")


def dict2h5ebsdgroup(dictionary, group, **kwargs):
    """Write a dictionary from ``metadata`` to datasets in a new group
    in an opened HDF file in the h5ebsd format.

    Parameters
    ----------
    dictionary : dict
        ``Metadata``, with keys as dataset names.
    group : h5py:Group
        HDF group to write dictionary to.
    **kwargs :
        Keyword arguments passed to :meth:`h5py:Group.require_dataset`.
    """

    for key, val in dictionary.items():
        ddtype = type(val)
        dshape = (1,)
        written = False
        if isinstance(val, (dict, DictionaryTreeBrowser)):
            if isinstance(val, DictionaryTreeBrowser):
                val = val.as_dictionary()
            dict2h5ebsdgroup(val, group.create_group(key), **kwargs)
            written = True
        elif isinstance(val, str):
            ddtype = "S" + str(len(val) + 1)
            val = val.encode()
        elif isinstance(val, (np.ndarray, da.Array)):
            overwrite_dataset(group, val, key, **kwargs)
            written = True
        elif ddtype == np.dtype("O"):
            try:
                ddtype = h5py.special_dtype(vlen=val[0].dtype)
                dshape = np.shape(val)
            except TypeError:
                warnings.warn(
                    "The hdf5 writer could not write the following information "
                    "to the file '{} : {}'.".format(key, val)
                )
                break
        if written:
            continue  # Jump to next item in dictionary
        group.create_dataset(key, shape=dshape, dtype=ddtype, **kwargs)
        group[key][()] = val
