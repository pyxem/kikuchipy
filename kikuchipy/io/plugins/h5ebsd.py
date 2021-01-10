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

"""Read/write support for EBSD patterns in some HDF5 file formats."""

import os
from typing import Union, List, Tuple, Optional, Dict
import warnings

import dask.array as da
from hyperspy.misc.utils import DictionaryTreeBrowser
from hyperspy.exceptions import VisibleDeprecationWarning
from hyperspy.io_plugins.hspy import overwrite_dataset
import h5py
import numpy as np

from kikuchipy.io._util import (
    _get_input_variable,
    _get_nested_dictionary,
    _delete_from_nested_dictionary,
)
from kikuchipy.signals.util._metadata import (
    ebsd_metadata,
    metadata_nodes,
    _phase_metadata,
    _update_phase_info,
)


# Plugin characteristics
# ----------------------
format_name = "h5ebsd"
description = (
    "Read/write support for electron backscatter diffraction patterns "
    "stored in an HDF5 file formatted in the h5ebsd format introduced "
    "in Jackson et al.: h5ebsd: an archival data format for electron"
    "back-scatter diffraction data sets. Integrating Materials and"
    "Manufacturing Innovation 2014 3:4, doi: "
    "https://dx.doi.org/10.1186/2193-9772-3-4."
)
full_support = False
# Recognised file extension
file_extensions = ["h5", "hdf5", "h5ebsd"]
default_extension = 1
# Writing capabilities (signal dimensions, navigation dimensions)
writes = [(2, 2), (2, 1), (2, 0)]

# Unique HDF5 footprint
footprint = ["manufacturer", "version"]


def file_reader(
    filename: str,
    scan_group_names: Union[None, str, List[str]] = None,
    lazy: bool = False,
    **kwargs,
) -> List[dict]:
    """Read electron backscatter diffraction patterns from an h5ebsd
    file [Jackson2014]_. A valid h5ebsd file has at least one top group
    with the subgroup 'EBSD' with the subgroups 'Data' (patterns etc.)
    and 'Header' (``metadata`` etc.).

    Parameters
    ----------
    filename
        Full file path of the HDF file.
    scan_group_names
        Name or a list of names of HDF5 top group(s) containing the
        scan(s) to return. If None, the first scan in the file is
        returned.
    lazy
        Open the data lazily without actually reading the data from disk
        until required. Allows opening arbitrary sized datasets. Default
        is False.
    kwargs
        Key word arguments passed to :obj:`h5py:File`.

    Returns
    -------
    scan_dict_list: list of dicts
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

    # Check if h5ebsd file, and get all scan groups
    check_h5ebsd(f)

    # Get manufacturer and version and check if reading the file is supported
    man, ver = manufacturer_version(f)
    man_pats = manufacturer_pattern_names()
    if any(man.lower() == s for s in man_pats.keys()) is not True:
        raise IOError(
            f"Manufacturer {man} not among recognised manufacturers "
            f"{list(man_pats.keys())}."
        )

    # Get scans to return
    scans_return = get_desired_scan_groups(
        file=f, scan_group_names=scan_group_names
    )

    # Parse file
    scan_dict_list = []
    for scan in scans_return:
        scan_dict_list.append(h5ebsd2signaldict(scan, man, ver, lazy=lazy))

    if not lazy:
        f.close()

    return scan_dict_list


def check_h5ebsd(file: h5py.File):
    """Check if HDF file is an h5ebsd file by searching for datasets
    containing manufacturer, version and scans in the top group.

    Parameters
    ----------
    file: h5py:File
        File where manufacturer, version and scan datasets should
        reside in the top group.
    """
    top_groups = list(file["/"].keys())
    scan_groups = get_scan_groups(file)
    n_groups = len(top_groups)
    if len(scan_groups) != n_groups - 2:
        raise IOError(
            f"'{file.filename}' is not an h5ebsd file, as manufacturer and/or"
            " version could not be read from its top group."
        )

    if not any(
        "EBSD/Data" in group and "EBSD/Header" in group for group in scan_groups
    ):
        raise IOError(
            f"'{file.filename}' is not an h5ebsd file, as no top groups with "
            "subgroup name 'EBSD' with subgroups 'Data' and 'Header' was "
            "detected."
        )


def get_scan_groups(file: h5py.File) -> List[h5py.Group]:
    """Return a list of the scan group names from an h5ebsd file.

    Parameters
    ----------
    file : h5py:File
        File where manufacturer, version and scan datasets should reside
        in the top group.

    Returns
    -------
    scan_groups : h5py:Group
        List of available scan groups.
    """
    scan_groups = []
    for key in file["/"].keys():
        if key.lstrip().lower() not in ["manufacturer", "version"]:
            scan_groups.append(file[key])

    return scan_groups


def manufacturer_version(file: h5py.File) -> Tuple[str, str]:
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
    for key, val in hdf5group2dict(group=file["/"]).items():
        if key.lower() == "manufacturer":
            manufacturer = val
        if key.lower() == "version":
            version = val
    return manufacturer, version


def manufacturer_pattern_names() -> Dict[str, str]:
    """Return mapping of string of supported manufacturers to the names
    of their HDF dataset where the patterns are stored.

    Returns
    -------
    dict
    """
    return {
        "kikuchipy": "patterns",
        "edax": "Pattern",
        "bruker nano": "RawPatterns",
    }


def get_desired_scan_groups(
    file: h5py.File, scan_group_names: Union[None, str, List[str]] = None,
) -> List[h5py.Group]:
    """Get the desired HDF5 groups with scans within them.

    Parameters
    ----------
    file : h5py:File
        File where manufacturer, version and scan datasets should
        reside in the top group.
    scan_group_names
        Name or a list of names of the desired top HDF5 group(s). If
        None, the first scan group is returned.

    Returns
    -------
    scan_groups
        A list of the desired scan group(s) in the file.
    """
    # Get available scan group names in the file
    scan_groups_file = get_scan_groups(file)

    # Get desired scan groups
    scan_groups = []
    if scan_group_names is None:  # Return the first scan group
        scan_groups.append(scan_groups_file[0])
    else:
        if isinstance(scan_group_names, str):
            scan_group_names = [scan_group_names]
        for desired_scan in scan_group_names:
            scan_is_here = False
            for scan in scan_groups_file:
                if desired_scan == scan.name.lstrip("/"):
                    scan_groups.append(scan)
                    scan_is_here = True
                    break
            if not scan_is_here:
                error_str = (
                    f"Scan '{desired_scan}' is not among the available scans "
                    f"{scan_groups_file} in '{file.name}'."
                )
                if len(scan_group_names) == 1:
                    raise IOError(error_str)
                else:
                    warnings.warn(error_str)

    return scan_groups


def hdf5group2dict(
    group: h5py.Group,
    dictionary: Union[None, dict, DictionaryTreeBrowser] = None,
    recursive: bool = False,
    data_dset_names: Optional[list] = None,
    **kwargs,
) -> dict:
    """Return a dictionary with values from datasets in a group in an
    opened HDF5 file.

    Parameters
    ----------
    group : h5py:Group
        HDF group object.
    dictionary
        To fill dataset values into.
    recursive
        Whether to add subgroups to dictionary.
    data_dset_names
        List of names of HDF data sets with data to not read.

    Returns
    -------
    dictionary : dict
        Dataset values in group (and subgroups if recursive=True).
    """
    if "lazy" in kwargs.keys():
        warnings.warn(
            "The 'lazy' parameter is not used in this method. Passing it will "
            "raise an error from v0.3.",
            VisibleDeprecationWarning,
        )

    if data_dset_names is None:
        data_dset_names = []
    if dictionary is None:
        dictionary = {}
    for key, val in group.items():
        # Prepare value for entry in dictionary
        if isinstance(val, h5py.Dataset):
            if key not in data_dset_names:
                val = val[()]
            if isinstance(val, np.ndarray) and len(val) == 1:
                val = val[0]
                key = key.lstrip()  # EDAX has some leading whitespaces
            if isinstance(val, bytes):
                val = val.decode("latin-1")
        # Check whether to extract subgroup or write value to dictionary
        if isinstance(val, h5py.Group) and recursive:
            dictionary[key] = {}
            hdf5group2dict(
                group=group[key],
                dictionary=dictionary[key],
                data_dset_names=data_dset_names,
                recursive=recursive,
            )
        elif key in data_dset_names:
            pass
        else:
            dictionary[key] = val
    return dictionary


def h5ebsd2signaldict(
    scan_group: h5py.Group, manufacturer: str, version: str, lazy: bool = False,
) -> dict:
    """Return a dictionary with ``signal``, ``metadata`` and
    ``original_metadata`` from an h5ebsd scan.

    Parameters
    ----------
    scan_group : h5py:Group
        HDF group of scan.
    manufacturer
        Manufacturer of file. Options are
        "kikuchipy"/"EDAX"/"Bruker Nano".
    version
        Version of manufacturer software.
    lazy
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
        # Data is stored image by image
        pw = [(0, ny * nx * sy * sx - data.size)]
        if lazy:
            data = da.pad(data.flatten(), pw, mode="constant")
        else:
            data = np.pad(data.flatten(), pw, mode="constant")
        data = data.reshape((ny, nx, sy, sx))
    scan["data"] = data

    units = ["um"] * 4
    scales = np.ones(4)
    # Calibrate scan dimension and detector dimension
    scales[0] *= scan_size.step_y
    scales[1] *= scan_size.step_x
    scales[2] *= scan_size.delta
    scales[3] *= scan_size.delta
    # Set axes names
    names = ["y", "x", "dy", "dx"]
    if data.ndim == 3:
        if ny > nx:
            names.remove("x")
            scales = np.delete(scales, 1)
        else:
            names.remove("y")
            scales = np.delete(scales, 0)
    elif data.ndim == 2:
        names = names[2:]
        scales = scales[2:]

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


def h5ebsdheader2dicts(
    scan_group: h5py.Group, manufacturer: str, version: str, lazy: bool = False
) -> Tuple[DictionaryTreeBrowser, DictionaryTreeBrowser, DictionaryTreeBrowser]:
    """Return three dictionaries in HyperSpy's
    :class:`hyperspy.misc.utils.DictionaryTreeBrowser` format, one
    with the h5ebsd scan header parameters as kikuchipy metadata,
    another with all datasets in the header as original metadata, and
    the last with info about scan size, image size and detector pixel
    size.

    Parameters
    ----------
    scan_group : h5py:Group
        HDF group of scan data and header.
    manufacturer
        Manufacturer of file. Options are
        "kikuchipy"/"EDAX"/"Bruker Nano"
    version
        Version of manufacturer software used to create file.
    lazy
        Read dataset lazily.

    Returns
    -------
    md
        kikuchipy ``metadata`` elements available in file.
    omd
        All metadata available in file.
    scan_size
        Scan, image, step and detector pixel size available in file.
    """
    md = ebsd_metadata()
    title = (
        scan_group.file.filename.split("/")[-1].split(".")[0]
        + " "
        + scan_group.name[1:].split("/")[0]
    )
    if len(title) > 20:
        title = "{:.20}...".format(title)
    md.set_item("General.title", title)

    if "edax" in manufacturer.lower():
        md, omd, scan_size = edaxheader2dicts(scan_group, md)
    elif "bruker" in manufacturer.lower():
        md, omd, scan_size = brukerheader2dicts(scan_group, md)
    else:  # kikuchipy
        md, omd, scan_size = kikuchipyheader2dicts(scan_group, md)

    ebsd_node = metadata_nodes("ebsd")
    md.set_item(ebsd_node + ".manufacturer", manufacturer)
    md.set_item(ebsd_node + ".version", version)

    return md, omd, scan_size


def kikuchipyheader2dicts(
    scan_group: h5py.Group, md: DictionaryTreeBrowser
) -> Tuple[DictionaryTreeBrowser, DictionaryTreeBrowser, DictionaryTreeBrowser]:
    """Return scan metadata dictionaries from a kikuchipy h5ebsd file.

    Parameters
    ----------
    scan_group : h5py:Group
        HDF group of scan data and header.
    md
        Dictionary with empty fields from kikuchipy's metadata.

    Returns
    -------
    md
        kikuchipy ``metadata`` elements available in kikuchipy file.
    omd
        All metadata available in kikuchipy file.
    scan_size
        Scan, image, step and detector pixel size available in
        kikuchipy file.
    """
    # Data sets to not read via hdf5group2dict
    pattern_dset_names = list(manufacturer_pattern_names().values())

    omd = DictionaryTreeBrowser()
    sem_node, ebsd_node = metadata_nodes(["sem", "ebsd"])
    md.set_item(
        ebsd_node,
        hdf5group2dict(
            group=scan_group["EBSD/Header"], data_dset_names=pattern_dset_names,
        ),
    )
    md = _delete_from_nested_dictionary(md, "Phases")
    phase_node = "Sample.Phases"
    md.set_item(
        sem_node,
        hdf5group2dict(
            group=scan_group["SEM/Header"], data_dset_names=pattern_dset_names,
        ),
    )
    md.set_item(
        phase_node,
        hdf5group2dict(
            group=scan_group["EBSD/Header/Phases"],
            data_dset_names=pattern_dset_names,
            recursive=True,
        ),
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


def edaxheader2dicts(
    scan_group: h5py.Group, md: DictionaryTreeBrowser
) -> Tuple[DictionaryTreeBrowser, DictionaryTreeBrowser, DictionaryTreeBrowser]:
    """Return scan metadata dictionaries from an EDAX TSL h5ebsd file.

    Parameters
    ----------
    scan_group : h5py:Group
        HDF group of scan data and header.
    md
        Dictionary with empty fields from kikuchipy's metadata.

    Returns
    -------
    md
        kikuchipy ``metadata`` elements available in EDAX file.
    omd
        All metadata available in EDAX file.
    scan_size
        Scan, image, step and detector pixel size available in EDAX
        file.
    """
    # Get header group as dictionary
    pattern_dset_names = list(manufacturer_pattern_names().values())
    hd = hdf5group2dict(
        group=scan_group["EBSD/Header"],
        data_dset_names=pattern_dset_names,
        recursive=True,
    )

    # Populate metadata dictionary
    sem_node, ebsd_node = metadata_nodes(["sem", "ebsd"])
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


def brukerheader2dicts(
    scan_group: h5py.Group, md: DictionaryTreeBrowser
) -> Tuple[DictionaryTreeBrowser, DictionaryTreeBrowser, DictionaryTreeBrowser]:
    """Return scan metadata dictionaries from a Bruker h5ebsd file.

    Parameters
    ----------
    scan_group : h5py:Group
        HDF group of scan data and header.
    md : hyperspy.misc.utils.DictionaryTreeBrowser
        Dictionary with empty fields from kikuchipy's metadata.

    Returns
    -------
    md
        kikuchipy ``metadata`` elements available in Bruker file.
    omd
        All metadata available in Bruker file.
    scan_size
        Scan, image, step and detector pixel size available in Bruker
        file.
    """
    # Get header group and data group as dictionaries
    pattern_dset_names = list(manufacturer_pattern_names().values())
    hd = hdf5group2dict(
        group=scan_group["EBSD/Header"],
        data_dset_names=pattern_dset_names,
        recursive=True,
    )
    dd = hdf5group2dict(
        group=scan_group["EBSD/Data"], data_dset_names=pattern_dset_names,
    )

    # Populate metadata dictionary
    sem_node, ebsd_node = metadata_nodes(["sem", "ebsd"])
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


def file_writer(
    filename: str,
    signal,
    add_scan: Optional[bool] = None,
    scan_number: int = 1,
    **kwargs,
):
    """Write an :class:`~kikuchipy.signals.EBSD` or
    :class:`~kikuchipy.signals.LazyEBSD` signal to an existing,
    but not open, or new h5ebsd file.

    Only writing to kikuchipy's h5ebsd format is supported.

    Parameters
    ----------
    filename
        Full path of HDF file.
    signal : kikuchipy.signals.EBSD or kikuchipy.signals.LazyEBSD
        Signal instance.
    add_scan
        Add signal to an existing, but not open, h5ebsd file. If it does
        not exist it is created and the signal is written to it.
    scan_number
        Scan number in name of HDF dataset when writing to an existing,
        but not open, h5ebsd file.
    kwargs
        Keyword arguments passed to :meth:`h5py:Group.require_dataset`.
    """
    # Set manufacturer and version to use in file
    from kikuchipy.release import version as ver_signal

    man_ver_dict = {"manufacturer": "kikuchipy", "version": ver_signal}

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
                f"Only writing to kikuchipy's (and not {man_file}'s) h5ebsd "
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
    # Add scan size, image size and detector pixel size to dictionary to write
    data_shape = [1] * 4  # (ny, nx, sy, sx)
    data_scales = [1] * 4  # (y, x, dy, dx)
    nav_extent = [0, 1, 0, 1]  # (x0, x1, y0, y1)
    am = signal.axes_manager
    nav_axes = am.navigation_axes
    nav_dim = am.navigation_dimension
    if nav_dim == 1:
        nav_axis = nav_axes[0]
        if nav_axis.name == "y":
            data_shape[0] = nav_axis.size
            data_scales[0] = nav_axis.scale
            nav_extent[2:] = am.navigation_extent
        else:  # nav_axis.name == "x" or something else
            data_shape[1] = nav_axis.size
            data_scales[1] = nav_axis.scale
            nav_extent[:2] = am.navigation_extent
    elif nav_dim == 2:
        data_shape[:2] = [i.size for i in nav_axes][::-1]
        data_scales[:2] = [i.scale for i in nav_axes][::-1]
        nav_extent = am.navigation_extent
    data_shape[2:] = am.signal_shape
    data_scales[2:] = [i.scale for i in am.signal_axes]
    ny, nx, sy, sx = data_shape
    scale_ny, scale_nx, scale_sy, _ = data_scales
    md = signal.metadata.deepcopy()
    sem_node, ebsd_node = metadata_nodes(["sem", "ebsd"])
    md.set_item(ebsd_node + ".pattern_width", sx)
    md.set_item(ebsd_node + ".pattern_height", sy)
    md.set_item(ebsd_node + ".n_columns", nx)
    md.set_item(ebsd_node + ".n_rows", ny)
    md.set_item(ebsd_node + ".step_x", scale_nx)
    md.set_item(ebsd_node + ".step_y", scale_ny)
    md.set_item(ebsd_node + ".detector_pixel_size", scale_sy)
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
    dset_pattern_name = man_pats["kikuchipy"]
    overwrite_dataset(
        scan_group.create_group("EBSD/Data"),
        signal.data.reshape(nx * ny, sy, sx),
        dset_pattern_name,
        signal_axes=(2, 1),
        **kwargs,
    )
    nx_start, nx_stop, ny_start, ny_stop = nav_extent
    sample_pos = {
        "y_sample": np.tile(np.linspace(ny_start, ny_stop, ny), nx),
        "x_sample": np.tile(np.linspace(nx_start, nx_stop, nx), ny),
    }
    dict2h5ebsdgroup(sample_pos, scan_group["EBSD/Data"])

    f.close()


def dict2h5ebsdgroup(dictionary: dict, group: h5py.Group, **kwargs):
    """Write a dictionary from ``metadata`` to datasets in a new group
    in an opened HDF file in the h5ebsd format.

    Parameters
    ----------
    dictionary
        ``Metadata``, with keys as dataset names.
    group : h5py:Group
        HDF group to write dictionary to.
    kwargs
        Keyword arguments passed to :meth:`h5py:Group.require_dataset`.
    """
    for key, val in dictionary.items():
        ddtype = type(val)
        dshape = (1,)
        if isinstance(val, (dict, DictionaryTreeBrowser)):
            if isinstance(val, DictionaryTreeBrowser):
                val = val.as_dictionary()
            dict2h5ebsdgroup(val, group.create_group(key), **kwargs)
            continue  # Jump to next item in dictionary
        elif isinstance(val, str):
            ddtype = "S" + str(len(val) + 1)
            val = val.encode()
        elif ddtype == np.dtype("O"):
            try:
                if isinstance(val, (np.ndarray, da.Array)):
                    ddtype = val.dtype
                else:
                    ddtype = val[0].dtype
                dshape = np.shape(val)
            except TypeError:
                warnings.warn(
                    "The hdf5 writer could not write the following information "
                    "to the file '{} : {}'.".format(key, val)
                )
                break  # or continue?
        group.create_dataset(key, shape=dshape, dtype=ddtype, **kwargs)
        group[key][()] = val
