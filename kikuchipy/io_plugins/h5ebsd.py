# -*- coding: utf-8 -*-
# Copyright 2019 The KikuchiPy developers
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

import os
import h5py
import warnings
import logging
import numpy as np
import dask.array as da
from hyperspy.misc.utils import DictionaryTreeBrowser
from hyperspy.io_plugins.hspy import overwrite_dataset, get_signal_chunks
from kikuchipy.utils.io_utils import (kikuchipy_metadata, get_input_variable,
                                      metadata_nodes)

_logger = logging.getLogger(__name__)


# Plugin characteristics
# ----------------------
format_name = 'h5ebsd'
description = 'Read/write support for electron backscatter patterns stored in '\
    'an HDF5 file formatted in the h5ebsd format introduced in '\
    'Jackson et al.: h5ebsd: an archival data format for electron'\
    'back-scatter diffraction data sets. Integrating Materials and'\
    'Manufacturing Innovation 2014 3:4, doi: '\
    'https://dx.doi.org/10.1186/2193-9772-3-4.'
full_support = False
# Recognised file extension
file_extensions = ['h5', 'hdf5', 'h5ebsd']
default_extension = 0
# Writing capabilities
writes = [(2, 2), (2, 1), (2, 0)]


def file_reader(filename, scans=None, lazy=False, **kwargs):
    """Read electron backscatter patterns from an h5ebsd file [1]_. A
    valid h5ebsd file has at least one group with the name
    '/Scan x/EBSD' with the groups 'Data' (patterns etc.) and 'Header'
    (metadata etc.) , where 'x' is the scan_number.

    Parameters
    ----------
    filename : str
        Full file path of the HDF file.
    scans : list of ints
        List of scan numbers to return. If None is passed the first
        scan in the file is returned.
    lazy : bool, optional

    Returns
    -------
    dictionary : dict
        Data, axes, metadata and original metadata.

    References
    ----------
    .. [1] Jackson et al.: h5ebsd: an archival data format for electron
           back-scatter diffraction data sets. Integrating Materials and
           Manufacturing Innovation 2014 3:4, doi:
           https://dx.doi.org/10.1186/2193-9772-3-4.
    """
    mode = kwargs.pop('mode', 'r')
    f = h5py.File(filename, mode=mode, **kwargs)

    # Check if h5ebsd file
    check_h5ebsd(f)

    # Get manufacturer and version and check if reading the file is supported
    man, ver = manufacturer_version(f)
    man_pats = manufacturer_pattern_names()
    if any(man == s for s in man_pats.keys()) is not True:
        raise IOError("Manufacturer {} not among recognised manufacturers {}."
                      "".format(man, list(man_pats.keys())))

    # Get scans to return
    scans_file = [f[k] for k in f['/'].keys() if 'Scan' in k]
    scans_return = []
    if scans is None:  # Return first scan
        scans_return.append(scans_file[0])
    else:
        if isinstance(scans, int):
            scans = [scans, ]
        for scan_no in scans:  # Wanted scans
            scan_is_here = False
            for scan in scans_file:
                if scan_no == int(scan.name.split()[-1]):
                    scans_return.append(scan)
                    scan_is_here = True
                    break
            if not scan_is_here:
                scan_nos = [int(i.name.split()[-1]) for i in scans_file]
                error_str = ("Scan {} is not among the available scans {} "
                             "in '{}'".format(scan_no, scan_nos, filename))
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
    file : h5py.File
        File where manufacturer, version and scan datasets should
        reside in the top group.
    """
    file_keys_lower = [key.lstrip().lower() for key in file['/'].keys()]
    if not any(s in file_keys_lower for s in ['manufacturer', 'version']):
        raise IOError("'{}' is not an h5ebsd file, as manufacturer and/or "
                      "version could not be read from its top group."
                      "".format(file.filename))

    if not any('Scan' in key and 'EBSD/Data' in file[key]
               and 'EBSD/Header' in file[key] for key in file['/'].keys()):
        raise IOError("'{}' is not an h5ebsd file, as no scans in a group "
                      "with name 'Scan <scan_number>/EBSD' with groups 'Data' "
                      "and 'Header' could be read.".format(file.filename))


def manufacturer_version(file):
    """Get manufacturer and version from h5ebsd file.

    Parameters
    ----------
    file : h5py.File
        File with manufacturer and version datasets in the top group.

    Returns
    -------
    manufacturer, version : str
    """
    manufacturer = None
    version = None
    for key, val in h5ebsdgroup2dict(file['/']).items():
        if key.lower() == 'manufacturer':
            manufacturer = val
        if key.lower() == 'version':
            version = val
    return manufacturer, version


def manufacturer_pattern_names():
    """Return mapping of string of supported manufacturers to the names
    of their HDF dataset where the patterns are stored.

    Returns
    -------
    dict
    """
    return {'KikuchiPy': 'patterns', 'EDAX': 'Pattern',
            'Bruker Nano': 'RawPatterns'}


def h5ebsdgroup2dict(group, dictionary=None, recursive=False,
                     lazy=False):
    """Return a dictionary with values from datasets in a group in an
    opened HDF5 file.

    Parameters
    ----------
    group : h5py.Group
        HDF group object.
    dictionary : {dict, DictionaryTreeBrowser, None}, optional
        To fill dataset values into.
    recursive : bool, optional
        Whether to add subgroups to dictionary.
    lazy : bool, optional
        Read dataset lazily.

    Returns
    -------
    dictionary : dict
        Dataset values in group (and subgroups if `recursive` is True).
    """
    man_pats = manufacturer_pattern_names()
    if dictionary is None:
        dictionary = {}
    elif isinstance(dictionary, DictionaryTreeBrowser):
        dictionary = dictionary.as_dictionary()
    if not isinstance(group, h5py.Group):
        return dictionary
    for key, val in group.items():
        # Prepare value for entry in dictionary
        if isinstance(val, h5py.Dataset):
            if key not in man_pats.values():
                val = val[()]
            if isinstance(val, np.ndarray) and len(val) == 1:
                val = val[0]
                key = key.lstrip()  # TSL has some leading whitespaces
            if val.dtype.char == 'S':
                val = val.decode()
        # Check whether to extract subgroup or write value to dictionary
        if isinstance(val, h5py.Group) and recursive:
            dictionary[key] = {}
            h5ebsdgroup2dict(group[key], dictionary[key], recursive=recursive,
                             lazy=lazy)
        else:
            dictionary[key] = val
    return dictionary


def h5ebsd2signaldict(scan_group, manufacturer, version, lazy=False):
    """Return a dictionary with signal, metadata and original metadata
    from an h5ebsd scan.

    Parameters
    ----------
    scan_group : h5py.Group
        HDF group of scan.
    manufacturer : {'KikuchiPy', 'EDAX', 'Bruker Nano'}
        Manufacturer of file.
    version : str
        Version of manufacturer software.
    lazy : bool, optional

    Returns
    -------
    scan : dict
        Dictionary with patterns, metadata and original metadata.
    """
    ebsd_node = metadata_nodes(sem=False)
    md, omd = h5ebsdheader2dicts(scan_group, manufacturer, version, lazy)
    nx = md.get_item(ebsd_node + '.n_columns')
    ny = md.get_item(ebsd_node + '.n_rows')
    sx = md.get_item(ebsd_node + '.pattern_width')
    sy = md.get_item(ebsd_node + '.pattern_height')
    md.set_item('Signal.signal_type', 'EBSD')
    md.set_item('Signal.record_by', 'image')

    scan = {'metadata': md.as_dictionary(),
            'original_metadata': omd.as_dictionary(), 'attributes': {}}

    # Get data group
    man_pats = manufacturer_pattern_names()
    for man, pats in man_pats.items():
        if manufacturer.lower() == man.lower():
            data = scan_group['EBSD/Data/' + pats]

    # Fetch data from group
    if lazy:
        chunks = data.chunks
        if chunks is None:
            chunks = get_signal_chunks(data.shape, data.dtype, [1, 2])
        data = da.from_array(data, chunks=chunks)
        scan['attributes']['_lazy'] = True
    else:
        data = np.asanyarray(data)

    try:
        data = data.reshape((ny, nx, sy, sx)).squeeze()
    except ValueError:
        warnings.warn("Pattern size ({} x {}) and scan size ({} x {}) larger "
                      "than file size. Will attempt to load by zero padding "
                      "incomplete frames.".format(sx, sy, nx, ny))
        # Data is stored pattern by pattern
        pw = [(0, ny * nx * sy * sx - data.size)]
        if lazy:
            data = da.pad(data, pw, mode='constant')
        else:
            data = np.pad(data, pw, mode='constant')
        data = data.reshape((ny, nx, sy, sx))
    scan['data'] = data

    units = [u'\u03BC'+'m', u'\u03BC'+'m', 'A^{-1}', 'A^{-1}']
    names = ['y', 'x', 'dy', 'dx']
    scales = np.ones(4)

    # Calibrate scan dimension
    step_x = md.get_item(ebsd_node + '.step_x')
    step_y = md.get_item(ebsd_node + '.step_y')
    if step_x == -1. or step_y == -1.:
        warnings.warn("Could not calibrate scan dimension, this can be done "
                      "using set_scan_calibration()")
    else:
        scales[0] = scales[0] * step_x
        scales[1] = scales[1] * step_y

    # Create axis objects for each axis
    axes = [{'size': data.shape[i], 'index_in_array': i, 'name': names[i],
            'scale': scales[i], 'offset': 0.0, 'units': units[i]}
            for i in range(data.ndim)]
    scan['axes'] = axes

    return scan


def h5ebsdheader2dicts(scan_group, manufacturer, version, lazy=False):
    """Return two dictionaries in HyperSpy's DictionaryTreeBrowser
    format, one with the h5ebsd scan header parameters as KikuchiPy
    metadata, the other with all datasets in the header as original
    metadata.

    Parameters
    ----------
    scan_group : h5py.Group
        HDF group of scan data and header.
    manufacturer : {'KikuchiPy', 'EDAX', 'Bruker Nano'}
        Manufacturer of file.
    version : str
        Version of manufacturer software used to create file.
    lazy : bool, optional

    Returns
    -------
    md : DictionaryTreeBrowser
    omd : DictionaryTreeBrowser
    """
    md = kikuchipy_metadata()
    title = (scan_group.file.filename.split('/')[-1].split('.')[0] + ' ' +
             scan_group.name[1:].split('/')[0])
    md.set_item('General.title', title)

    if 'edax' in manufacturer.lower():
        md, omd = tslheader2dicts(scan_group, md, lazy)
    elif 'bruker' in manufacturer.lower():
        md, omd = brukerheader2dicts(scan_group, md, lazy)
    else:  # KikuchiPy
        md, omd = kikuchipyheader2dicts(scan_group, md, lazy)

    ebsd_node = metadata_nodes(sem=False)
    md.set_item(ebsd_node + '.manufacturer', manufacturer)
    md.set_item(ebsd_node + '.version', version)

    return md, omd


def kikuchipyheader2dicts(scan_group, md, lazy=False):
    """Return metadata and original metadata as dictionaries in
    HyperSpy's `DictionaryTreeBrowser` format populated with values
    from KikuchiPy's h5ebsd implementation.

    Parameters
    ----------
    scan_group : h5py.Group
        HDF group of scan data and header.
    md : DictionaryTreeBrowser
        Dictionary with empty fields from KikuchiPy's metadata.
    lazy : bool, optional

    Returns
    -------
    md, omd : DictionaryTreeBrowser
    """
    omd = DictionaryTreeBrowser()
    sem_node, ebsd_node = metadata_nodes()
    md.set_item(ebsd_node, h5ebsdgroup2dict(scan_group['EBSD/Header'],
                                            lazy=lazy))
    md.set_item(sem_node, h5ebsdgroup2dict(scan_group['SEM/Header'], lazy=lazy))
    return md, omd


def tslheader2dicts(scan_group, md, lazy=False):
    """Return metadata and original metadata as dictionaries in
    HyperSpy's `DictionaryTreeBrowser` format populated with values
    from EDAX TSL's h5ebsd implementation.

    Parameters
    ----------
    scan_group : h5py.Group
        HDF group of scan data and header.
    md : DictionaryTreeBrowser
        Dictionary with empty fields from KikuchiPy's metadata.
    lazy : bool, optional

    Returns
    -------
    md : DictionaryTreeBrowser
    omd : DictionaryTreeBrowser
    """
    # Get all metadata from file
    hg = scan_group['EBSD/Header']  # Header group
    omd = DictionaryTreeBrowser({
        'tsl_header': h5ebsdgroup2dict(hg, recursive=True, lazy=lazy)})

    # Populate dictionary
    sem_node, ebsd_node = metadata_nodes()
    md.set_item(ebsd_node + '.azimuth_angle', hg['Camera Azimuthal Angle'][0])
    md.set_item(ebsd_node + '.elevation_angle', hg['Camera Elevation Angle'][0])
    grid_type = hg['Grid Type'][0].decode()
    if grid_type == 'SqrGrid':
        md.set_item(ebsd_node + '.grid_type', 'square')
    else:
        raise IOError("Only square grids are supported, however a {} grid was "
                      "passed".format(grid_type))
    md.set_item(ebsd_node + '.pattern_height', hg['Pattern Height'][0])
    md.set_item(ebsd_node + '.pattern_width', hg['Pattern Width'][0])
    md.set_item(ebsd_node + '.sample_tilt', hg['Sample Tilt'][0])
    md.set_item(ebsd_node + '.step_x', hg['Step X'][0])
    md.set_item(ebsd_node + '.step_y', hg['Step Y'][0])
    md.set_item(ebsd_node + '.n_rows', hg['nRows'][0])
    md.set_item(ebsd_node + '.n_columns', hg['nColumns'][0])
    md.set_item('General.authors', hg['Operator'][0].decode())
    md.set_item('General.notes', hg['Notes'][0].decode())
    md.set_item(ebsd_node + '.xpc', hg['Pattern Center Calibration/x-star'][0])
    md.set_item(ebsd_node + '.ypc', hg['Pattern Center Calibration/y-star'][0])
    md.set_item(ebsd_node + '.zpc', hg['Pattern Center Calibration/z-star'][0])
    md.set_item(sem_node + '.working_distance', hg['Working Distance'][0])
    if 'SEM-PRIAS Images' in scan_group.keys():
        md.set_item(sem_node + '.magnification',
                    scan_group['SEM-PRIAS Images/Header/Mag'][0])

    return md, omd


def brukerheader2dicts(scan_group, md, lazy=False):
    """Return metadata and original metadata as dictionaries in
    HyperSpy's `DictionaryTreeBrowser` format populated with
    values from Bruker's h5ebsd implementation.

    Parameters
    ----------
    scan_group : h5py.Group
        HDF group of scan data and header.
    md : DictionaryTreeBrowser
        Dictionary with empty fields from KikuchiPy's metadata.
    lazy : bool, optional

    Returns
    -------
    md, omd : DictionaryTreeBrowser
    """
    # Get all metadata from file
    hg = scan_group['EBSD/Header']  # Header group
    dg = scan_group['EBSD/Data']  # Data group with microscope info
    omd = DictionaryTreeBrowser({
        'bruker_header': h5ebsdgroup2dict(hg, recursive=True, lazy=lazy)})

    # Populate dictionary
    sem_node, ebsd_node = metadata_nodes()
    md.set_item(ebsd_node + '.elevation_angle', hg['CameraTilt'][()])
    grid_type = hg['Grid Type'][()].decode()
    if grid_type == 'isometric':
        md.set_item(ebsd_node + '.grid_type', 'square')
    else:
        raise IOError("Only square grids are supported, however a {} grid was "
                      "passed".format(grid_type))
    md.set_item(ebsd_node + '.pattern_height', hg['PatternHeight'][()])
    md.set_item(ebsd_node + '.pattern_width', hg['PatternWidth'][()])
    md.set_item(ebsd_node + '.sample_tilt', hg['SampleTilt'][()])
    md.set_item(ebsd_node + '.step_x', hg['XSTEP'][()])
    md.set_item(ebsd_node + '.step_y', hg['YSTEP'][()])
    md.set_item(ebsd_node + '.n_rows', hg['NROWS'][()])
    md.set_item(ebsd_node + '.n_columns', hg['NCOLS'][()])
    md.set_item(ebsd_node + '.xpc', np.mean(dg['PCX'][()]))
    md.set_item(ebsd_node + '.ypc', np.mean(dg['PCY'][()]))
    md.set_item(ebsd_node + '.zpc', np.mean(dg['DD']))
    md.set_item(ebsd_node + '.detector_pixel_size',
                hg['DetectorFullHeightMicrons'][()] /
                hg['UnClippedPatternHeight'][()])
    md.set_item(ebsd_node + '.static_background',
                hg['StaticBackground'][()][0, :, :])
    md.set_item(sem_node + '.working_distance', hg['WD'][()])
    md.set_item(sem_node + '.beam_energy', hg['KV'][()])
    md.set_item(sem_node + '.magnification', hg['Magnification'][()])

    return md, omd


def file_writer(filename, signal, add_scan=None, scan_number=1,
                **kwargs):
    """Write an EBSD signal to an existing, but not open, or new h5ebsd
    file. Only writing to KikuchiPy's h5ebsd format is supported.

    Parameters
    ----------
    filename : str
        Full path of HDF file.
    signal : {kikuchipy.signals.EBSD, kikuchipy.lazy_signals.LazyEBSD}
        Signal instance.
    add_scan : {bool, None}, optional
        Add signal to an existing, but not open, h5ebsd file. If it does
        not exist it is created and the signal is written to it.
    scan_number : int, optional
        Scan number in name of HDF dataset when writing to an existing,
        but not open, h5ebsd file.
    **kwargs :
        Keyword arguments passed to h5py.
    """
    # Set manufacturer and version to use in file
    from kikuchipy.version import __version__ as ver_signal
    man_ver_dict = {'manufacturer': 'KikuchiPy', 'version': ver_signal}

    # Open file in correct mode
    mode = 'w'
    if os.path.isfile(filename) and add_scan:
        mode = 'r+'
    try:
        f = h5py.File(filename, mode=mode)
    except OSError:
        raise OSError("Cannot write to an already open file (e.g. a file from "
                      "which data has been read lazily).")

    if os.path.isfile(filename) and add_scan:
        check_h5ebsd(f)
        man_file, ver_file = manufacturer_version(f)
        if man_ver_dict['manufacturer'].lower() != man_file.lower():
            f.close()
            raise IOError("Only writing to KikuchiPy's (and not {}'s) h5ebsd "
                          "format is supported.".format(man_file))
        man_ver_dict['version'] = ver_file

        # Get valid scan number
        scans_file = [f[k] for k in f['/'].keys() if 'Scan' in k]
        scan_nos = [int(i.name.split()[-1]) for i in scans_file]
        for i in scan_nos:
            if i == scan_number:
                q = "Scan {} already in file, enter another scan "\
                    "number:\n".format(i)
                scan_number = get_input_variable(q, int)
                if scan_number is None:
                    raise IOError("Invalid scan number.")
    else:  # File did not exist
        dict2h5ebsdgroup(man_ver_dict, f['/'], **kwargs)

    scan_group = f.create_group('Scan ' + str(scan_number))

    # Create scan dictionary with (updated) EBSD and SEM metadata
    md = signal.metadata
    nx, ny, sx, sy = signal.axes_manager.shape
    sem_node, ebsd_node = metadata_nodes()
    md.set_item(ebsd_node + '.n_columns', nx)
    md.set_item(ebsd_node + '.n_rows', ny)
    md.set_item(ebsd_node + '.pattern_width', sx)
    md.set_item(ebsd_node + '.pattern_height', sy)
    det_str, ebsd_str = ebsd_node.split('.')[-2:]
    md_sem = signal.metadata.get_item(sem_node).copy().as_dictionary()
    md_det = md_sem.pop(det_str)
    scan = {'EBSD': {'Header': md_det.pop(ebsd_str)}, 'SEM': {'Header': md_sem}}
    dict2h5ebsdgroup(scan, scan_group)

    # Write signal to file
    man_pats = manufacturer_pattern_names()
    dset_pattern_name = man_pats['KikuchiPy']
    overwrite_dataset(scan_group.create_group('EBSD/Data'),
                      signal.data.reshape(nx * ny, sx, sy),
                      dset_pattern_name, signal_axes=(2, 1), **kwargs)
    nx_start, nx_stop, ny_start, ny_stop = signal.axes_manager.navigation_extent
    sample_pos = {'x_sample': np.tile(np.linspace(nx_start, nx_stop, nx), ny),
                  'y_sample': np.tile(np.linspace(ny_start, ny_stop, ny), nx)}
    dict2h5ebsdgroup(sample_pos, scan_group['EBSD/Data'])

    f.close()
    _logger.info("File closed.")


def dict2h5ebsdgroup(dictionary, group, **kwargs):
    """Write a dictionary from metadata to datasets in a new group in an
    opened HDF file in the h5ebsd format.

    Parameters
    ----------
    dictionary : dict
        Metadata, with keys as dataset names.
    group : h5py.Group
        HDF group to write dictionary to.
    **kwargs :
        Keyword arguments passed to h5py.
    """
    for key, val in dictionary.items():
        ddtype = type(val)
        dshape = (1, )
        written = False
        if isinstance(val, (dict, DictionaryTreeBrowser)):
            if isinstance(val, DictionaryTreeBrowser):
                val = val.as_dictionary()
            dict2h5ebsdgroup(val, group.create_group(key), **kwargs)
            written = True
        elif isinstance(val, str):
            ddtype = 'S' + str(len(val) + 1)
            val = val.encode()
        elif isinstance(val, (np.ndarray, da.Array)):
            overwrite_dataset(group, val, key, **kwargs)
            written = True
        elif ddtype == np.dtype('O'):
            ddtype = h5py.special_dtype(vlen=val[0].dtype)
            dshape = np.shape(val)

        if written:
            continue  # Jump to next item in dictionary
        try:
            group.create_dataset(key, shape=dshape, dtype=ddtype, **kwargs)
            group[key][()] = val
        except (TypeError, IndexError):
            warnings.warn("The hdf5 writer could not write the following "
                          "information to the file '{} : {}'.".format(key, val))
