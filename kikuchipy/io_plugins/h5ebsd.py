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
                                      metadata_nodes, phase_metadata)

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
    dictionary
    """

    return {'KikuchiPy': 'patterns', 'EDAX': 'Pattern',
            'Bruker Nano': 'RawPatterns'}


def h5ebsdgroup2dict(group, dictionary=None, recursive=False,
                     lazy=False):
    """Return a dictionary with values from datasets in a group in an
    opened h5ebsd file.

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
    dictionary : dictionary
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
    scan : dictionary
        Dictionary with patterns, metadata and original metadata.
    """

    md, omd, scan_size = h5ebsdheader2dicts(scan_group, manufacturer, version,
                                            lazy)
    md.set_item('Signal.signal_type', 'EBSD')
    md.set_item('Signal.record_by', 'image')

    scan = {'metadata': md.as_dictionary(),
            'original_metadata': omd.as_dictionary(), 'attributes': {}}

    # Get data group
    man_pats = manufacturer_pattern_names()
    for man, pats in man_pats.items():
        if manufacturer.lower() == man.lower():
            data = scan_group['EBSD/Data/' + pats]

    # Get data from group
    if lazy:
        chunks = data.chunks
        if chunks is None:
            chunks = get_signal_chunks(data.shape, data.dtype, [1, 2])
        data = da.from_array(data, chunks=chunks)
        scan['attributes']['_lazy'] = True
    else:
        data = np.asanyarray(data)

    sx, sy = scan_size.sx, scan_size.sy
    nx, ny = scan_size.nx, scan_size.ny
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

    units = np.repeat(u'\u03BC'+'m', 4)
    names = ['y', 'x', 'dy', 'dx']
    scales = np.ones(4)

    # Calibrate scan dimension and detector dimension
    step_x, step_y = scan_size.step_x, scan_size.step_y
    scales[0] = scales[0] * step_x
    scales[1] = scales[1] * step_y
    detector_pixel_size = scan_size.delta
    scales[2] = scales[2] * detector_pixel_size
    scales[3] = scales[3] * detector_pixel_size

    # Create axis objects for each axis
    axes = [{'size': data.shape[i], 'index_in_array': i, 'name': names[i],
            'scale': scales[i], 'offset': 0.0, 'units': units[i]}
            for i in range(data.ndim)]
    scan['axes'] = axes

    return scan


def h5ebsdheader2dicts(scan_group, manufacturer, version, lazy=False):
    """Return three dictionaries in HyperSpy's DictionaryTreeBrowser
    format, one with the h5ebsd scan header parameters as KikuchiPy
    metadata, another with all datasets in the header as original
    metadata, and the last with info about scan size, pattern size and
    detector pixel size.

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
    md, omd, scan_size : DictionaryTreeBrowser
    """

    md = kikuchipy_metadata()
    title = (scan_group.file.filename.split('/')[-1].split('.')[0] + ' ' +
             scan_group.name[1:].split('/')[0])
    md.set_item('General.title', title)

    if 'edax' in manufacturer.lower():
        md, omd, scan_size = tslheader2dicts(scan_group, md)
    elif 'bruker' in manufacturer.lower():
        md, omd, scan_size = brukerheader2dicts(scan_group, md)
    else:  # KikuchiPy
        md, omd, scan_size = kikuchipyheader2dicts(scan_group, md, lazy)

    ebsd_node = metadata_nodes(sem=False)
    md.set_item(ebsd_node + '.manufacturer', manufacturer)
    md.set_item(ebsd_node + '.version', version)

    return md, omd, scan_size


def kikuchipyheader2dicts(scan_group, md, lazy=False):
    """Return scan metadata dictionaries from a KikuchiPy h5ebsd file.

    Parameters
    ----------
    scan_group : h5py.Group
        HDF group of scan data and header.
    md : DictionaryTreeBrowser
        Dictionary with empty fields from KikuchiPy's metadata.
    lazy : bool, optional

    Returns
    -------
    md, omd, scan_size : DictionaryTreeBrowser
    """

    omd = DictionaryTreeBrowser()
    sem_node, ebsd_node = metadata_nodes()
    md.set_item(ebsd_node, h5ebsdgroup2dict(scan_group['EBSD/Header'],
                                            lazy=lazy))
    md.set_item(sem_node, h5ebsdgroup2dict(scan_group['SEM/Header'], lazy=lazy))

    # Remove scan info values from metadata
    scan_size = DictionaryTreeBrowser()
    enl = ebsd_node.split('.')
    md = md.as_dictionary()
    md_ebsd = md[enl[0]][enl[1]][enl[2]][enl[3]]
    scan_size.set_item('sx', md_ebsd.pop('pattern_width'))
    scan_size.set_item('sy', md_ebsd.pop('pattern_height'))
    scan_size.set_item('nx', md_ebsd.pop('n_columns'))
    scan_size.set_item('ny', md_ebsd.pop('n_rows'))
    scan_size.set_item('step_x', md_ebsd.pop('step_x'))
    scan_size.set_item('step_y', md_ebsd.pop('step_y'))
    scan_size.set_item('delta', md_ebsd.pop('detector_pixel_size'))

    return DictionaryTreeBrowser(md), omd, scan_size


def tslheader2dicts(scan_group, md):
    """Return scan metadata dictionaries from an EDAX TSL h5ebsd file.

    Parameters
    ----------
    scan_group : h5py.Group
        HDF group of scan data and header.
    md : DictionaryTreeBrowser
        Dictionary with empty fields from KikuchiPy's metadata.

    Returns
    -------
    md, omd, scan_size : DictionaryTreeBrowser
    """

    # Get header group as dictionary
    hd = h5ebsdgroup2dict(scan_group['EBSD/Header'], recursive=True)

    # Populate metadata dictionary
    sem_node, ebsd_node = metadata_nodes()
    md.set_item(ebsd_node + '.azimuth_angle', hd['Camera Azimuthal Angle'])
    md.set_item(ebsd_node + '.elevation_angle', hd['Camera Elevation Angle'])
    grid_type = hd['Grid Type']
    if grid_type == 'SqrGrid':
        md.set_item(ebsd_node + '.grid_type', 'square')
    else:
        raise IOError("Only square grids are supported, however a {} grid was "
                      "passed".format(grid_type))
    md.set_item(ebsd_node + '.sample_tilt', hd['Sample Tilt'])
    md.set_item('General.authors', hd['Operator'])
    md.set_item('General.notes', hd['Notes'])
    md.set_item(ebsd_node + '.xpc', hd['Pattern Center Calibration']['x-star'])
    md.set_item(ebsd_node + '.ypc', hd['Pattern Center Calibration']['y-star'])
    md.set_item(ebsd_node + '.zpc', hd['Pattern Center Calibration']['z-star'])
    md.set_item(sem_node + '.working_distance', hd['Working Distance'])
    if 'SEM-PRIAS Images' in scan_group.keys():
        md.set_item(sem_node + '.magnification',
                    scan_group['SEM-PRIAS Images/Header/Mag'][0])
    # Loop over phases in group
    md.add_node('Sample.Phases')
    for phase_no, phase in hd['Phase'].items():
        phase['material_name'] = phase['MaterialName']
        [phase.pop(i) for i in ['hkl Families', 'NumberFamilies',
                                'MaterialName']]
        phase = {key.lower().replace(' ', '_'): value
                 for key, value in phase.items()}
        pmd = phase_metadata()
        pmd.update(phase)  # Overwrite default values
        md.Sample.Phases.add_dictionary({phase_no: pmd})

    # Populate original metadata dictionary
    omd = DictionaryTreeBrowser({'tsl_header': hd})

    # Populate scan size dictionary
    scan_size = DictionaryTreeBrowser()
    scan_size.set_item('sx', hd['Pattern Width'])
    scan_size.set_item('sy', hd['Pattern Height'])
    scan_size.set_item('nx', hd['nColumns'])
    scan_size.set_item('ny', hd['nRows'])
    scan_size.set_item('step_x', hd['Step X'])
    scan_size.set_item('step_y', hd['Step Y'])
    scan_size.set_item('delta', 1.)

    return md, omd, scan_size


def brukerheader2dicts(scan_group, md):
    """Return scan metadata dictionaries from a Bruker h5ebsd file.

    Parameters
    ----------
    scan_group : h5py.Group
        HDF group of scan data and header.
    md : DictionaryTreeBrowser
        Dictionary with empty fields from KikuchiPy's metadata.

    Returns
    -------
    md, omd, scan_size : DictionaryTreeBrowser
    """

    # Get header group and data group as dictionaries
    hd = h5ebsdgroup2dict(scan_group['EBSD/Header'], recursive=True)
    dd = h5ebsdgroup2dict(scan_group['EBSD/Data'])

    # Populate metadata dictionary
    sem_node, ebsd_node = metadata_nodes()
    md.set_item(ebsd_node + '.elevation_angle', hd['CameraTilt'])
    grid_type = hd['Grid Type']
    if grid_type == 'isometric':
        md.set_item(ebsd_node + '.grid_type', 'square')
    else:
        raise IOError("Only square grids are supported, however a {} grid was "
                      "passed".format(grid_type))
    md.set_item(ebsd_node + '.sample_tilt', hd['SampleTilt'])
    md.set_item(ebsd_node + '.xpc', np.mean(dd['PCX']))
    md.set_item(ebsd_node + '.ypc', np.mean(dd['PCY']))
    md.set_item(ebsd_node + '.zpc', np.mean(dd['DD']))
    md.set_item(ebsd_node + '.static_background', hd['StaticBackground'])
    md.set_item(sem_node + '.working_distance', hd['WD'])
    md.set_item(sem_node + '.beam_energy', hd['KV'])
    md.set_item(sem_node + '.magnification', hd['Magnification'])
    # Loop over phases
    md.add_node('Sample.Phases')
    for phase_no, phase in hd['Phases'].items():
        pmd = phase_metadata()
        pmd['material_name'] = phase['Name']
        pmd['setting'] = phase['Setting']
        pmd['formula'] = phase['Formula']
        pmd['space_group'] = phase['IT']
        (pmd['lattice_constant_a'], pmd['lattice_constant_b'],
         pmd['lattice_constant_c'], pmd['lattice_constant_alpha'],
         pmd['lattice_constant_beta'],
         pmd['lattice_constant_gamma']) = phase['LatticeConstants']
        atom_keys = list(pmd['atom_positions']['1'].keys())
        pmd['atom_positions'].pop('1')
        for atom_no, atom in phase['AtomPositions'].items():
            atom = atom.split(',')  # Make list from string
            atom[1:] = list(map(float, atom[1:]))  # Numbers as float, not str
            pmd['atom_positions'][atom_no] = {atom_keys[i]: atom[i]
                                              for i in range(len(atom_keys))}
        md.Sample.Phases.add_dictionary({phase_no: pmd})

    # Populate original metadata dictionary
    omd = DictionaryTreeBrowser({'bruker_header': hd})

    # Populate scan size dictionary
    scan_size = DictionaryTreeBrowser()
    scan_size.set_item('sx', hd['PatternWidth'])
    scan_size.set_item('sy', hd['PatternHeight'])
    scan_size.set_item('nx', hd['NCOLS'])
    scan_size.set_item('ny', hd['NROWS'])
    scan_size.set_item('step_x', hd['XSTEP'])
    scan_size.set_item('step_y', hd['YSTEP'])
    scan_size.set_item('delta', hd['DetectorFullHeightMicrons']
                       / hd['UnClippedPatternHeight'])

    return md, omd, scan_size


def file_writer(filename, signal, add_scan=None, scan_number=1,
                **kwargs):
    """Write an EBSD signal to an existing, but not open, or new h5ebsd
    file. Only writing to KikuchiPy's h5ebsd format is supported.

    Parameters
    ----------
    filename : str
        Full path of HDF file.
    signal : kikuchipy.signals.EBSD or kikuchipy.lazy_signals.LazyEBSD
        Signal instance.
    add_scan : bool or None, optional
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

    # Create scan dictionary with EBSD and SEM metadata
    # Add scan size, pattern size and detector pixel size to dictionary to write
    sx, sy = signal.axes_manager.signal_shape
    nx, ny = signal.axes_manager.navigation_shape
    nav_indices = signal.axes_manager.navigation_indices_in_array
    sig_indices = signal.axes_manager.signal_indices_in_array
    md = signal.metadata.deepcopy()
    sem_node, ebsd_node = metadata_nodes()
    md.set_item(ebsd_node + '.pattern_width', sx)
    md.set_item(ebsd_node + '.pattern_height', sy)
    md.set_item(ebsd_node + '.n_columns', nx)
    md.set_item(ebsd_node + '.n_rows', ny)
    md.set_item(ebsd_node + '.step_x',
                signal.axes_manager[nav_indices[0]].scale)
    md.set_item(ebsd_node + '.step_y',
                signal.axes_manager[nav_indices[1]].scale)
    md.set_item(ebsd_node + '.detector_pixel_size',
                signal.axes_manager[sig_indices[1]].scale)
    # Separate EBSD and SEM metadata
    det_str, ebsd_str = ebsd_node.split('.')[-2:]  # Detector and EBSD nodes
    md_sem = md.get_item(sem_node).copy().as_dictionary()  # SEM node as dict
    md_det = md_sem.pop(det_str)  # Remove/assign detector node from SEM node
    md_ebsd = md_det.pop(ebsd_str)
    md_ebsd['Phases'] = md.Sample.Phases.as_dictionary()  # Phases in metadata
    scan = {'EBSD': {'Header': md_ebsd}, 'SEM': {'Header': md_sem}}

    # Write scan dictionary to HDF groups
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
        print(val)
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
