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

import h5py
import warnings
import numpy as np
import dask.array as da
from hyperspy.misc.utils import DictionaryTreeBrowser
from hyperspy.io_plugins.hspy import hdfgroup2dict
from kikuchipy.utils.io_utils import kikuchipy_metadata


# Plugin characteristics
# ----------------------
format_name = 'H5EBSD'
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

# DictionaryTreeBrowser nodes used by multiple functions
SEM_str = 'Acquisition_instrument.SEM.'
EBSD_str = SEM_str + 'Detector.EBSD.'


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
    mode = kwargs.pop('mode', 'r+')
    f = h5py.File(filename, mode=mode, **kwargs)

    # Get manufacturer and version
    dsets = ['manufacturer', 'version']
    dsets_dict = hdfdataset2dict(f['/'], dsets)
    try:
        manufacturer = dsets_dict[dsets[0]]
    except KeyError:
        raise IOError("This is not a valid h5ebsd file, as the file "
                      "manufacturer could not be read from a dataset with name "
                      "\'Manufacturer\' from the top group of file "
                      "'{}'".format(filename))
    try:
        version = dsets_dict[dsets[1]]
    except KeyError:
        version = ''

    # Get groups with valid h5ebsd scans
    scans_in_file = []
    for key in list(f.keys()):
        if 'Scan' in key and f.require_group(key + '/EBSD/Data') and \
                f.require_group(key + '/EBSD/Header'):
            scans_in_file.append(f[key])
    if not len(scans_in_file):
        raise IOError("This is not a valid h5ebsd file, as no scans within a "
                      "group with name \'/Scan x/EBSD\' with groups \'Data\' "
                      "and \'Header\', where \'x\' is the scan number, could be"
                      " read. You can still load the data using an hdf5 reader,"
                      " e.g. h5py, and manually creating an EBSD signal.")

    # Get scans to return
    scans_to_return = []
    if scans is None:  # Return first scan
        scans_to_return.append(scans_in_file[0])
    else:
        for scan_no in scans:  # Wanted scans
            scan_is_here = False
            for scan in scans_in_file:  # Scans in file
                if scan_no == int(scan.name.split()[-1]):
                    scans_to_return.append(scan)
                    scan_is_here = True
                    break
            if not scan_is_here:
                scan_nos = [int(i.name.split()[-1]) for i in scans_in_file]
                warnings.warn("Scan {} is not among the available scans {} in "
                              "'{}'".format(scan_no, scan_nos, filename))

    # Parse file
    scan_dict_list = []
    for scan in scans_to_return:
        scan_dict_list.append(h5ebsd2signaldict(scan, manufacturer=manufacturer,
                                                version=version, lazy=lazy))

    if not lazy:
        f.close()

    return scan_dict_list


def h5ebsd2signaldict(scan_group, manufacturer, version, lazy=False):
    """Return a dictionary with signal, metadata and original metadata
    from an h5ebsd dataset.

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
    md, omd = h5ebsdheader2dicts(scan_group, manufacturer, version, lazy)
    nx = md.get_item(EBSD_str + 'n_columns')
    ny = md.get_item(EBSD_str + 'n_rows')
    sx = md.get_item(EBSD_str + 'pattern_width')
    sy = md.get_item(EBSD_str + 'pattern_height')
    md.set_item('Signal.signal_type', 'EBSD')
    md.set_item('Signal.record_by', 'image')

    scan = {'metadata': md.as_dictionary(),
            'original_metadata': omd.as_dictionary(), 'attributes': {}}

    if manufacturer.lower() in ['edax', 'kikuchipy']:
        data = scan_group['EBSD/Data/Pattern']
    else:  # Bruker
        data = scan_group['EBSD/Data/RawPatterns']
    if lazy:
        data = da.from_array(data, chunks=data.chunks)
        scan['attributes']['_lazy'] = True
    else:
        data = np.array(data)

    try:
        if lazy:
            data = data.reshape((ny, nx, sy, sx)).squeeze()
        else:
            data = data.reshape((ny, nx, sy, sx), order='C').squeeze()
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
    names = ['x', 'y', 'dx', 'dy']
    scales = np.ones(4)

    # Calibrate scan dimension
    try:
        scales[0] = scales[0]*md.get_item(EBSD_str + 'step_x')
        scales[1] = scales[1]*md.get_item(EBSD_str + 'step_y')
    except BaseException:
        warnings.warn("Could not calibrate scan dimension, this can be done "
                      "using set_scan_calibration()")

    # Create axis objects for each axis
    dim = data.ndim
    axes = [
        {
            'size': data.shape[i],
            'index_in_array': i,
            'name': names[i],
            'scale': scales[i],
            'offset': 0.0,
            'units': units[i], }
        for i in range(dim)]
    scan['axes'] = axes

    return scan


def hdfdataset2dict(group, datasets, dictionary=None):
    """Return dictionary with values from datasets in a group in an
    opened HDF5 file.

    Parameters
    ----------
    group : h5py.Group
        HDF group name with required datasets.
    datasets : list of str
        List of HDF5 dataset names in group.
    dictionary : dict, optional
        Dictionary to fill dataset values into.

    Returns
    -------
    dictionary : dict
        Dictionary with filled values or None for required datasets.
    """
    if dictionary is None:
        dictionary = {}
    if isinstance(datasets, str):  # And not list of strings
        datasets = [datasets, ]
    for dset in datasets:
        for key, value in group.items():
            if dset in key.lower():
                value = value[()]
                if isinstance(value, np.ndarray):
                    value = value[0]
                if isinstance(value, bytes):
                    value = value.decode()
                dictionary[dset] = value
    return dictionary


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
    md.set_item(EBSD_str + 'manufacturer', manufacturer)
    md.set_item(EBSD_str + 'version', version)

    if 'edax' in manufacturer.lower():
        md, omd = tslheader2dicts(scan_group, md, lazy)
    elif 'bruker' in manufacturer.lower():
        md, omd = brukerheader2dicts(scan_group, md, lazy)
    else:  # KikuchiPy
        pass

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
        KikuchiPy's metadata structure with values from scan header.
    omd : DictionaryTreeBrowser
        All values from scan header.
    """
    # Get all metadata from file
    hg = scan_group['EBSD/Header']  # Header group
    omd = DictionaryTreeBrowser({'tsl_header': hdfgroup2dict(hg, lazy=lazy)})

    # Populate dictionary
    es = EBSD_str
    md.set_item(es + 'azimuth_angle', hg['Camera Azimuthal Angle'][0])
    md.set_item(es + 'elevation_angle', hg['Camera Elevation Angle'][0])
    grid_type = hg['Grid Type'][0].decode()
    if grid_type == 'SqrGrid':
        md.set_item(es + 'grid_type', 'square')
    else:
        raise IOError("Only square grids are supported, however a {} grid was "
                      "passed".format(grid_type))
    md.set_item(es + 'pattern_height', hg['Pattern Height'][0])
    md.set_item(es + 'pattern_width', hg['Pattern Width'][0])
    md.set_item(es + 'sample_tilt', hg['Sample Tilt'][0])
    md.set_item(es + 'step_x', hg['Step X'][0])
    md.set_item(es + 'step_y', hg['Step Y'][0])
    md.set_item(SEM_str + 'working_distance', hg['Working Distance'][0])
    md.set_item(es + 'n_rows', hg['nRows'][0])
    md.set_item(es + 'n_columns', hg['nColumns'][0])
    md.set_item('General.authors', hg['Operator'][0].decode())
    md.set_item('General.notes', hg['Notes'][0].decode())
    md.set_item(es + 'xpc', hg['Pattern Center Calibration/x-star'][0])
    md.set_item(es + 'ypc', hg['Pattern Center Calibration/y-star'][0])
    md.set_item(es + 'zpc', hg['Pattern Center Calibration/z-star'][0])

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
    md : DictionaryTreeBrowser
        KikuchiPy's metadata structure with values from scan header.
    omd : DictionaryTreeBrowser
        All values from scan header.
    """
    # Get all metadata from file
    hg = scan_group['EBSD/Header']  # Header group
    dg = scan_group['EBSD/Data']  # Data group with metadata info
    omd = DictionaryTreeBrowser({'bruker_header': hdfgroup2dict(hg, lazy=lazy)})

    # Populate dictionary
    es = EBSD_str
    md.set_item(es + 'elevation_angle', hg['CameraTilt'][()])
    grid_type = hg['Grid Type'][()].decode()
    if grid_type == 'isometric':
        md.set_item(es + 'grid_type', 'square')
    else:
        raise IOError("Only square grids are supported, however a {} grid was "
                      "passed".format(grid_type))
    md.set_item(es + 'pattern_height', hg['PatternHeight'][()])
    md.set_item(es + 'pattern_width', hg['PatternWidth'][()])
    md.set_item(es + 'sample_tilt', hg['SampleTilt'][()])
    md.set_item(es + 'step_x', hg['XSTEP'][()])
    md.set_item(es + 'step_y', hg['YSTEP'][()])
    md.set_item(SEM_str + 'working_distance', hg['WD'][()])
    md.set_item(SEM_str + 'beam_energy', hg['KV'][()])
    md.set_item(SEM_str + 'magnification', hg['Magnification'][()])
    md.set_item(es + 'n_rows', hg['NROWS'][()])
    md.set_item(es + 'n_columns', hg['NCOLS'][()])
    md.set_item(es + 'xpc', np.mean(dg['PCX'][()]))
    md.set_item(es + 'ypc', np.mean(dg['PCY'][()]))
    md.set_item(es + 'zpc', np.mean(dg['DD']))
    md.set_item(es + 'detector_pixel_size',
                hg['DetectorFullHeightMicrons'][()] /
                hg['UnClippedPatternHeight'][()])

    return md, omd
