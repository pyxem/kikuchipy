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

import numpy as np
import dask.array as da
import h5py
import warnings
from hyperspy.misc.utils import DictionaryTreeBrowser
from kikuchipy.utils.io_utils import ebsd_metadata


# Plugin characteristics
# ----------------------
format_name = 'TSLHDF'
description = 'Read support for EDAX TSL\'s HDF format'
full_support = False  # Does not write, and phases are not read
# Recognised file extension
file_extensions = ['h5', 'hdf5']
default_extension = 0
# Writing capabilities
writes = False

# Set common string
EBSD_str = 'Acquisition_instrument.SEM.Detector.EBSD.'


def tsl_metadata():
    """Return dictionary relating parameters in EDAX's TSL HDF file data
    header to KikuchiPy's internal EBSD metadata, also specifying TSL
    data type and default value.

    Returns
    -------
    md_header : dict
    """
    md_header = dict()
    md_header['Camera Azimuthal Angle'] = ('azimuth_angle', np.float32(0))
    md_header['Camera Elevation Angle'] = ('elevation_angle', np.float32(0))
    md_header['Grid Type'] = ('grid_type', np.string_(''))
    md_header['Pattern Height'] = ('pattern_height', np.float32(0))
    md_header['Pattern Width'] = ('pattern_width', np.float32(0))
    md_header['Sample Tilt'] = ('sample_tilt', np.float32(0))
    md_header['Step X'] = ('step_x', np.float32(0))
    md_header['Step Y'] = ('step_y', np.float32(0))
    md_header['Working Distance'] = ('working_distance', np.float32(0))
    md_header['nRows'] = ('n_rows', np.float32(0))
    md_header['nColumns'] = ('n_columns', np.float32(0))
    md_header['Coordinate System/EBSD View Reference Frame'] = \
        (None, np.zeros((207, 245, 3), dtype=np.uint8))
    md_header['Coordinate System/ID'] = (None, np.float32(0))
    md_header['Coordinate System/Schematic 1'] = \
        (None, np.zeros((351, 170, 3), dtype=np.uint8))
    md_header['Coordinate System/Schematic 2'] = \
        (None, np.zeros((351, 170, 3), dtype=np.uint8))
    md_header['Coordinate System/Schematic 3'] = \
        (None, np.zeros((351, 170, 3), dtype=np.uint8))
    md_header['Coordinate System/Schematic 4'] = \
        (None, np.zeros((351, 170, 3), dtype=np.uint8))
    md_header['Notes'] = (None, np.string_(''))
    md_header['Operator'] = (None, np.string_(''))
    md_header['Sample ID'] = (None, np.string_(''))
    md_header['Scan ID'] = (None, np.string_(''))
    md_header['Pattern Center Calibration/x-star'] = ('xpc', np.float32(0))
    md_header['Pattern Center Calibration/y-star'] = ('ypc', np.float32(0))
    md_header['Pattern Center Calibration/z-star'] = ('zpc', np.float32(0))
    # TODO: Read phases
    return md_header


def get_header(file, headername='Scan 1/EBSD/Header/'):
    """Read the header of a scan of electron backscatter patterns from
    an open EDAX TSL HDF file.

    Parameters
    ----------
    file : file object
    headername : {str, 'Scan 1/EBSD/Header'}, optional
        String with full EDAX TSL HDF dataset path with group names and
        the dataset name with the the header as the last name.

    Returns
    -------
    md : DictionaryTreeBrowser
        Metadata complying with HyperSpy's metadata structure.
    omd : DictionaryTreeBrowser
        Metadata that does not fit into HyperSpy's metadata structure.
    """
    # Get header with default values
    header_match = tsl_metadata()
    header = {key: value[1] for key, value in header_match.items()}

    # Overwrite default values if found in TSL HDF file
    header_group = file[headername]
    for dataset in header.keys():
        if dataset in header_group:
            entry = header_group[dataset][:]
            if isinstance(entry[0], np.bytes_):
                entry = entry[0].decode()  # Don't want byte strings, b''
            elif isinstance(entry, np.ndarray) and len(entry) == 1:
                entry = entry[0]  # Don't want all entries as lists
            header[dataset] = entry

    # Create metadata and original metadata structures
    md = ebsd_metadata()
    omd = DictionaryTreeBrowser()

    # General info and all metadata
    md.set_item('General.authors', header['Operator'])
    md.set_item('General.notes', header['Notes'])
    omd.set_item('tslhdf_header', header)

    # Overwrite KikuchiPy's default metadata with values from TSL header
    for key, values in header_match.items():
        match, default_value = values
        if match is not None:  # Parameter exists in KikuchiPy's EBSD metadata
            md.set_item(EBSD_str + match, header[key])

    return md, omd


def file_reader(filename, dataname='Scan 1/EBSD/Data/Pattern',
                headername='Scan 1/EBSD/Header', scan_size=None,
                pattern_size=None, lazy=False):
    """Read electron backscatter patterns from the EDAX TSL HDF format
    akin to the h5ebsd format [1]_.

    Parameters
    ----------
    filename : str
        Full file path of the EDAX TSL HDF file.
    dataname : {str, 'Scan 1/EBSD/Data/Pattern'}, optional
        String with full EDAX TSL HDF dataset path with group names and
        the dataset name with the patterns as the last name.
    headername : {str, 'Scan 1/EBSD/Header'}, optional
        String with full EDAX TSL HDF dataset path with group names and
        the dataset name with the the header as the last name.
    scan_size : {None, tuple}, optional
        Scan size in number of patterns in width and height.
    pattern_size : {None, tuple}, optional
        Pattern size in detector pixels in width and height.
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
    f = h5py.File(filename, mode='r')

    # Check if datasets exist
    for dsetname in [dataname, headername]:
        if dsetname not in f:
            raise IOError("Could not find dataset '{}' in file '{}'. Set "
                          "HDF dataset paths with the `dataname` and "
                          "`headername` parameters"
                          .format(dsetname, filename))

    # Write header to metadata and original metadata
    try:
        md, omd = get_header(f, headername)  # Overwrite empty dictionaries
        scan_size = (md.get_item(EBSD_str + 'n_columns'),
                     md.get_item(EBSD_str + 'n_rows'))
        pattern_size = (md.get_item(EBSD_str + 'pattern_width'),
                        md.get_item(EBSD_str + 'pattern_height'))
        if 'EBSD/Data/Pattern' in dataname:
            md.set_item('General.title',
                        dataname[:dataname.find('/EBSD/Data/Pattern')])
        else:
            md.set_item('General.title', dataname)
    except BaseException:
        md = ebsd_metadata()
        omd = DictionaryTreeBrowser()
        warnings.warn("Reading the HDF5 file header failed")

    # Set required parameters in metadata
    md.set_item('General.original_filename', filename)
    md.set_item('Signal.signal_type', 'electron_backscatter_diffraction')
    md.set_item('Signal.record_by', 'image')

    if scan_size is None and pattern_size is None:
        raise ValueError("No scan size and pattern size provided")

    # Set scan size and pattern size
    (nx, ny) = scan_size
    (sx, sy) = pattern_size

    # Read data from file
    data = f[dataname]
    if lazy:
        data = da.from_array(data, chunks=data.chunks)
    else:
        data = np.array(data)

    try:
        if lazy:
            data = data.reshape((ny, nx, sx, sy)).squeeze()
        else:
            data = data.reshape((ny, nx, sx, sy), order='C').squeeze()
    except ValueError:
        warnings.warn("Pattern size and scan size larger than file size! "
                      "Will attempt to load by zero padding incomplete "
                      "frames.")
        # Data is stored pattern by pattern
        pw = [(0, ny * nx * sx * sy - data.size)]
        if lazy:
            data = da.pad(data, pw, mode='constant')
        else:
            data = np.pad(data, pw, mode='constant')
        data = data.reshape((ny, nx, sx, sy))

    units = [u'\u03BC'+'m', u'\u03BC'+'m', 'A^{-1}', 'A^{-1}']
    names = ['y', 'x', 'dx', 'dy']
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

    dictionary = {'data': data,
                  'axes': axes,
                  'metadata': md.as_dictionary(),
                  'original_metadata': omd.as_dictionary()}

    if not lazy:
        f.close()

    return [dictionary, ]
