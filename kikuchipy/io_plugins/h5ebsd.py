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
from hyperspy.misc.utils import DictionaryTreeBrowser

# Plugin characteristics
# ----------------------
format_name = 'h5ebsd'
description = 'Write support for h5ebsd files'
full_support = False
# Recognised file extension
file_extensions = ['h5ebsd', 'h5', 'hdf5']
default_extension = 0
# Writing capabilities
writes = [(2, 2), (2, 1), (2, 0)]


def get_default_header():
    """Return a header prepopulated with default values."""
    header = dict()

    header['Camera Azimuthal Angle'] = np.float32(0)  # [degrees]
    header['Camera Elevation Angle'] = np.float32(0)  # [degrees]
    header['Grid Type'] = 'SqrGrid'
    header['Notes'] = ''
    header['Operator'] = ''
    header['Pattern Height'] = np.int32(0)
    header['Pattern Width'] = np.int32(0)
    header['Sample ID'] = ''
    header['Sample Tilt'] = np.float32(0)  # [degrees]
    header['Scan ID'] = ''
    header['Step X'] = np.float32(0)  # [um]
    header['Step Y'] = np.float32(0)  # [um]
    header['Working Distance'] = np.float32(0)  # [mm]
    header['nColumns'] = np.int32(0)
    header['nRows'] = np.int32(0)

    # Explanation of coordinate system in images
    cr_sys = 'Coordinate System/'
    header[cr_sys + 'EBSD View Reference Frame'] = np.zeros((207, 245, 3),
                                                            dtype=np.uint8)
    header[cr_sys + 'ID'] = 0
    for i in np.arange(1, 5):
        header[cr_sys + 'Schematic ' + str(i)] = np.zeros((351, 170, 3),
                                                          dtype=np.uint8)

    # Pattern center calibration
    pc_cal = 'Pattern Center Calibration/'
    header[pc_cal + 'x-star'] = np.float32(0)
    header[pc_cal + 'y-star'] = np.float32(0)
    header[pc_cal + 'z-star'] = np.float32(0)

    # Phase information
    # TODO: Enable reading of multiple phases
    phase = 'Phase/1/'
    header[phase + 'Formula'] = ''
    header[phase + 'Info'] = ''
    header[phase + 'Lattice Constant a'] = np.float32(0)  # [Å]
    header[phase + 'Lattice Constant b'] = np.float32(0)  # [Å]
    header[phase + 'Lattice Constant c'] = np.float32(0)  # [Å]
    header[phase + 'Lattice Constant alpha'] = np.float32(0)  # [degrees]
    header[phase + 'Lattice Constant beta'] = np.float32(0)  # [degrees]
    header[phase + 'Lattice Constant gamma'] = np.float32(0)  # [degrees]
    header[phase + 'Laue Group'] = ''
    header[phase + 'MaterialName'] = ''
    header[phase + 'NumberFamilies'] = ''
    header[phase + 'Point Group'] = ''
    header[phase + 'Symmetry'] = np.int32(0)
    header[phase + 'hkl Families'] = np.int32(0)

    return header


def get_header(file, headername='Scan 1/EBSD/Header/'):
    """Read the header of a scan of electron backscatter patterns from
    an open HDF5 file in the h5ebsd format.

    Parameters
    ----------
    file : file object
    headername : {str, 'Scan 1/EBSD/Header'}, optional
        A string containing the full HDF dataset path with group names
        and the dataset name with the the header as the last name.

    Returns
    -------
    md : DictionaryTreeBrowser
        Metadata complying with HyperSpy's metadata structure.
    omd : DictionaryTreeBrowser
        Original metadata that does not fit into HyperSpy's metadata
        structure.
    """

    # Set up `metadata` and `original_metadata` structures
    md = DictionaryTreeBrowser()
    omd = DictionaryTreeBrowser()

    # Get header with default values
    header = get_default_header()

    # Overwrite default values
    header_file = file[headername]
    for i, dataset in enumerate(header.keys()):
        if dataset in header_file:
            entry = header_file[dataset][:]
            if isinstance(entry[0], np.bytes_):
                entry = entry[0].decode()
            elif isinstance(entry, np.ndarray) and len(entry) == 1:
                entry = entry[0]
            header[dataset] = entry

    # Populate `metadata` and `original_metadata` with appropriate values

    return header


def file_reader(filename, dataname='Scan 1/EBSD/Data/Pattern',
                headername='Scan 1/EBSD/Header', lazy=False):
    """Read electron backscatter patterns from an HDF5 file in the
    h5ebsd format [1]_. The patterns and settings are assumed to reside
    in datasets named 'Pattern' and 'Header', respectively.

    Parameters
    ----------
    filename : str
        Full file path of h5ebsd file.
    dataname : {str, 'Scan 1/EBSD/Data/Pattern'}, optional
        A string containing the full HDF dataset path with group names
        and the dataset name with the patterns as the last name.
    headername : {str, 'Scan 1/EBSD/Header'}, optional
        A string containing the full HDF dataset path with group names
        and the dataset name with the the header as the last name.
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

    # Write header into `metadata` and `original_metadata`
    header = get_header(f, headername)

    if not lazy:
        f.close()

    return header


def file_writer(filename, signal, **kwargs):
    """Write electron backscatter diffraction patterns to an h5ebsd
    HDF5 file.

    Parameters
    ----------
    filename : str
        Full file path of NORDIF data file.
    signal : :obj:`kikuchipy.signals.EBSD` or
             :obj:`kikuchipy.signals.LazyEBSD`
    """
    with h5py.File(filename, mode='w') as f:
        nx, ny = signal.axes_manager.navigation_shape
        sx, sy = signal.axes_manager.signal_shape

        # Write to file
#        for name, dtype in h5ebsd_structure:
#            f.create_dataset(name=name, data)
#        f.create_dataset('Max X Points', data=nx, dtype=np.int32)
#        f.create_dataset('Max Y Points', data=ny, dtype=np.int32)
#        f.create_dataset('Index', data=nx * ny, dtype=np.int32)
        f.create_dataset('EBSD/Pattern', data=signal.data.reshape(nx * ny, sx, sy))
