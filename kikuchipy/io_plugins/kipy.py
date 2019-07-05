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

# Plugin characteristics
# ----------------------
format_name = 'KIPY'
description = 'Read/write support for KikuchiPy\'s default format based on the HDF5 standard'
full_support = False
# Recognised file extension
file_extensions = ['kipy', 'KIPY']
default_extension = 1
# Writing capabilities
writes = False


def file_reader(filename, dataname='Scan 1/EBSD/Data/Pattern',
                headername='Scan 1/EBSD/Header', lazy=False, **kwargs):
    """Read electron backscatter patterns from the KikuchiPy HDF format
    akin to the h5ebsd format [1]_.

    Parameters
    ----------
    filename : str
        Full file path of the KikuchiPy HDF file.
    dataname : {str, 'Scan 1/EBSD/Data/Pattern'}, optional
        String with full KikuchiPy HDF dataset path with group names and
        the dataset name with the patterns as the last name.
    headername : {str, 'Scan 1/EBSD/Header'}, optional
        String with full KikuchiPy HDF dataset path with group names and
        the dataset name with the the header as the last name.
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
    f = h5py.File(filename, mode='r', **kwargs)
