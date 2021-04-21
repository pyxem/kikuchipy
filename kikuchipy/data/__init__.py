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

"""Test data.

Some datasets must be downloaded from the web. For more test datasets,
see :doc:`open datasets <open_datasets>`.
"""

from pathlib import Path
import os
from typing import Union

import pooch as ppooch

from kikuchipy.signals import EBSD, EBSDMasterPattern
from kikuchipy import load
from kikuchipy.release import version
from kikuchipy.data._registry import registry, registry_urls


__all__ = [
    "nickel_ebsd_small",
    "nickel_ebsd_large",
    "nickel_ebsd_master_pattern_small",
]


fetcher = ppooch.create(
    path=ppooch.os_cache("kikuchipy"),
    base_url="",
    version=version.replace(".dev", "+"),
    env="KIKUCHIPY_DATA_DIR",
    registry=registry,
    urls=registry_urls,
)
cache_data_path = fetcher.path.joinpath("data")
package_data_path = Path(os.path.abspath(os.path.dirname(__file__)))


def _has_hash(path, expected_hash):
    """Check if the provided path has the expected hash."""
    if not os.path.exists(path):
        return False
    else:
        return ppooch.utils.file_hash(path) == expected_hash


def _cautious_downloader(url, output_file, pooch):
    if pooch.allow_download:
        delattr(pooch, "allow_download")
        # HTTPDownloader() requires tqdm, a HyperSpy dependency, so
        # adding it to our dependencies doesn't cost anything
        download = ppooch.HTTPDownloader(progressbar=True)
        download(url, output_file, pooch)
    else:
        raise ValueError(
            "The dataset must be (re)downloaded from the kikuchipy-data "
            "repository on GitHub (https://github.com/pyxem/kikuchipy-data) to "
            "your local cache with the pooch Python package. Pass "
            "`allow_download=True` to allow this download."
        )


def _fetch(filename: str, allow_download: bool = False):
    resolved_path = os.path.join(package_data_path, "..", filename)
    expected_hash = registry[filename]
    if _has_hash(resolved_path, expected_hash):  # File already in data module
        return resolved_path
    else:  # Pooch must download the data to the local cache
        fetcher.allow_download = allow_download  # Extremely ugly
        resolved_path = fetcher.fetch(filename, downloader=_cautious_downloader)
    return resolved_path


def _load(filename: str, **kwargs) -> Union[EBSD, EBSDMasterPattern]:
    allow_download = kwargs.pop("allow_download", False)
    return load(_fetch(filename, allow_download=allow_download), **kwargs)


def nickel_ebsd_small(**kwargs) -> EBSD:
    """9 EBSD patterns in a (3, 3) navigation shape of (60, 60) detector
    pixels from Nickel, acquired on a NORDIF UF-1100 detector
    :cite:`aanes2019electron`.

    Parameters
    ----------
    kwargs
        Keyword arguments passed to :func:`~kikuchipy.io._io.load`.

    Returns
    -------
    signal : EBSD
        EBSD signal.
    """
    return _load(filename="data/kikuchipy/patterns.h5", **kwargs)


def nickel_ebsd_master_pattern_small(**kwargs) -> EBSDMasterPattern:
    """(401, 401) `uint8` square Lambert or stereographic projection of the
    northern and southern hemisphere of a Nickel master pattern at 20
    keV accelerating voltage.

    Parameters
    ----------
    kwargs
        Keyword arguments passed to :func:`~kikuchipy.io._io.load`.

    Returns
    -------
    signal : EBSDMasterPattern
        EBSD master pattern signal.

    Notes
    -----
    Initially generated using the EMsoft EMMCOpenCL and EMEBSDMaster
    programs. The included file was rewritten to disk with
    :mod:`h5py`, where the master patterns' data type is converted from
    `float32` to `uint8` with
    :meth:`~kikuchipy.signals.EBSDMasterPattern.rescale_intensity`, all
    datasets were written with
    :meth:`~kikuchipy.io.plugins.h5ebsd.dict2h5ebsdgroup` with
    keyword arguments `compression="gzip"` and `compression_opts=9`. All
    other HDF5 groups and datasets are the same as in the original file.
    """
    fname = "data/emsoft_ebsd_master_pattern/ni_mc_mp_20kv_uint8_gzip_opts9.h5"
    return _load(fname, **kwargs)


def nickel_ebsd_large(allow_download: bool = False, **kwargs) -> EBSD:
    """4125 EBSD patterns in a (55, 75) navigation shape of (60, 60)
    detector pixels from Nickel, acquired on a NORDIF UF-1100 detector
    :cite:`aanes2019electron`.

    Parameters
    ----------
    allow_download : bool
        Whether to allow downloading the dataset from the kikuchipy-data
        GitHub repository (https://github.com/pyxem/kikuchipy-data) to
        the local cache with the pooch Python package. Default is False.
    kwargs
        Keyword arguments passed to :func:`~kikuchipy.io._io.load`.

    Returns
    -------
    signal : EBSD
        EBSD signal.
    """
    return _load(
        filename="data/nickel_ebsd_large/patterns.h5",
        allow_download=allow_download,
        **kwargs,
    )
