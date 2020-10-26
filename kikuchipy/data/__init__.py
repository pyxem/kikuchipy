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

"""Standard test data.

For more test data sets, see :doc:`open datasets <open_datasets>`.
"""

import os
from typing import Union

from kikuchipy.signals import EBSD, EBSDMasterPattern
from kikuchipy import load


__all__ = [
    "nickel_small",
]


DATA_DIR = os.path.dirname(__file__)


def _load(filename: str, **kwargs) -> Union[EBSD, EBSDMasterPattern]:
    """Load a data set located in the data directory.

    Parameters
    ----------
    filename : str
        File name.
    kwargs
        Keyword arguments passed to :func:`~kikuchipy.io._io.load`.

    Returns
    -------
    signal : EBSD or EBSDMasterPattern
        EBSD or master pattern signal.
    """
    file = os.path.join(DATA_DIR, filename)
    return load(file, **kwargs)


def nickel_small(**kwargs) -> EBSD:
    """9 EBSD patterns in a (3, 3) navigation shape of (60, 60) detector
    pixels from Nickel, acquired on a NORDIF UF-1100 detector.

    Parameters
    ----------
    kwargs
        Keyword arguments passed to :func:`~kikuchipy.io._io.load`.

    Returns
    -------
    signal : EBSD
        EBSD signal.
    """
    return _load(filename="kikuchipy/patterns.h5", **kwargs)
