# Copyright 2019-2024 The kikuchipy developers
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

import lazy_loader

# Initial committer first, then sorted by line contributions
credits = [
    "Håkon Wiik Ånes",
    "Lars Andreas Hastad Lervik",
    "Ole Natlandsmyr",
    "Tina Bergh",
    "Eric Prestat",
    "Andreas V. Bugten",
    "Erlend Mikkelsen Østvold",
    "Zhou Xu",
    "Carter Francis",
    "Magnus Nord",
]
__version__ = "0.11.dev0"

# Attempt (and fail) import of optional dependencies only once
try:
    import pyvista

    _pyvista_installed = True
except ImportError:  # pragma: no cover
    _pyvista_installed = False

try:
    import nlopt

    _nlopt_installed = True
except ImportError:  # pragma: no cover
    _nlopt_installed = False

try:
    from pyebsdindex import ebsd_index, pcopt

    _pyebsdindex_installed = True
except ImportError:  # pragma: no cover
    _pyebsdindex_installed = False

# PyOpenCL context available for use with PyEBSDIndex? Required for
# Hough indexing of Dask arrays.
# PyOpenCL is an optional dependency of PyEBSDIndex, so it should not be
# an optional kikuchipy dependency.
try:  # pragma: no cover
    import pyopencl as cl

    platform = cl.get_platforms()[0]
    gpu = platform.get_devices(device_type=cl.device_type.GPU)
    ctx = cl.Context(devices=gpu)
    if ctx is None:
        _pyopencl_context_available = False
    else:
        _pyopencl_context_available = True
except:  # pragma: no cover
    # Have to use bare except because PyOpenCL might raise its own
    # LogicError, but we also want to catch import errors here
    _pyopencl_context_available = False

__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)

del lazy_loader
