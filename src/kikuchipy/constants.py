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

"""Constants and such useful across modules."""

from importlib.metadata import version

from packaging.version import Version

deps_for_version_check = [
    # Core
    "hyperspy",
    "matplotlib",
    "numpy",
    # Optional
    "nlopt",
    "pyvista",
    "pyebsdindex",
]
dependency_version: dict[str, Version | None] = {}
for dep in deps_for_version_check:
    try:
        dep_version = Version(version(dep))
    except ImportError:  # pragma: no cover
        dep_version = None
    dependency_version[dep] = dep_version


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
        pyopencl_context_available = False
    else:
        pyopencl_context_available = True
except:  # pragma: no cover
    # Have to use bare except because PyOpenCL might raise its own
    # LogicError, but we also want to catch import errors here
    pyopencl_context_available = False


# TODO: Remove and use numpy.exceptions.VisibleDeprecationWarning once
# NumPy 1.25 is minimal supported version
try:
    # Added in NumPy 1.25.0
    from numpy.exceptions import VisibleDeprecationWarning
except ImportError:  # pragma: no cover
    # Removed in NumPy 2.0.0
    from numpy import VisibleDeprecationWarning

del dep_version, deps_for_version_check, version
