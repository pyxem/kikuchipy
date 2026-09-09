#
# Copyright 2026 the kikuchipy developers
#
# SPDX-License-Identifier: BSD-3-Clause
#

"""Constants and such useful across modules."""

from importlib.metadata import version

from packaging.version import Version

# TODO: Create this list dynamically using importlib.metadata.require
deps_for_version_check = [
    # Required
    "hyperspy",
    "matplotlib",
    "numpy",
    "rosettasciio",
    "scikit-image",
    # Optional
    "ebsdsim",
    "IPython",
    "ipywidgets",
    "nlopt",
    "pooch",
    "psygnal",
    "pyvista",
    "pyebsdindex",
    "scikit-optimize",
]
dependency_version: dict[str, Version | None] = {}
for dep in deps_for_version_check:
    try:
        dep_version = Version(version(dep))
    except ImportError:  # pragma: no cover
        dep_version = None
    dependency_version[dep] = dep_version


def verify_dependency_or_raise(package: str, reason: str) -> None:
    """Raise an informative ImportError if a *package* required for some
    *reason* is not installed.
    """
    if dependency_version[package] is None:
        raise ImportError(f"{reason} requires that {package!r} is installed")


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
except Exception:  # pragma: no cover
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
    from numpy import VisibleDeprecationWarning  # noqa: F401

del dep_version, deps_for_version_check, version
