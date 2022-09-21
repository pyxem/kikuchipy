# Copyright 2019-2022 The kikuchipy developers
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

from kikuchipy.release import version as __version__


# Attempt import only once
try:
    import pyvista

    _pyvista_installed = True
except ImportError:  # pragma: no cover
    _pyvista_installed = False


def set_log_level(level: str):  # pragma: no cover
    """Set level of kikuchipy logging messages.

    Parameters
    ----------
    level
        Any value accepted by :meth:`logging.Logger.setLevel()`. Levels
        are ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"`` and
        ``"CRITICAL"``.

    Notes
    -----
    See https://docs.python.org/3/howto/logging.html.

    Examples
    --------
    >>> import kikuchipy as kp
    >>> kp.set_log_level("DEBUG")
    >>> s = kp.data.nickel_ebsd_small()
    >>> s2 = s.deepcopy()  # doctest: +SKIP
    DEBUG:kikuchipy.signals._kikuchipy_signal:Transfer custom properties when deep copying
    """
    import logging

    logging.basicConfig()
    logging.getLogger("kikuchipy").setLevel(level)


__all__ = [
    "__version__",
    "_pyvista_installed",
    "crystallography",
    "data",
    "detectors",
    "draw",
    "filters",
    "generators",
    "indexing",
    "io",
    "load",
    "pattern",
    "projections",
    "release",
    "set_log_level",
    "signals",
    "simulations",
]


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    _import_mapping = {
        "load": "io._io",
    }
    if name in __all__:
        import importlib

        if name in _import_mapping.keys():
            import_path = f"{__name__}.{_import_mapping.get(name)}"
            return getattr(importlib.import_module(import_path), name)
        else:  # pragma: no cover
            return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
