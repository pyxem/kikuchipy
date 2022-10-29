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

"""Example datasets for use when testing functionality.

Some datasets are packaged with the source code while others must be
downloaded from the web. For more test datasets, see
:doc:`/user/open_datasets`.

Datasets are placed in a local cache, in the location returned from
``pooch.os_cache("kikuchipy")`` by default. The location can be
overwritten with a global ``KIKUCHIPY_DATA_DIR`` environment variable.

With every new version of kikuchipy, a new directory of datasets with
the version name is added to the cache directory. Any old directories
are not deleted automatically, and should then be deleted manually if
desired.
"""

__all__ = [
    "nickel_ebsd_large",
    "nickel_ebsd_master_pattern_small",
    "nickel_ebsd_small",
    "silicon_ebsd_moving_screen_in",
    "silicon_ebsd_moving_screen_out10mm",
    "silicon_ebsd_moving_screen_out5mm",
]


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    _import_mapping = {
        "nickel_ebsd_large": "_data",
        "nickel_ebsd_master_pattern_small": "_data",
        "nickel_ebsd_small": "_data",
        "silicon_ebsd_moving_screen_in": "_data",
        "silicon_ebsd_moving_screen_out10mm": "_data",
        "silicon_ebsd_moving_screen_out5mm": "_data",
    }
    if name in __all__:
        import importlib

        if name in _import_mapping.keys():
            import_path = f"{__name__}.{_import_mapping.get(name)}"
            return getattr(importlib.import_module(import_path), name)
        else:  # pragma: no cover
            return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
