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

"""Input/output plugins.

.. currentmodule:: kikuchipy.io.plugins

.. rubric:: Modules

.. autosummary::
    :toctree: ../generated/
    :template: custom-module-template.rst

    bruker_h5ebsd
    ebsd_directory
    edax_binary
    edax_h5ebsd
    emsoft_ebsd
    emsoft_ebsd_master_pattern
    emsoft_ecp_master_pattern
    emsoft_tkd_master_pattern
    kikuchipy_h5ebsd
    nordif
    nordif_calibration_patterns
    oxford_binary
    oxford_h5ebsd
"""

__all__ = [
    "bruker_h5ebsd",
    "ebsd_directory",
    "edax_binary",
    "edax_h5ebsd",
    "emsoft_ebsd",
    "emsoft_ebsd_master_pattern",
    "emsoft_ecp_master_pattern",
    "emsoft_tkd_master_pattern",
    "kikuchipy_h5ebsd",
    "nordif",
    "nordif_calibration_patterns",
    "oxford_binary",
    "oxford_h5ebsd",
]


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    if name in __all__:
        import importlib

        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
