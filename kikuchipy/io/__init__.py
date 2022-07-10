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

"""Read and write signals from and to file.

.. currentmodule:: kikuchipy.io

.. rubric:: Modules

.. autosummary::
    :toctree: ../generated/
    :template: custom-module-template.rst

    plugins
"""

__all__ = [
    "plugins",
]


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    if name in __all__:  # pragma: no cover
        import importlib

        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
