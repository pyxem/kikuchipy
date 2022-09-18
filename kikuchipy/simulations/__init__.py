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

"""Simulations returned by a generator and handling of Kikuchi bands and
zone axes.
"""

import logging


class DisableMatplotlibWarningFilter(logging.Filter):  # pragma: no cover
    # Filter to suppress warnings with the below warning message
    # emitted by Matplotlib whenever coordinate arrays of text
    # positions contain NaN. This happens in most cases when we plot
    # zone axes label markers with HyperSpy.
    #
    # Filter has to be placed here to be executed at all due to lazy
    # imports.

    def filter(self):
        message_to_disable = "posx and posy should be finite values"
        return not self.msg == message_to_disable


logging.getLogger("matplotlib.text").addFilter(DisableMatplotlibWarningFilter)


__all__ = [
    "GeometricalKikuchiPatternSimulation",
    "KikuchiPatternSimulator",
]


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    _import_mapping = {
        "GeometricalKikuchiPatternSimulation": "_kikuchi_pattern_simulation",
        "KikuchiPatternSimulator": "kikuchi_pattern_simulator",
    }
    if name in __all__:
        import importlib

        if name in _import_mapping.keys():
            import_path = f"{__name__}.{_import_mapping.get(name)}"
            return getattr(importlib.import_module(import_path), name)
        else:  # pragma: no cover
            return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
