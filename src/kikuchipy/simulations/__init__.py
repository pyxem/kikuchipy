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

"""Simulations returned by a generator and handling of Kikuchi bands and
zone axes.
"""

import logging

import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)


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

del lazy_loader, logging
