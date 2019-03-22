# -*- coding: utf-8 -*-
# Copyright 2019 The KikuchiPy developers
#
# This file is part of KikuchiPy.
#
# KikuchiPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# KikuchiPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with KikuchiPy. If not, see <http://www.gnu.org/licenses/>.

import logging

from kikuchipy.io_plugins import nordif

io_plugins = [nordif]

_logger = logging.getLogger(__name__)

try:
    from hyperspy.io_plugins import hspy
    io_plugins.append(hspy)
except ImportError:
    _logger.warning("The HDF5 IO features are not available. "
                    "It is highly recommended to install h5py")

try:
    from hyperspy.io_plugins import image
    io_plugins.append(image)
except ImportError:
    _logger.info("The Signal2D (PIL) IO features are not available")

default_write_ext = set()
for plugin in io_plugins:
    if plugin.writes:
        default_write_ext.add(
            plugin.file_extensions[plugin.default_extension])
