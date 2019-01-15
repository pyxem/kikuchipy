# -*- coding: utf-8 -*-
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
