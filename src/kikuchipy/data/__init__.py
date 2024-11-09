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

import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)


del lazy_loader
