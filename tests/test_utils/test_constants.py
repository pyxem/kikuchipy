#
# Copyright 2019-2026 the kikuchipy developers
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
#

import pytest

from kikuchipy._constants import dependency_version, verify_dependency_or_raise


@pytest.mark.skipif(
    dependency_version["ipywidgets"] is not None, reason="ipywidgets is installed"
)
def test_verify_dependency_or_raise():
    verify_dependency_or_raise("numpy", "")

    with pytest.raises(ImportError, match="for some reason"):
        verify_dependency_or_raise("ipywidgets", "for some reason")
