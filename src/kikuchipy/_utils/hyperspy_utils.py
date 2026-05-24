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

"""Utilities for handling compatibility with HyperSpy."""

from packaging.version import Version

from kikuchipy._constants import dependency_version

if dependency_version["hyperspy"] >= Version("2.4.0"):
    from hyperspy.learn import LearningResults
else:
    from hyperspy.learn.mva import LearningResults

__all__ = [
    "LearningResults",
]
