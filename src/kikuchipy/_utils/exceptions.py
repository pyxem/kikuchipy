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

from typing import Any


class UnknownHemisphereError(ValueError):
    def __init__(self, given: Any = None, *args: object) -> None:
        msg = "Unknown hemisphere"
        if given is not None:
            msg += f" {given!r}"
        msg += ", options are 'upper', 'lower', or 'both'"
        super().__init__(msg, *args)


class UnknownProjectionError(ValueError):
    def __init__(self, given: Any = None, *args: object) -> None:
        msg = "Unknown projection"
        if given is not None:
            msg += f" {given!r}"
        msg += ", options are 'stereographic' and 'lambert'"
        super().__init__(msg, *args)
