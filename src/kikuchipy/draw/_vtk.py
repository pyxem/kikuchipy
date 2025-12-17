#
# Copyright 2019-2025 the kikuchipy developers
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

"""Utilites for plotting with VTK."""

from functools import cache
import os
from subprocess import PIPE, Popen, TimeoutExpired
import sys


@cache
def system_supports_plotting() -> bool:
    """Return whether the system supports plotting with VTK (/PyVista).

    Supported OSes are Linux, macOS, and Windows.

    Taken from PyVista:
    https://github.com/pyvista/pyvista/blob/main/pyvista/plotting/tools.py.
    """
    platf = sys.platform
    if platf == "linux":
        return linux_supports_plotting()
    elif platf == "darwin":
        return macos_supports_plotting()
    else:
        return windows_supports_plotting()


def linux_supports_plotting() -> bool:
    """Return whether attempting to print current settings of the X
    Window server is successful.
    """
    try:
        proc = Popen(["xset", "-q"], stdout=PIPE, stderr=PIPE, encoding="utf8")
        proc.communicate(timeout=10)
    except (OSError, TimeoutExpired):
        return False
    else:
        return proc.returncode == 0


def macos_supports_plotting() -> bool:
    """Return whether Finder is available or if there exists an
    environment variable named "DISPLAY".

    Taken from PyVista:
    https://github.com/pyvista/pyvista/blob/main/pyvista/plotting/tools.py.
    """
    proc = Popen(["pgrep", "-qx", "Finder"], stdout=PIPE, stderr=PIPE, encoding="utf8")
    try:
        proc.communicate(timeout=10)
    except TimeoutExpired:
        return False
    if proc.returncode == 0:
        return True
    return "DISPLAY" in os.environ


def windows_supports_plotting() -> bool:
    return supports_open_gl()


def supports_open_gl() -> bool:
    """Return whether the system supports OpenGL.

    This function checks if the system supports OpenGL by creating a VTK
    render window and querying its OpenGL support.

    Taken from PyVista:
    https://github.com/pyvista/pyvista/blob/main/pyvista/plotting/tools.py.
    """
    from vtk import vtkRenderWindow

    render_window = vtkRenderWindow()
    return bool(render_window.SupportsOpenGL())
