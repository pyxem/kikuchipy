# -*- coding: utf-8 -*-
# Copyright 2019-2021 The kikuchipy developers
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

import re

from outdated import check_outdated


with open("../../kikuchipy/release.py") as fid:
    for line in fid:
        if line.startswith("version"):
            branch_version = line.strip().split(" = ")[-1][1:-1]

# Within a try/except because we don't want to throw the error if a new
# tagged release draft is to be made, we just want to know if the branch
# version is different (hopefully always newer if different) from the
# PyPI version
try:
    make_release, pypi_version = check_outdated("kikuchipy", branch_version)
except ValueError as e:
    pypi_version = re.findall(r"\s([\d.]+)", e.args[0])[1]
    make_release = True

# These three prints are collected by a bash script using `eval` and
# passed to GitHub Action environment variables to be used in a workflow
print(make_release)
print(pypi_version)
print(branch_version)
