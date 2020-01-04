# -*- coding: utf-8 -*-
# Copyright 2019-2020 The KikuchiPy developers
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

from itertools import chain
from setuptools import setup, find_packages

# Get release information without importing anything from the project
with open("kikuchipy/release.py") as fid:
    for line in fid:
        if line.startswith("author"):
            author = line.strip().split(" = ")[-1][1:-1]
        elif line.startswith("maintainer_email"):  # Must be before 'maintainer'
            maintainer_email = line.strip(" = ").split()[-1][1:-1]
        elif line.startswith("maintainer"):
            maintainer = line.strip().split(" = ")[-1][1:-1]
        elif line.startswith("name"):
            name = line.strip().split()[-1][1:-1]
        elif line.startswith("version"):
            version = line.strip().split(" = ")[-1][1:-1]

# Projects with optional features for building the documentation and running
# tests. From setuptools:
# https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras-optional-features-with-their-own-dependencies
extra_feature_requirements = {
    "doc": [
        "sphinx >= 2.3.1",
        "sphinx-rtd-theme >= 0.4.3",
        "sphinx-copybutton >= 0.2.5",
    ],
    "tests": [
        "pytest >= 5.3.2",
        "pytest-cov >= 2.8.1",
        "coverage == 4.5.4",  # 5.0 have some issues with reporting to Coveralls
    ],
}

# Create a development project, including both the doc and tests projects
extra_feature_requirements["dev"] = [
    "black >= 19.3b0",
    "pre-commit >= 1.16",
] + list(chain(*list(extra_feature_requirements.values())))

setup(
    name=name,
    version=version,
    description=(
        "Processing of electron backscatter diffraction (EBSD) patterns"
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author=author,
    author_email=maintainer_email,
    maintainer=maintainer,
    maintainer_email=maintainer_email,
    keywords=[
        "EBSD",
        "electron backscatter diffraction",
        "EBSP",
        "electron backscatter pattern",
        "BKD",
        "backscatter kikuchi diffraction",
        "SEM",
        "scanning electron microscopy",
        "kikuchi pattern",
    ],
    url="https://github.com/kikuchipy/kikuchipy",
    package_dir={"kikuchipy": "kikuchipy"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    packages=find_packages(),
    install_requires=[
        "dask[array]",
        "hyperspy >= 1.5.2",
        "h5py",
        "matplotlib",
        "numpy >= 1.17",
        "pyxem >= 0.10",
        "scikit-image",
        "scikit-learn",
        "scipy",
    ],
    extras_require=extra_feature_requirements,
    package_data={
        "": ["LICENSE", "README.md"],
        "kikuchipy": ["*.py", "hyperspy_extension.yaml", "data"],
    },
    entry_points={"hyperspy.extensions": "kikuchipy = kikuchipy"},
    python_requires=">=3.7",
)
