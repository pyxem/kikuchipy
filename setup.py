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

from itertools import chain
from setuptools import setup, find_packages


# Get release information without importing anything from the project
with open("kikuchipy/release.py") as fid:
    for line in fid:
        if line.startswith("author"):
            AUTHOR = line.strip().split(" = ")[-1][1:-1]
        elif line.startswith("maintainer_email"):  # Must be before 'maintainer'
            MAINTAINER_EMAIL = line.strip(" = ").split()[-1][1:-1]
        elif line.startswith("maintainer"):
            MAINTAINER = line.strip().split(" = ")[-1][1:-1]
        elif line.startswith("name"):
            NAME = line.strip().split()[-1][1:-1]
        elif line.startswith("version"):
            VERSION = line.strip().split(" = ")[-1][1:-1]
        elif line.startswith("license"):
            LICENSE = line.strip().split(" = ")[-1][1:-1]
        elif line.startswith("platforms"):
            PLATFORMS = line.strip().split(" = ")[-1][1:-1]

# Projects with optional features for building the documentation and running
# tests. From setuptools:
# https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras-optional-features-with-their-own-dependencies
extra_feature_requirements = {
    "doc": [
        "nbsphinx >= 0.7",
        "sphinx >= 3.0.2",
        "sphinx-rtd-theme >= 0.4.3",
        "sphinx-copybutton >= 0.2.5",
        "sphinx-autodoc-typehints >= 1.10.3",
        "sphinx-gallery >= 0.6",
        "sphinxcontrib-bibtex >= 1.0",
    ],
    "tests": ["coverage >= 5.0", "pytest >= 5.4", "pytest-cov >= 2.8.1"],
}

# Create a development project, including both the doc and tests projects
extra_feature_requirements["dev"] = [
    "black >= 19.3b0",
    "pre-commit >= 1.16",
] + list(chain(*list(extra_feature_requirements.values())))

setup(
    # Package description
    name=NAME,
    version=VERSION,
    license=LICENSE,
    url="https://kikuchipy.org",
    python_requires=">=3.7",
    description=(
        "Processing and analysis of electron backscatter diffraction (EBSD) "
        "patterns."
    ),
    long_description=open("README.rst").read(),
    long_description_content_type="text/x-rst",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        (
            "License :: OSI Approved :: GNU General Public License v3 or later "
            "(GPLv3+)"
        ),
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    platforms=PLATFORMS,
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
    zip_safe=True,
    # Contact
    author=AUTHOR,
    author_email=MAINTAINER_EMAIL,
    download_url="https://pypi.python.org/pypi/kikuchipy",
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    project_urls={
        "Bug Tracker": "https://github.com/pyxem/kikuchipy/issues",
        "Documentation": "https://kikuchipy.org",
        "Source Code": "https://github.com/pyxem/kikuchipy",
    },
    # Dependencies
    extras_require=extra_feature_requirements,
    install_requires=[
        "dask[array] >= 2.14",
        "diffsims >= 0.4",
        "hyperspy >= 1.5.2",
        "h5py >= 2.10",
        "matplotlib >= 3.2",
        "numpy >= 1.18",
        "numba >= 0.48",
        "orix >= 0.5",
        "pooch >= 0.13",
        "psutil",
        "tqdm >= 0.5.2",
        "scikit-image >= 0.16",
        "scikit-learn",
        "scipy",
    ],
    entry_points={"hyperspy.extensions": "kikuchipy = kikuchipy"},
    # Files to include when distributing package
    packages=find_packages(),
    package_dir={"kikuchipy": "kikuchipy"},
    include_package_data=True,
    package_data={
        "": ["LICENSE", "README.rst"],
        "kikuchipy": ["*.py", "hyperspy_extension.yaml", "data/*"],
    },
)
