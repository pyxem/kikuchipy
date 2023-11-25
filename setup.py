# Copyright 2019-2023 The kikuchipy developers
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
# fmt: off
extra_feature_requirements = {
    "doc": [
        "memory_profiler",
        "nbsphinx                       >= 0.7",
        "numpydoc",
        "nlopt",
        "pydata-sphinx-theme",
        "pyebsdindex                    ~= 0.2",
        "pyvista",
        "sphinx                         >= 3.0.2",
        "sphinx-codeautolink[ipython]   < 0.14",
        "sphinx-copybutton              >= 0.2.5",
        "sphinx-design",
        "sphinx-gallery",
        "sphinxcontrib-bibtex           >= 1.0",
    ],
    "tests": [
        "coverage                       >= 5.0",
        "numpydoc",
        "pytest                         >= 5.4",
        "pytest-benchmark",
        "pytest-cov                     >= 2.8.1",
        "pytest-rerunfailures",
        "pytest-xdist",
    ],
    "all": [
        "matplotlib                     >= 3.5",
        "nlopt",
        # We ask for a compatible release of PyEBSDIndex as we
        # anticipate breaking changes in coming releases. We do so
        # because there were breaking changes between 0.1.2 and 0.2.0.
        # We can change from ~= to >= once we consider PyEBSDIndex
        # stable. This is typically when a minor release with no or
        # only minor breaking changes is made available.
        "pyebsdindex                    ~= 0.2",
        "pyvista",
    ],
}
# fmt: on

# Create a development project including all extra dependencies
extra_feature_requirements["dev"] = [
    "black[jupyter]                     >= 23.1",
    "manifix",
    "outdated",
    "pre-commit                         >= 1.16",
] + list(chain(*list(extra_feature_requirements.values())))


setup(
    # Package description
    name=NAME,
    version=VERSION,
    license=LICENSE,
    url="https://kikuchipy.org",
    python_requires=">=3.7",
    description=(
        "Processing, simulating and indexing of electron backscatter diffraction "
        "(EBSD) patterns."
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
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
        "dictionary indexing",
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
    # fmt: off
    install_requires=[
        "dask[array]        >= 2021.8.1",
        "diffpy.structure   >= 3",
        "diffsims           >= 0.5.1",
        "hyperspy           >= 1.7.3, < 2",
        "h5py               >= 2.10",
        "imageio",
        "matplotlib         >= 3.5",
        "numba              >= 0.55",
        "numpy              >= 1.21.6",
        "orix               >= 0.11.1",
        "pooch              >= 1.3.0",
        "pyyaml",
        "rosettasciio",
        "tqdm               >= 0.5.2",
        "scikit-image       >= 0.16.2",
        "scikit-learn",
        "scipy              >= 1.7",
    ],
    # fmt: on
    entry_points={"hyperspy.extensions": "kikuchipy = kikuchipy"},
    # Files to include when distributing package
    packages=find_packages(),
    package_dir={"kikuchipy": "kikuchipy"},
    include_package_data=True,
    package_data={
        "": ["LICENSE", "README.md"],
        "kikuchipy": ["*.py", "hyperspy_extension.yaml", "data/*"],
    },
)
