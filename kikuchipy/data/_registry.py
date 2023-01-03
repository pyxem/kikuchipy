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

# All hashes are MD5 hashes and can be checked locally with e.g. md5sum.
# All file paths are relative to the cache directory
# kikuchipy/<version>/data unless stated otherwise.

# fmt: off
_registry_hashes = {
    # In package (relative to the kikuchipy/data directory)
    "kikuchipy_h5ebsd/patterns.h5": "md5:f5e24fc55befedd08ee1b5a507e413ad",
    "emsoft_ebsd_master_pattern/ni_mc_mp_20kv_uint8_gzip_opts9.h5": "md5:807c8306a0d02b46effbcb12bd44cd02",
    "nickel_ebsd_large/patterns.h5": "md5:51d6bc0f5ff23dcb0c1a8e1f4c52d4d4",
    # From GitHub
    "silicon_ebsd_moving_screen/si_in.h5": "md5:d8561736f6174e6520a45c3be19eb23a",
    "silicon_ebsd_moving_screen/si_out5mm.h5": "md5:77dd01cc2cae6c1c5af6708260c94cab",
    "silicon_ebsd_moving_screen/si_out10mm.h5": "md5:0b4ece1533f380a42b9b81cfd0dd202c",
    # From Zenodo
    "ebsd_si_wafer.zip": "md5:444ec4188ba8c8bda5948c2bf4f9a672",
    "si_wafer/Pattern.dat": "md5:58952a93c3ecacff22955f1ad7c61246",
    "scan1_gain0db.zip": "md5:7393a03afe5d52aec56dfc62e5cefdc3",
    "ni_gain0/Pattern.dat": "md5:79febebf41b0d0a12781501a7564a721",
    "ni_gain0/Setting.txt": "md5:776b1a2da5c359b0d399b50be5b5144b",
}
# How to use permanent links to files on GitHub:
# https://docs.github.com/en/repositories/working-with-files/using-files/getting-permanent-links-to-files
KP_DATA_REPO_URL = "https://raw.githubusercontent.com/pyxem/kikuchipy-data/"
_registry_urls = {
    # From GitHub
    "nickel_ebsd_large/patterns.h5": KP_DATA_REPO_URL + "bcab8f7a4ffdb86a97f14e2327a4813d3156a85e/nickel_ebsd_large/patterns_v2.h5",
    "silicon_ebsd_moving_screen/si_in.h5": KP_DATA_REPO_URL + "bcab8f7a4ffdb86a97f14e2327a4813d3156a85e/silicon_ebsd_moving_screen/si_in.h5",
    "silicon_ebsd_moving_screen/si_out5mm.h5": KP_DATA_REPO_URL + "bcab8f7a4ffdb86a97f14e2327a4813d3156a85e/silicon_ebsd_moving_screen/si_out5mm.h5",
    "silicon_ebsd_moving_screen/si_out10mm.h5": KP_DATA_REPO_URL + "bcab8f7a4ffdb86a97f14e2327a4813d3156a85e/silicon_ebsd_moving_screen/si_out10mm.h5",
    # From Zenodo
    "ebsd_si_wafer.zip": "https://zenodo.org/record/7491388/files/ebsd_si_wafer.zip",
    "scan1_gain0db.zip": "https://zenodo.org/record/7498632/files/scan1_gain0db.zip",
}
# fmt: on

# Prepend "data/" to all keys
registry_hashes = {}
for k, v in _registry_hashes.items():
    registry_hashes["data/" + k] = v
registry_urls = {}
for k, v in _registry_urls.items():
    registry_urls["data/" + k] = v
