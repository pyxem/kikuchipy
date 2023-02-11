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
# kikuchipy/<version>/data/ unless stated otherwise.

# fmt: off
_registry_hashes = {
    # In package (relative to the kikuchipy/data directory)
    "kikuchipy_h5ebsd/patterns.h5":                                 "md5:f5e24fc55befedd08ee1b5a507e413ad",
    "emsoft_ebsd_master_pattern/ni_mc_mp_20kv_uint8_gzip_opts9.h5": "md5:807c8306a0d02b46effbcb12bd44cd02",
    "nickel_ebsd_large/patterns.h5":                                "md5:51d6bc0f5ff23dcb0c1a8e1f4c52d4d4",
    # From GitHub
    "silicon_ebsd_moving_screen/si_in.h5":                          "md5:d8561736f6174e6520a45c3be19eb23a",
    "silicon_ebsd_moving_screen/si_out5mm.h5":                      "md5:77dd01cc2cae6c1c5af6708260c94cab",
    "silicon_ebsd_moving_screen/si_out10mm.h5":                     "md5:0b4ece1533f380a42b9b81cfd0dd202c",
    # From Zenodo
    "ebsd_si_wafer.zip":                                            "md5:444ec4188ba8c8bda5948c2bf4f9a672",
    "si_wafer/Pattern.dat":                                         "md5:58952a93c3ecacff22955f1ad7c61246",
    "scan1_gain0db.zip":                                            "md5:7393a03afe5d52aec56dfc62e5cefdc3",
    "scan2_gain3db.zip":                                            "md5:bc6f53c88d5423027e376136656e1d65",
    "scan3_gain6db.zip":                                            "md5:625ef7f32b978c4b84ada054d14832ef",
    "scan4_gain9db.zip":                                            "md5:d30f45bde3a9106baf400c42e50e25a7",
    "scan5_gain12db.zip":                                           "md5:4aea8a15a8e71c7fd34dc0d581fa9a11",
    "scan6_gain15db.zip":                                           "md5:6de4fd16be1eca00952220df25624f56",
    "scan7_gain17db.zip":                                           "md5:30bc7ae9a9a85f98c2ee1ccf8c5e5cf2",
    "scan8_gain20db.zip":                                           "md5:84ccb7c47a596f346cb310c2e88d0207",
    "scan9_gain22db.zip":                                           "md5:635cca32729bcf9aa9fdb93fb09c0e58",
    "scan10_gain24db.zip":                                          "md5:d1d75a8cc182bb0e7e79bdacb8fc3603",
    "ni_gain/1/Pattern.dat":                                        "md5:79febebf41b0d0a12781501a7564a721",
    "ni_gain/1/Setting.txt":                                        "md5:776b1a2da5c359b0d399b50be5b5144b",
    "ni_gain/2/Pattern.dat":                                        "md5:4659a9e492b14b02d1f5492c5b8cf05a",
    "ni_gain/2/Setting.txt":                                        "md5:3f227e27ee71dc4bcf164c5d3043f03a",
    "ni_gain/3/Pattern.dat":                                        "md5:b923be74ef642d8fe961c2356c160236",
    "ni_gain/3/Setting.txt":                                        "md5:c1c19b77ced0cc644827b1edac615e21",
    "ni_gain/4/Pattern.dat":                                        "md5:b91a8f63ac5f5cdcc508074aa6ffe598",
    "ni_gain/4/Setting.txt":                                        "md5:3f68f0b1f4ca16f1a8f8e6b36613e0c2",
    "ni_gain/5/Pattern.dat":                                        "md5:94773dc46aa3ca5142dd1b70715bbb77",
    "ni_gain/5/Setting.txt":                                        "md5:e6e2c83c5903a3fdac92bd8b5afc9aa7",
    "ni_gain/6/Pattern.dat":                                        "md5:fd444d5bc7d283230fd1a76f220c42db",
    "ni_gain/6/Setting.txt":                                        "md5:21a0e8530930ba8df35dbb68c330241f",
    "ni_gain/7/Pattern.dat":                                        "md5:7d04e558adc3ed4249768cb9515b0c04",
    "ni_gain/7/Setting.txt":                                        "md5:1fb6b657c07daa719865e8acc57b335c",
    "ni_gain/8/Pattern.dat":                                        "md5:c2106626d0a06118c647c21e1acc3f11",
    "ni_gain/8/Setting.txt":                                        "md5:86a108169e410018db460e3ce1e8978e",
    "ni_gain/9/Pattern.dat":                                        "md5:106c8e6eb1083c08f8ca2bc2f735cb31",
    "ni_gain/9/Setting.txt":                                        "md5:7d6d422b0ee00b4b497c1503ae88dc42",
    "ni_gain/10/Pattern.dat":                                       "md5:bd9be321d3a4cd8f3954bb8774fc70ba",
    "ni_gain/10/Setting.txt":                                       "md5:515b3d8e4657dbc0b7566977b4a3eaca",
    "ebsd_master_pattern/al_mc_mp_20kv.h5":                         "md5:be0f79dd025d9c82e413ce8d635d48f4",
    "ebsd_master_pattern/ni_mc_mp_20kv.h5":                         "md5:8b69c071a036ad3488d465093b67fe4d",
    "ebsd_master_pattern/si_mc_mp_20kv.h5":                         "md5:d4962b97bf364c42e3bd5ce1b2711d02",
    "ebsd_master_pattern/austenite_mc_mp_20kv.h5":                  "md5:ca5c9961ce8c9ebf33802d0769876256",
    "ebsd_master_pattern/ferrite_mc_mp_20kv.h5":                    "md5:4b6c1456ed2d90e190c7a21c4c4c1aff",
    "ebsd_master_pattern/steel_sigma_mc_mp_20kv.h5":                "md5:2d965e399dbc13cb5983f29ceef6dfcd",
    "ebsd_master_pattern/steel_chi_mc_mp_20kv.h5":                  "md5:9e4dd974bf78a3f7d159575ff0d0a28a",
}
# How to use permanent links to files on GitHub:
# https://docs.github.com/en/repositories/working-with-files/using-files/getting-permanent-links-to-files
KP_DATA_REPO_URL = "https://raw.githubusercontent.com/pyxem/kikuchipy-data/"
_registry_urls = {
    # From GitHub
    "nickel_ebsd_large/patterns.h5":                    KP_DATA_REPO_URL + "bcab8f7a4ffdb86a97f14e2327a4813d3156a85e/nickel_ebsd_large/patterns_v2.h5",
    "silicon_ebsd_moving_screen/si_in.h5":              KP_DATA_REPO_URL + "bcab8f7a4ffdb86a97f14e2327a4813d3156a85e/silicon_ebsd_moving_screen/si_in.h5",
    "silicon_ebsd_moving_screen/si_out5mm.h5":          KP_DATA_REPO_URL + "bcab8f7a4ffdb86a97f14e2327a4813d3156a85e/silicon_ebsd_moving_screen/si_out5mm.h5",
    "silicon_ebsd_moving_screen/si_out10mm.h5":         KP_DATA_REPO_URL + "bcab8f7a4ffdb86a97f14e2327a4813d3156a85e/silicon_ebsd_moving_screen/si_out10mm.h5",
    # From Zenodo
    "ebsd_si_wafer.zip":                                "https://zenodo.org/record/7491388/files/ebsd_si_wafer.zip",
    "scan1_gain0db.zip":                                "https://zenodo.org/record/7498632/files/scan1_gain0db.zip",
    "scan2_gain3db.zip":                                "https://zenodo.org/record/7498632/files/scan2_gain3db.zip",
    "scan3_gain6db.zip":                                "https://zenodo.org/record/7498632/files/scan3_gain6db.zip",
    "scan4_gain9db.zip":                                "https://zenodo.org/record/7498632/files/scan4_gain9db.zip",
    "scan5_gain12db.zip":                               "https://zenodo.org/record/7498632/files/scan5_gain12db.zip",
    "scan6_gain15db.zip":                               "https://zenodo.org/record/7498632/files/scan6_gain15db.zip",
    "scan7_gain17db.zip":                               "https://zenodo.org/record/7498632/files/scan7_gain17db.zip",
    "scan8_gain20db.zip":                               "https://zenodo.org/record/7498632/files/scan8_gain20db.zip",
    "scan9_gain22db.zip":                               "https://zenodo.org/record/7498632/files/scan9_gain22db.zip",
    "scan10_gain24db.zip":                              "https://zenodo.org/record/7498632/files/scan10_gain24db.zip",
    "ebsd_master_pattern/al_mc_mp_20kv.h5":             "https://zenodo.org/record/7628365/files/al_mc_mp_20kv.h5",
    "ebsd_master_pattern/ni_mc_mp_20kv.h5":             "https://zenodo.org/record/7498645/files/ni_mc_mp_20kv.h5",
    "ebsd_master_pattern/si_mc_mp_20kv.h5":             "https://zenodo.org/record/7498729/files/si_mc_mp_20kv.h5",
    "ebsd_master_pattern/austenite_mc_mp_20kv.h5":  "https://zenodo.org/record/7628387/files/austenite_mc_mp_20kv.h5",
    "ebsd_master_pattern/ferrite_mc_mp_20kv.h5":    "https://zenodo.org/record/7628394/files/ferrite_mc_mp_20kv.h5",
    "ebsd_master_pattern/steel_chi_mc_mp_20kv.h5":      "https://zenodo.org/record/7628417/files/steel_chi_mc_mp_20kv.h5",
    "ebsd_master_pattern/steel_sigma_mc_mp_20kv.h5":    "https://zenodo.org/record/7628443/files/steel_sigma_mc_mp_20kv.h5",
}
# fmt: on

# Prepend "data/" to all keys
registry_hashes = {}
for k, v in _registry_hashes.items():
    registry_hashes["data/" + k] = v
registry_urls = {}
for k, v in _registry_urls.items():
    registry_urls["data/" + k] = v
