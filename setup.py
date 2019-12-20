# -*- coding: utf-8 -*-
# Copyright 2019 The KikuchiPy developers
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

from setuptools import setup, find_packages

import kikuchipy.release as release

setup(
    name='kikuchipy',
    version=release.version,
    description=(
        'Processing of electron backscatter diffraction (EBSD) patterns'),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author=release.author,
    maintainer=release.maintainer,
    maintainer_email=release.maintainer_email,
    keywords=[
        'EBSD', 'electron backscatter diffraction',
        'EBSP', 'electron backscatter pattern',
        'BKD', 'backscatter kikuchi diffraction',
        'SEM', 'scanning electron microscopy',
        'kikuchi pattern',
    ],
    url='https://github.com/kikuchipy/kikuchipy',
    package_dir={'kikuchipy': 'kikuchipy'},
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    packages=find_packages(),
    install_requires=[
        'dask', 'hyperspy >= 1.5.2', 'h5py', 'matplotlib', 'numpy', 'pyxem',
        'scikit-image', 'scikit-learn', 'scipy'],
    tests_require=['pytest', 'pytest-cov'],
    package_data={
        '': ['LICENSE', 'README.md'],
        'kikuchipy': ['*.py', 'hyperspy_extension.yaml', 'data']},
    entry_points={'hyperspy.extensions': 'kikuchipy = kikuchipy'},
    python_requires=">=3.7",
)
