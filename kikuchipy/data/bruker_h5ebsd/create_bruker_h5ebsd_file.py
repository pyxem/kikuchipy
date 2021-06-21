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

import os

from h5py import File
import numpy as np
import skimage.color as skc

import kikuchipy as kp


s = kp.data.nickel_ebsd_small()
ny, nx = s.axes_manager.navigation_shape[::-1]
n = ny * nx
sy, sx = s.axes_manager.signal_shape[::-1]

dir_data = os.path.abspath(os.path.dirname(__file__))


## File with no region of interest (ROI)
f1 = File(os.path.join(dir_data, "patterns.h5"), mode="w")

# Top group
manufacturer = f1.create_dataset("Manufacturer", shape=(1,), dtype="|S11")
manufacturer[()] = b"Bruker Nano"
version = f1.create_dataset("Version", shape=(1,), dtype="|S10")
version[()] = b"Esprit 2.X"
scan = f1.create_group("Scan 0")

# EBSD
ebsd = scan.create_group("EBSD")

ones9 = np.ones(n, dtype=np.float32)
zeros9 = np.zeros(n, dtype=np.float32)

ebsd_data = ebsd.create_group("Data")
ebsd_data.create_dataset("DD", dtype=np.float32, data=ones9)
ebsd_data.create_dataset("MAD", dtype=np.float32, data=ones9)
ebsd_data.create_dataset("MADPhase", dtype=np.int32, data=ones9)
ebsd_data.create_dataset("NIndexedBands", dtype=np.float32, data=ones9)
ebsd_data.create_dataset("PCX", dtype=np.float32, data=ones9)
ebsd_data.create_dataset("PCY", dtype=np.float32, data=ones9)
ebsd_data.create_dataset("PHI", dtype=np.float32, data=ones9)
ebsd_data.create_dataset("Phase", dtype=np.int32, data=ones9)
ebsd_data.create_dataset("RadonBandCount", dtype=np.int32, data=ones9)
ebsd_data.create_dataset("RadonQuality", dtype=np.float32, data=ones9)
ebsd_data.create_dataset("RawPatterns", data=s.data.reshape((n, sy, sx)))
ebsd_data.create_dataset("X BEAM", dtype=np.int32, data=ones9)
ebsd_data.create_dataset("X SAMPLE", dtype=np.float32, data=ones9)
ebsd_data.create_dataset("Y BEAM", dtype=np.int32, data=ones9)
ebsd_data.create_dataset("Y SAMPLE", dtype=np.float32, data=ones9)
ebsd_data.create_dataset("Z SAMPLE", dtype=np.float32, data=zeros9)
ebsd_data.create_dataset("phi1", dtype=np.float32, data=ones9)
ebsd_data.create_dataset("phi2", dtype=np.float32, data=ones9)

ebsd_header = ebsd.create_group("Header")
ebsd_header.create_dataset("CameraTilt", dtype=float, data=0)
ebsd_header.create_dataset("DetectorFullHeightMicrons", dtype=np.int32, data=sy)
ebsd_header.create_dataset("DetectorFullWidthMicrons", dtype=np.int32, data=sx)
grid_type = ebsd_header.create_dataset("Grid Type", shape=(1,), dtype="|S9")
grid_type[()] = b"isometric"
ebsd_header.create_dataset("KV", dtype=float, data=20)
ebsd_header.create_dataset("MADMax", dtype=float, data=1.5)
ebsd_header.create_dataset("Magnification", dtype=float, data=200)
ebsd_header.create_dataset("MapStepFactor", dtype=float, data=4)
ebsd_header.create_dataset("MaxRadonBandCount", dtype=np.int32, data=11)
ebsd_header.create_dataset("MinIndexedBands", dtype=np.int32, data=5)
ebsd_header.create_dataset("NCOLS", dtype=np.int32, data=nx)
ebsd_header.create_dataset("NROWS", dtype=np.int32, data=ny)
ebsd_header.create_dataset("NPoints", dtype=np.int32, data=n)
original_file = ebsd_header.create_dataset("OriginalFile", shape=(1,), dtype="|S50")
original_file[()] = b"/a/home/for/your/data.h5"
ebsd_header.create_dataset("PatternHeight", dtype=np.int32, data=sy)
ebsd_header.create_dataset("PatternWidth", dtype=np.int32, data=sx)
ebsd_header.create_dataset("PixelByteCount", dtype=np.int32, data=1)
s_mean = s.nanmean((2, 3)).data.astype(np.uint8)
ebsd_header.create_dataset("SEM Image", data=skc.gray2rgb(s_mean))
ebsd_header.create_dataset("SEPixelSizeX", dtype=float, data=1)
ebsd_header.create_dataset("SEPixelSizeY", dtype=float, data=1)
ebsd_header.create_dataset("SampleTilt", dtype=float, data=70)
bg = s.metadata.Acquisition_instrument.SEM.Detector.EBSD.static_background
ebsd_header.create_dataset("StaticBackground", dtype=np.uint16, data=bg)
ebsd_header.create_dataset("TopClip", dtype=float, data=1)
ebsd_header.create_dataset("UnClippedPatternHeight", dtype=np.int32, data=sy)
ebsd_header.create_dataset("WD", dtype=float, data=1)
ebsd_header.create_dataset("XSTEP", dtype=float, data=1.5)
ebsd_header.create_dataset("YSTEP", dtype=float, data=1.5)
ebsd_header.create_dataset("ZOffset", dtype=float, data=0)

phase = ebsd_header.create_group("Phases/1")
formula = phase.create_dataset("Formula", shape=(1,), dtype="|S2")
formula[()] = b"Ni"
phase.create_dataset("IT", dtype=np.int32, data=225)
phase.create_dataset(
    "LatticeConstants", dtype=np.float32, data=np.array([3.56, 3.56, 3.56, 90, 90, 90])
)
name = phase.create_dataset("Name", shape=(1,), dtype="|S6")
name[()] = b"Nickel"
phase.create_dataset("Setting", dtype=np.int32, data=1)
space_group = phase.create_dataset("SpaceGroup", shape=(1,), dtype="|S5")
space_group[()] = b"Fm-3m"
atom_pos = phase.create_group("AtomPositions")
atom_pos1 = atom_pos.create_dataset("1", shape=(1,), dtype="|S17")
atom_pos1[()] = b"Ni,0,0,0,1,0.0035"

# SEM
sem = scan.create_group("SEM")
sem.create_dataset("SEM IX", dtype=np.int32, data=np.ones(1))
sem.create_dataset("SEM IY", dtype=np.int32, data=np.ones(1))
sem.create_dataset("SEM Image", data=skc.gray2rgb(s_mean))
sem.create_dataset("SEM ImageHeight", dtype=np.int32, data=3)
sem.create_dataset("SEM ImageWidth", dtype=np.int32, data=3)
sem.create_dataset("SEM KV", dtype=float, data=20)
sem.create_dataset("SEM Magnification", dtype=float, data=200)
sem.create_dataset("SEM WD", dtype=float, data=24.5)
sem.create_dataset("SEM XResolution", dtype=float, data=1)
sem.create_dataset("SEM YResolution", dtype=float, data=1)
sem.create_dataset("SEM ZOffset", dtype=float, data=0)

f1.close()


## File with rectangular ROI (SEM group under EBSD group as well)
f2 = File(os.path.join(dir_data, "patterns_roi.h5"), mode="w")

# Top group
manufacturer = f2.create_dataset("Manufacturer", shape=(1,), dtype="|S11")
manufacturer[()] = b"Bruker Nano"
version = f2.create_dataset("Version", shape=(1,), dtype="|S10")
version[()] = b"Esprit 2.X"
scan = f2.create_group("Scan 0")

# EBSD
ebsd = scan.create_group("EBSD")

# ROI and shape
roi = np.array(
    [
        [0, 1, 1],  # 0, 1, 2 | (0, 0) (0, 1) (0, 2)
        [0, 1, 1],  # 3, 4, 5 | (1, 0) (1, 1) (1, 2)
        [0, 1, 1],  # 6, 7, 8 | (2, 0) (2, 1) (2, 2)
    ],
    dtype=bool,
).flatten()
# Order of ROI patterns: 4, 1, 2, 5, 7, 8
iy = np.array([1, 0, 0, 1, 2, 2], dtype=int)
ix = np.array([1, 1, 2, 2, 1, 2], dtype=int)
n_roi = np.sum(roi)

# Data
ones9 = np.ones(9, dtype=np.float32)[roi]
zeros9 = np.zeros(9, dtype=np.float32)[roi]
ebsd_data = ebsd.create_group("Data")
ebsd_data.create_dataset("DD", dtype=np.float32, data=ones9)
ebsd_data.create_dataset("MAD", dtype=np.float32, data=ones9)
ebsd_data.create_dataset("MADPhase", dtype=np.int32, data=ones9)
ebsd_data.create_dataset("NIndexedBands", dtype=np.float32, data=ones9)
ebsd_data.create_dataset("PCX", dtype=np.float32, data=ones9)
ebsd_data.create_dataset("PCY", dtype=np.float32, data=ones9)
ebsd_data.create_dataset("PHI", dtype=np.float32, data=ones9)
ebsd_data.create_dataset("Phase", dtype=np.int32, data=ones9)
ebsd_data.create_dataset("RadonBandCount", dtype=np.int32, data=ones9)
ebsd_data.create_dataset("RadonQuality", dtype=np.float32, data=ones9)
ebsd_data.create_dataset("RawPatterns", data=s.data.reshape((n, sy, sx))[roi])
ebsd_data.create_dataset("X BEAM", dtype=np.int32, data=ones9)
ebsd_data.create_dataset("X SAMPLE", dtype=np.float32, data=ones9)
ebsd_data.create_dataset("Y BEAM", dtype=np.int32, data=ones9)
ebsd_data.create_dataset("Y SAMPLE", dtype=np.float32, data=ones9)
ebsd_data.create_dataset("Z SAMPLE", dtype=np.float32, data=zeros9)
ebsd_data.create_dataset("phi1", dtype=np.float32, data=ones9)
ebsd_data.create_dataset("phi2", dtype=np.float32, data=ones9)

# Header
ebsd_header = ebsd.create_group("Header")
ebsd_header.create_dataset("CameraTilt", dtype=float, data=0)
ebsd_header.create_dataset("DetectorFullHeightMicrons", dtype=np.int32, data=23700)
ebsd_header.create_dataset("DetectorFullWidthMicrons", dtype=np.int32, data=31600)
grid_type = ebsd_header.create_dataset("Grid Type", shape=(1,), dtype="|S9")
grid_type[()] = b"isometric"
ebsd_header.create_dataset("KV", dtype=float, data=20)
ebsd_header.create_dataset("MADMax", dtype=float, data=1.5)
ebsd_header.create_dataset("Magnification", dtype=float, data=200)
ebsd_header.create_dataset("MapStepFactor", dtype=float, data=4)
ebsd_header.create_dataset("MaxRadonBandCount", dtype=np.int32, data=11)
ebsd_header.create_dataset("MinIndexedBands", dtype=np.int32, data=5)
ebsd_header.create_dataset("NCOLS", dtype=np.int32, data=nx)
ebsd_header.create_dataset("NROWS", dtype=np.int32, data=ny)
ebsd_header.create_dataset("NPoints", dtype=np.int32, data=n)
original_file = ebsd_header.create_dataset("OriginalFile", shape=(1,), dtype="|S50")
original_file[()] = b"/a/home/for/your/data.h5"
ebsd_header.create_dataset("PatternHeight", dtype=np.int32, data=sy)
ebsd_header.create_dataset("PatternWidth", dtype=np.int32, data=sx)
ebsd_header.create_dataset("PixelByteCount", dtype=np.int32, data=1)
s_mean = s.nanmean((2, 3)).data.astype(np.uint8)
ebsd_header.create_dataset("SEM Image", data=skc.gray2rgb(s_mean))
ebsd_header.create_dataset("SEPixelSizeX", dtype=float, data=1)
ebsd_header.create_dataset("SEPixelSizeY", dtype=float, data=1)
ebsd_header.create_dataset("SampleTilt", dtype=float, data=70)
bg = s.metadata.Acquisition_instrument.SEM.Detector.EBSD.static_background
ebsd_header.create_dataset("StaticBackground", dtype=np.uint16, data=bg)
ebsd_header.create_dataset("TopClip", dtype=float, data=1)
ebsd_header.create_dataset("UnClippedPatternHeight", dtype=np.int32, data=sy)
ebsd_header.create_dataset("WD", dtype=float, data=1)
ebsd_header.create_dataset("XSTEP", dtype=float, data=1.5)
ebsd_header.create_dataset("YSTEP", dtype=float, data=1.5)
ebsd_header.create_dataset("ZOffset", dtype=float, data=0)
# Phases
phase = ebsd_header.create_group("Phases/1")
formula = phase.create_dataset("Formula", shape=(1,), dtype="|S2")
formula[()] = b"Ni"
phase.create_dataset("IT", dtype=np.int32, data=225)
phase.create_dataset(
    "LatticeConstants",
    dtype=np.float32,
    data=np.array([3.56, 3.56, 3.56, 90, 90, 90]),
)
name = phase.create_dataset("Name", shape=(1,), dtype="|S6")
name[()] = b"Nickel"
phase.create_dataset("Setting", dtype=np.int32, data=1)
space_group = phase.create_dataset("SpaceGroup", shape=(1,), dtype="|S5")
space_group[()] = b"Fm-3m"
atom_pos = phase.create_group("AtomPositions")
atom_pos1 = atom_pos.create_dataset("1", shape=(1,), dtype="|S17")
atom_pos1[()] = b"Ni,0,0,0,1,0.0035"

# SEM
sem = ebsd.create_group("SEM")
sem.create_dataset("IX", dtype=np.int32, data=ix)
sem.create_dataset("IY", dtype=np.int32, data=iy)
sem.create_dataset("ZOffset", dtype=float, data=0)

f2.close()


## File with non-rectangular ROI (SEM group under EBSD group as well)
f3 = File(os.path.join(dir_data, "patterns_roi_nonrectangular.h5"), mode="w")

# Top group
manufacturer = f3.create_dataset("Manufacturer", shape=(1,), dtype="|S11")
manufacturer[()] = b"Bruker Nano"
version = f3.create_dataset("Version", shape=(1,), dtype="|S10")
version[()] = b"Esprit 2.X"
scan = f3.create_group("Scan 0")

# EBSD
ebsd = scan.create_group("EBSD")

# ROI and shape
roi = np.array(
    [
        [0, 1, 1],  # 0, 1, 2 | (0, 0) (0, 1) (0, 2)
        [0, 1, 1],  # 3, 4, 5 | (1, 0) (1, 1) (1, 2)
        [0, 1, 0],  # 6, 7, 8 | (2, 0) (2, 1) (2, 2)
    ],
    dtype=bool,
).flatten()
# Order of ROI patterns: 4, 1, 2, 7, 5
iy = np.array([1, 0, 0, 2, 1], dtype=int)
ix = np.array([1, 1, 2, 1, 2], dtype=int)
n_roi = np.sum(roi)

# Data
ones9 = np.ones(n, dtype=np.float32)[roi]
zeros9 = np.zeros(n, dtype=np.float32)[roi]
ebsd_data = ebsd.create_group("Data")
ebsd_data.create_dataset("DD", dtype=np.float32, data=ones9)
ebsd_data.create_dataset("MAD", dtype=np.float32, data=ones9)
ebsd_data.create_dataset("MADPhase", dtype=np.int32, data=ones9)
ebsd_data.create_dataset("NIndexedBands", dtype=np.float32, data=ones9)
ebsd_data.create_dataset("PCX", dtype=np.float32, data=ones9)
ebsd_data.create_dataset("PCY", dtype=np.float32, data=ones9)
ebsd_data.create_dataset("PHI", dtype=np.float32, data=ones9)
ebsd_data.create_dataset("Phase", dtype=np.int32, data=ones9)
ebsd_data.create_dataset("RadonBandCount", dtype=np.int32, data=ones9)
ebsd_data.create_dataset("RadonQuality", dtype=np.float32, data=ones9)
ebsd_data.create_dataset("RawPatterns", data=s.data.reshape((n, sy, sx))[roi])
ebsd_data.create_dataset("X BEAM", dtype=np.int32, data=ones9)
ebsd_data.create_dataset("X SAMPLE", dtype=np.float32, data=ones9)
ebsd_data.create_dataset("Y BEAM", dtype=np.int32, data=ones9)
ebsd_data.create_dataset("Y SAMPLE", dtype=np.float32, data=ones9)
ebsd_data.create_dataset("Z SAMPLE", dtype=np.float32, data=zeros9)
ebsd_data.create_dataset("phi1", dtype=np.float32, data=ones9)
ebsd_data.create_dataset("phi2", dtype=np.float32, data=ones9)

# Header
ebsd_header = ebsd.create_group("Header")
ebsd_header.create_dataset("CameraTilt", dtype=float, data=0)
ebsd_header.create_dataset("DetectorFullHeightMicrons", dtype=np.int32, data=sy)
ebsd_header.create_dataset("DetectorFullWidthMicrons", dtype=np.int32, data=sx)
grid_type = ebsd_header.create_dataset("Grid Type", shape=(1,), dtype="|S9")
grid_type[()] = b"isometric"
# ebsd_header.create_dataset("KV", dtype=float, data=20)
ebsd_header.create_dataset("MADMax", dtype=float, data=1.5)
ebsd_header.create_dataset("Magnification", dtype=float, data=200)
ebsd_header.create_dataset("MapStepFactor", dtype=float, data=4)
ebsd_header.create_dataset("MaxRadonBandCount", dtype=np.int32, data=11)
ebsd_header.create_dataset("MinIndexedBands", dtype=np.int32, data=5)
ebsd_header.create_dataset("NCOLS", dtype=np.int32, data=nx)
ebsd_header.create_dataset("NROWS", dtype=np.int32, data=ny)
ebsd_header.create_dataset("NPoints", dtype=np.int32, data=n)
original_file = ebsd_header.create_dataset("OriginalFile", shape=(1,), dtype="|S50")
original_file[()] = b"/a/home/for/your/data.h5"
ebsd_header.create_dataset("PatternHeight", dtype=np.int32, data=sy)
ebsd_header.create_dataset("PatternWidth", dtype=np.int32, data=sx)
ebsd_header.create_dataset("PixelByteCount", dtype=np.int32, data=1)
s_mean = s.nanmean((2, 3)).data.astype(np.uint8)
ebsd_header.create_dataset("SEM Image", data=skc.gray2rgb(s_mean))
ebsd_header.create_dataset("SEPixelSizeX", dtype=float, data=1)
ebsd_header.create_dataset("SEPixelSizeY", dtype=float, data=1)
ebsd_header.create_dataset("SampleTilt", dtype=float, data=70)
bg = s.metadata.Acquisition_instrument.SEM.Detector.EBSD.static_background
ebsd_header.create_dataset("StaticBackground", dtype=np.uint16, data=bg)
ebsd_header.create_dataset("TopClip", dtype=float, data=1)
ebsd_header.create_dataset("UnClippedPatternHeight", dtype=np.int32, data=sy)
ebsd_header.create_dataset("WD", dtype=float, data=1)
ebsd_header.create_dataset("XSTEP", dtype=float, data=1.5)
ebsd_header.create_dataset("YSTEP", dtype=float, data=1.5)
ebsd_header.create_dataset("ZOffset", dtype=float, data=0)
# Phases
phase = ebsd_header.create_group("Phases/1")
formula = phase.create_dataset("Formula", shape=(1,), dtype="|S2")
formula[()] = b"Ni"
phase.create_dataset("IT", dtype=np.int32, data=225)
phase.create_dataset(
    "LatticeConstants", dtype=np.float32, data=np.array([3.56, 3.56, 3.56, 90, 90, 90])
)
name = phase.create_dataset("Name", shape=(1,), dtype="|S6")
name[()] = b"Nickel"
phase.create_dataset("Setting", dtype=np.int32, data=1)
space_group = phase.create_dataset("SpaceGroup", shape=(1,), dtype="|S5")
space_group[()] = b"Fm-3m"
atom_pos = phase.create_group("AtomPositions")
atom_pos1 = atom_pos.create_dataset("1", shape=(1,), dtype="|S17")
atom_pos1[()] = b"Ni,0,0,0,1,0.0035"

# SEM
sem = ebsd.create_group("SEM")
sem.create_dataset("IX", dtype=np.int32, data=ix)
sem.create_dataset("IY", dtype=np.int32, data=iy)
sem.create_dataset("ZOffset", dtype=float, data=0)

f3.close()
