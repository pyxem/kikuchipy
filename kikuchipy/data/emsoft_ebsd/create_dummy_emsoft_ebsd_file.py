import os

import numpy as np
import h5py
import kikuchipy as kp


ddir = "/home/hakon/kode/kikuchipy/kikuchipy/data/emsoft_ebsd"
fname = "simulated_ebsd.h5"
f = h5py.File(os.path.join(ddir, fname), mode="w")

data_shape = (10, 10, 10)
data = {
    "Manufacturer": "EMsoft",
    "CrystalData": {
        "AtomData": np.array(
            [[0.1587, 0], [0.6587, 0], [0, 0.25], [1, 1], [0.005, 0.005]],
            dtype=np.float32,
        ),
        "Atomtypes": np.array([13, 29], dtype=np.int32),
        "CrystalSystem": 2,
        "LatticeParameters": np.array([0.5949, 0.5949, 0.5821, 90, 90, 90]),
        "Natomtypes": 2,
        "Source": "Su Y.C., Yan J., Lu P.T., Su J.T.: Thermodynamic...",
        "SpaceGroupNumber": 140,
        "SpaceGroupSetting": 1,
    },
    "EMData": {
        "EBSD": {
            "EBSDPatterns": np.zeros(data_shape, dtype=int),
            "EulerAngles": np.zeros((data_shape[0], 3), dtype=int),
            "numangles": data_shape[0],
            "xtalname": "ni/ni.xtal",
        }
    },
    "NMLparameters": {
        "EBSDNameList": {
            "Ftensor": np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            ),
            "L": 2123.3,
            "alphaBD": 0.0,
            "anglefile": "pattern/ni/2020/1/ni1_sda_eulers.txt",
            "anglefiletype": "orientations",
            "applyDeformation": "n",
            "axisangle": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            "beamcurrent": 150.0,
            "binning": 1,
            "bitdepth": "8bit",
            "datafile": "pattern/ni/2020/1/ni1_sda_sim.h5",
            "delta": 70.0,
            "dwelltime": 100.0,
            "energyaverage": 0,
            "energyfile": "crystal_data/ni/ni_mc_mp_20kv.h5",
            "energymax": 20.0,
            "energymin": 10.0,
            "eulerconvention": "tsl",
            "gammavalue": 0.3333,
            "hipassw": 0.05,
            "includebackground": "n",
            "makedictionary": "n",
            "maskpattern": "n",
            "maskradius": 240,
            "masterfile": "crystal_data/ni/ni_mc_mp_20kv.h5",
            "nregions": 10,
            "nthreads": 24,
            "numsx": data_shape[2],
            "numsy": data_shape[1],
            "outputformat": "gui",
            "poisson": "n",
            "scalingmode": "gam",
            "stdout": 6,
            "thetac": 0.0,
            "xpc": 4.86,
            "ypc": 16.58,
        }
    },
    "EMheader": {
        "EBSD": {
            "ProgramName": np.array([b"EMEBSD.f90"], dtype="S10"),
            "Version": np.array([b"5_0_20200217_0"], dtype="S14"),
        },
    },
}

kp.io.plugins.h5ebsd.dict2h5ebsdgroup(dictionary=data, group=f["/"])

f.close()
