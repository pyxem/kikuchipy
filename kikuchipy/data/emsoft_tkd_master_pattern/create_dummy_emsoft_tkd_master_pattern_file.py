import os

import numpy as np
import h5py

from kikuchipy.io.plugins._h5ebsd import _dict2hdf5group
import kikuchipy as kp


f = h5py.File(
    os.path.join(os.path.dirname(__file__), "tkd_master_pattern.h5"), mode="w"
)

npx = 6
signal_shape = (npx * 2 + 1,) * 2
energies = np.linspace(10, 20, 11, dtype=np.float32)
data_shape = (len(energies),) + signal_shape

mp_lambert_north = np.ones((1,) + data_shape, dtype=np.float32) * energies.reshape(
    (1, 11, 1, 1)
)
mp_lambert_south = mp_lambert_north
circle = kp.filters.Window(shape=signal_shape).astype(np.float32)
mp_spherical_north = mp_lambert_north.squeeze() * circle
mp_spherical_south = mp_spherical_north

data = {
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
        "TKDmaster": {
            "BetheParameters": np.array([4, 8, 50, 1], dtype=np.float32),
            "EkeVs": np.linspace(10, 20, 11, dtype=np.float32),
            "mLPNH": mp_lambert_north,
            "masterSPNH": mp_spherical_north,
            "masterSPSH": mp_spherical_south,
            "numEbins": len(energies),
            "numset": 1,
        }
    },
    "NMLparameters": {
        "TKDMasterNameList": {"dmin": 0.05, "npx": npx},
        "MCCLfoilNameList": {
            "Ebinsize": energies[1] - energies[0],
            "Ehistmin": np.min(energies),
            "EkeV": np.max(energies),
            "MCmode": "CSDA",
            "dataname": "crystal_data/al2cu/al2cu_mc_mp_20kv.h5",
            "depthmax": 100.0,
            "depthstep": 1.0,
            "numsx": npx,
            "sig": -20.0,
            "totnum_el": 2000000000,
        },
        "BetheList": {"c1": 4.0, "c2": 8.0, "c3": 50.0, "sgdbdiff": 1.0},
    },
    "EMheader": {
        "TKDmaster": {"ProgramName": np.array([b"EMTKDmaster.f90"], dtype="S15")},
    },
}

_dict2hdf5group(dictionary=data, group=f["/"])

# One chunked data set
f["EMData/TKDmaster"].create_dataset("mLPSH", data=mp_lambert_south, chunks=True)

# One byte string with latin-1 stuff
creation_time = b"12:30:13.559 PM\xf0\x14\x1e\xc8\xbcU"
f["CrystalData"].create_dataset("CreationTime", data=creation_time)

f.close()
