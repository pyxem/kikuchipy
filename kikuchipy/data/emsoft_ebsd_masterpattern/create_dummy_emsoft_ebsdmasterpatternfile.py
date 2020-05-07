import os

import numpy as np
import h5py
import kikuchipy as kp


ddir = "/home/hakon/kode/kikuchipy/kikuchipy/data/emsoft_ebsd_masterpattern/"
fname = "master_patterns.h5"
f = h5py.File(os.path.join(ddir, fname), mode="w")

npx = 6
energies = np.linspace(10, 20, 11, dtype=np.float32)
data_shape = (len(energies),) + (npx * 2 + 1,) * 2
data = {
    "EMData": {
        "EBSDmaster": {
            "BetheParameters": np.array([4, 8, 50, 1], dtype=np.float32),
            "EkeVs": np.linspace(10, 20, 11, dtype=np.float32),
            "mLPNH": np.ones((1,) + data_shape, dtype=np.float32),
            #            "mLPSH": np.ones((1,) + data_shape, dtype=np.float32),
            "masterSPNH": np.ones(data_shape, dtype=np.float32),
            "masterSPSH": np.ones(data_shape, dtype=np.float32),
            "numEbins": len(energies),
        }
    },
    "NMLparameters": {
        "EBSDMasterNameList": {
            "dmin": 0.05,
            "latgridtype": np.array([b"Lambert"], dtype="S7"),
            "npx": npx,
        },
    },
    "EMheader": {
        "EBSDmaster": {
            "ProgramName": np.array([b"EMEBSDmaster.f90"], dtype="S16")
        },
    },
}

kp.io.plugins.h5ebsd.dict2h5ebsdgroup(dictionary=data, group=f["/"])

# One chunked data set
mLPSH = np.ones((1,) + data_shape, dtype=np.float32)
f["EMData/EBSDmaster"].create_dataset("mLPSH", data=mLPSH, chunks=True)

f.close()
