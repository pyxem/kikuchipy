#
# Copyright 2026 the kikuchipy developers
#
# SPDX-License-Identifier: BSD-3-Clause
#

from kikuchipy._constants import verify_dependency_or_raise

verify_dependency_or_raise("ebsdsim", "Reading of an EBSD master pattern from ebsdsim")

from pathlib import Path
from typing import Any, Literal, TypedDict, get_args

from diffpy.structure import Atom, Lattice, Structure
import ebsdsim as es
import numpy as np

HEMISPHERE = Literal["upper", "lower", "both"]


class ebsdsimSite(TypedDict):
    index: int
    atomic_number: int
    symbol: str
    fract: list[float]
    occupancy: float
    b_iso_angstrom_sq: float
    b_iso_nm_sq: float
    multiplicity: int


class ebsdsimCell(TypedDict):
    a_angstrom: float
    b_angstrom: float
    c_angstrom: float
    alpha_deg: float
    beta_deg: float
    gamma_deg: float
    volume_angstrom3: float
    density_g_cm3: float
    lattice_centering: Literal["A", "B", "C", "F", "I", "R", "H", "P"]
    space_group: int
    pg_num: int
    average_atomic_number: float
    average_atomic_weight: float
    n_sites: int
    sites: list[ebsdsimSite]


def get_orix_phase_dictionary_from_ebsdsim_cell(cell: ebsdsimCell) -> dict[str, Any]:
    """Return a dictionary that can be passed directly to the orix phase
    constructor, parsed from the ebsdsim cell dictionary.
    """
    sg = cell["space_group"]
    atoms = []
    for site in cell["sites"]:
        atom = Atom(
            atype=site["symbol"],
            xyz=site["fract"],
            occupancy=site["occupancy"],
            Uisoequiv=site["b_iso_angstrom_sq"] / (8 * np.pi**2),
        )
        atoms.append(atom)
    lattice = Lattice(
        a=cell["a_angstrom"],
        b=cell["b_angstrom"],
        c=cell["c_angstrom"],
        alpha=cell["alpha_deg"],
        beta=cell["beta_deg"],
        gamma=cell["gamma_deg"],
    )
    structure = Structure(lattice=lattice, atoms=atoms)

    return {"space_group": sg, "structure": structure}


def get_upper_energy_bin_edges_kv(
    voltages: np.ndarray, meta: dict[str, Any]
) -> np.ndarray:
    """Return the upper edge in kV of each energy bin, given the bin
    *voltages* and the *meta* data read from an ebsdsim file.

    ebsdsim < 0.1.6:

        bin_voltages_kv = voltage_kv - (i + 0.5) * energy_binwidth_keV.

    ebsdsim >= 0.1.6:

        bin_voltages_kv = voltage_kv - i * energy_binwidth_keV.

    Nothing in the file distinguishes the two (``format_version`` did
    not change across the switch), so which one it is has to be
    inferred: centers sit half an energy bin width away from a whole
    number of bins below the beam voltage, upper edges sit a whole
    number of bins below it.

    Files whose metadata does not give both the beam voltage and the
    energy bin width are assumed to hold upper edges.
    """
    beam_kv = meta.get("voltage_kv")
    binwidth = meta.get("energy_binwidth_keV")
    if beam_kv is None or binwidth is None or binwidth <= 0:
        return voltages

    # Distance from the beam voltage in whole energy bins: ~i for upper
    # edges, ~i + 0.5 for centers
    offsets = (float(beam_kv) - voltages.astype(np.float64)) / float(binwidth)
    fractions = np.abs(offsets - np.round(offsets))
    if np.median(fractions) > 0.25:
        voltages += np.float32(binwidth / 2)

    return voltages


def file_reader(
    filename: str | Path, hemisphere: HEMISPHERE = "upper", lazy: bool = False
) -> list[dict]:
    """Read an ebsdsim EBSD master pattern from a NumPy .npz file.

    Not meant to be used directly; use :func:`~kikuchipy.load`.

    Parameters
    ----------
    filename
        Path to the .npz file written by
        :func:`ebsdsim.save_master_pattern`.
    hemisphere
        Which hemisphere to return: "upper" (default), "lower", or
        "both".
    lazy
        Not supported; included for API compatibility. Data are always
        loaded eagerly.

    Returns
    -------
    signal_dict_list
        Data, axes, metadata, original metadata, and a phase dict.

    Notes
    -----
    This function carries a BSD 3-Clause license.
    """
    hemi = hemisphere.lower()
    hemi_options = get_args(HEMISPHERE)
    if hemi not in hemi_options:
        hemi_options_str = ", ".join(hemi_options)
        raise ValueError(
            f"Hemisphere must be one of {hemi_options_str}, not {hemisphere!r}"
        )

    fpath = Path(filename)

    loaded = es.load_master_pattern(fpath)

    # loaded.data shape: (E, S, H, side, side)
    # Take site-integrated (S index 0) and drop that axis.
    energy_data = loaded.data[:, 0, :, :, :]  # (E, H, side, side)

    axes_meta = loaded.axes
    n_bins = loaded.n_bins
    n_hemispheres = axes_meta["hemisphere_dim"]

    # Use per-bin energy slices when available; fall back to the
    # integrated slice
    if n_bins > 0 and axes_meta["energy_dim"] > 1:
        bin_indices = axes_meta["bin_to_energy_index"]
        # (n_bins, H, side, side)
        energy_data = energy_data[bin_indices, :, :, :]
        voltages = get_upper_energy_bin_edges_kv(
            np.asarray(loaded.bin_voltages_kv, dtype=np.float32), loaded.meta
        )
    else:
        energy_data = energy_data[0:1, :, :, :]  # (1, H, side, side)
        voltages = np.zeros(1, dtype=np.float32)

    n_energy = energy_data.shape[0]

    # Hemisphere selection
    if hemi == "upper":
        data = energy_data[:, 0, :, :]  # (n_energy, side, side)
    elif hemi == "lower":
        if n_hemispheres < 2:
            raise ValueError(
                "File does not contain lower hemisphere data; use hemisphere='upper'"
            )
        data = energy_data[:, 1, :, :]  # (n_energy, side, side)
    else:  # "both"
        # For centrosymmetric materials, ebsdsim only stores the
        # northern hemisphere (hemisphere_dim=1). Duplicate it so "both"
        # always yields a size-2 hemisphere axis, consistent with EMsoft
        # behaviour.
        nh = energy_data[:, 0:1, :, :]  # (n_energy, 1, side, side)
        if n_hemispheres == 1:
            sh = nh
        else:
            sh = energy_data[:, 1:2, :, :]
        data = np.moveaxis(
            np.concatenate([nh, sh], axis=1), 1, 0
        )  # (2, n_energy, side, side)

    # Squeeze size-1 leading navigation axes so single-energy files
    # produce a plain 2-D (or 1-D navigation) signal.
    if n_energy == 1:
        if hemi == "both":
            data = data[:, 0, :, :]  # (2, side, side)
        else:
            data = data[0, :, :]  # (side, side)

    # Build axes list
    side = loaded.side
    axes = []
    idx = 0

    if hemi == "both":
        axes.append(
            {
                "size": 2,
                "index_in_array": idx,
                "name": "hemisphere",
                "scale": 1.0,
                "offset": 0.0,
                "units": "",
            }
        )
        idx += 1

    if n_energy > 1:
        # Voltages are the upper edge of each energy bin, so they can be
        # used as the energy axis values directly
        energy_scale = float(np.mean(np.diff(voltages)))
        energy_offset = float(voltages[0])
        axes.append(
            {
                "size": n_energy,
                "index_in_array": idx,
                "name": "energy",
                "scale": energy_scale,
                "offset": energy_offset,
                "units": "keV",
            }
        )
        idx += 1

    axes.append(
        {
            "size": side,
            "index_in_array": idx,
            "name": "height",
            "scale": 1.0,
            "offset": -(side // 2),
            "units": "px",
        }
    )
    axes.append(
        {
            "size": side,
            "index_in_array": idx + 1,
            "name": "width",
            "scale": 1.0,
            "offset": -(side // 2),
            "units": "px",
        }
    )

    # Metadata
    md = {
        "Signal": {"signal_type": "EBSDMasterPattern", "record_by": "image"},
        "General": {"title": fpath.stem, "original_filename": fpath.name},
    }

    # Return the raw cell dictionary so that the caller can build an
    # orix phase
    phase_dict = get_orix_phase_dictionary_from_ebsdsim_cell(loaded.meta["cell"])
    source = loaded.meta.get("source", None)
    if source is not None:
        name = Path(source).stem
    else:
        name = fpath.stem
    phase_dict["name"] = name

    return [
        {
            "data": data,
            "axes": axes,
            "metadata": md,
            "original_metadata": loaded.meta,
            "projection": "lambert",
            "phase": phase_dict,
            "hemisphere": hemisphere,
        }
    ]
