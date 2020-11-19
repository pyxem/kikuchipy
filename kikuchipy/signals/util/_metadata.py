# -*- coding: utf-8 -*-
# Copyright 2019-2020 The kikuchipy developers
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

import re
from typing import Union, List
import warnings

import numpy as np
from hyperspy.misc.utils import DictionaryTreeBrowser

from kikuchipy.io._util import _get_nested_dictionary


def ebsd_metadata() -> DictionaryTreeBrowser:
    """Return a dictionary in HyperSpy's DictionaryTreeBrowser format
    with the default kikuchipy EBSD metadata.

    See :meth:`~kikuchipy.signals.EBSD.set_experimental_parameters` for
    an explanation of the parameters.

    Returns
    -------
    md : hyperspy.misc.utils.DictionaryTreeBrowser
    """
    md = DictionaryTreeBrowser()
    sem_node, ebsd_node = metadata_nodes(["sem", "ebsd"])
    ebsd = {
        "azimuth_angle": -1.0,
        "binning": 1,
        "detector": "",
        "elevation_angle": -1.0,
        "exposure_time": -1,
        "frame_number": -1,
        "frame_rate": -1,
        "gain": -1.0,
        "grid_type": "",
        "sample_tilt": -1.0,
        "scan_time": -1.0,
        "static_background": -1,
        "xpc": -1.0,
        "ypc": -1.0,
        "zpc": -1.0,
    }
    sem = {
        "microscope": "",
        "magnification": -1,
        "beam_energy": -1.0,
        "working_distance": -1.0,
    }
    md.set_item(sem_node, sem)
    md.set_item(ebsd_node, ebsd)
    return md


def metadata_nodes(
    nodes: Union[None, str, List[str]] = None
) -> Union[List[str], str, List]:
    """Return SEM and/or EBSD metadata nodes.

    This is a convenience function so that we only have to define these
    node strings here.

    Parameters
    ----------
    nodes
        Metadata nodes to return. Options are "sem", "ebsd", or None.
        If None (default) is passed, all nodes are returned.

    Returns
    -------
    nodes_to_return : list of str or str
    """
    available_nodes = {
        "sem": "Acquisition_instrument.SEM",
        "ebsd": "Acquisition_instrument.SEM.Detector.EBSD",
    }

    if nodes is None:
        nodes_to_return = list(available_nodes.values())
    else:
        if not hasattr(nodes, "__iter__") or isinstance(nodes, str):
            nodes = (nodes,)  # Make iterable
        nodes_to_return = []
        for n in nodes:
            for name, node in available_nodes.items():
                if n == name:
                    nodes_to_return.append(node)

    if len(nodes_to_return) == 1:
        nodes_to_return = nodes_to_return[0]
    return nodes_to_return


def _phase_metadata() -> dict:
    """Return a dictionary with a default kikuchipy phase structure.

    Returns
    -------
    pd : dict
    """
    pd = {
        "atom_coordinates": {
            "1": {
                "atom": "",
                "coordinates": np.zeros(3),
                "site_occupation": 0.0,
                "debye_waller_factor": 0.0,
            }
        },
        "formula": "",
        "info": "",
        "lattice_constants": np.zeros(6),
        "laue_group": "",
        "material_name": "",
        "point_group": "",
        "setting": 0,
        "space_group": 0,
        "symmetry": 0,
        "source": "",
    }
    return pd


def _update_phase_info(
    metadata: DictionaryTreeBrowser, dictionary: dict, phase_number: int = 1
) -> DictionaryTreeBrowser:
    """Update information of phase in metadata, adding it if it doesn't
    already exist.

    Parameters
    ----------
    metadata
        Metadata to update.
    dictionary
        Dictionary with only values to update.
    phase_number
        Number of phase to update.

    Returns
    -------
    metadata : DictionaryTreeBrowser
        Updated metadata.
    """
    # Check if metadata has phases
    if not metadata.has_item("Sample.Phases"):
        metadata.add_node("Sample.Phases")

    # Check if phase number is already in metadata
    phase = metadata.Sample.Phases.get_item(str(phase_number))
    if phase is None:
        phase = _phase_metadata()
    phase = dict(phase)

    # Loop over input dictionary and update items in phase dictionary
    for key, val in dictionary.items():
        key = re.sub(r"(\w)([A-Z])", r"\1 \2", key)  # Space before UPPERCASE
        key = key.lower()
        key = key.replace(" ", "_")
        if key in phase:
            if isinstance(val, list):
                val = np.array(val)
            phase[key] = val

    # Update phase info in metadata
    metadata.Sample.Phases.add_dictionary({str(phase_number): phase})

    return metadata


def _write_parameters_to_dictionary(
    parameters: dict, dictionary: DictionaryTreeBrowser, node: str
):
    """Write dictionary of parameters to DictionaryTreeBrowser.

    Parameters
    ----------
    parameters
        Dictionary of parameters to write to dictionary.
    dictionary
        Dictionary to write parameters to.
    node
        String like 'Acquisition_instrument.SEM' etc. with dictionary
        nodes to write parameters to.
    """
    for key, val in parameters.items():
        if val is not None:
            dictionary.set_item(node + "." + key, val)


def _set_metadata_from_mapping(
    omd: dict, md: DictionaryTreeBrowser, mapping: dict,
):
    """Update metadata dictionary inplace from original metadata
    dictionary via a mapping.

    Parameters
    ----------
    omd
        Dictionary with original metadata.
    md
        Dictionary with metadata to update.
    mapping
        Mapping between `omd` and `md`.
    """
    for key_out, key_in in mapping.items():
        try:
            if isinstance(key_in, list):
                value = _get_nested_dictionary(omd, key_in)
            else:
                value = omd[key_in]
            md.set_item(key_out, value)
        except KeyError:
            warnings.warn(f"Could not read {key_in} from file.")
