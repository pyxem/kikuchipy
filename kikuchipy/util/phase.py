# -*- coding: utf-8 -*-
# Copyright 2019-2020 The KikuchiPy developers
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

import numpy as np
import re
from hyperspy.misc.utils import DictionaryTreeBrowser


def _phase_metadata():
    """Return a dictionary with a default KikuchiPy phase structure.

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
    }
    return pd


def _update_phase_info(metadata, dictionary, phase_number=1):
    """Update information of phase in metadata, adding it if it doesn't
    already exist.

    Parameters
    ----------
    metadata : DictionaryTreeBrowser
        Metadata to update.
    dictionary : dict
        Dictionary with only values to update.
    phase_number : int, optional
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
