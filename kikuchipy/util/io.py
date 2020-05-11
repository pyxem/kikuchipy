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

from typing import Union, Tuple, Any, List
import warnings

from hyperspy.misc.utils import DictionaryTreeBrowser
from hyperspy.exceptions import VisibleDeprecationWarning


def ebsd_metadata() -> DictionaryTreeBrowser:
    """Return a dictionary in HyperSpy's DictionaryTreeBrowser format
    with the default kikuchipy EBSD metadata.

    See :meth:`~kikuchipy.signals.ebsd.EBSD.set_experimental_parameters`
    for an explanation of the parameters.

    Returns
    -------
    md : hyperspy.misc.utils.DictionaryTreeBrowser

    """

    md = DictionaryTreeBrowser()
    sem_node, ebsd_node = metadata_nodes(["sem", "ebsd"])
    ebsd = {
        "azimuth_angle": -1.0,
        "binning": -1.0,
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


def kikuchipy_metadata() -> DictionaryTreeBrowser:
    warnings.warn(
        "This function is deprecated in favor of 'ebsd_metadata()' due to more "
        "signals being added, and will be removed in v0.3.",
        VisibleDeprecationWarning,
    )
    return ebsd_metadata()


def ebsd_master_pattern_metadata() -> DictionaryTreeBrowser:
    """Return a dictionary in HyperSpy's DictionaryTreeBrowser format
    with the default kikuchipy EBSD master pattern metadata.

    The parameters are chosen based on the contents in EMsoft's EBSD
    master pattern HDF5 file.

    See
    :meth:`~kikuchipy.signals.ebsd_master_pattern.EBSDMasterPattern.set_simulation_parameters`
    for an explanation of the parameters.

    Returns
    -------
    md : DictionaryTreeBrowser

    """

    ebsd_master_pattern = {
        "BSE_simulation": {
            "depth_step": -1.0,
            "energy_step": -1.0,
            "incident_beam_energy": -1.0,
            "max_depth": -1.0,
            "min_beam_energy": -1.0,
            "mode": "",
            "number_of_electrons": -1,
            "pixels_along_x": -1,
            "sample_tilt": -1.0,
        },
        "Master_pattern": {
            "Bethe_parameters": {
                "complete_cutoff": -1.0,
                "strong_beam_cutoff": -1.0,
                "weak_beam_cutoff": -1.0,
            },
            "smallest_interplanar_spacing": -1.0,
            "projection": "",
            "hemisphere": "",
        },
    }

    md = DictionaryTreeBrowser()
    md.set_item(metadata_nodes("ebsd_master_pattern"), ebsd_master_pattern)

    return md


def metadata_nodes(
    nodes: Union[None, str, List[str]] = None
) -> Union[List[str], str, List]:
    """Return SEM, EBSD and/or EBSD master pattern metadata nodes.

    This is a convenience function so that we only have to define these
    node strings here.

    Parameters
    ----------
    nodes
        Metadata nodes to return. Options are "sem", "ebsd",
        "ebsd_master_pattern" or None. If None (default) is passed, all
        nodes are returned.

    Returns
    -------
    nodes_to_return

    """

    available_nodes = {
        "sem": "Acquisition_instrument.SEM",
        "ebsd": "Acquisition_instrument.SEM.Detector.EBSD",
        "ebsd_master_pattern": "Simulation.EBSD_master_pattern",
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


def _get_input_bool(question: str) -> bool:
    """Get input from user on boolean choice, returning the answer.

    Parameters
    ----------
    question
        Question to ask user.

    """

    try:
        answer = input(question)
        answer = answer.lower()
        while (answer != "y") and (answer != "n"):
            print("Please answer y or n.")
            answer = input(question)
        if answer.lower() == "y":
            return True
        elif answer.lower() == "n":
            return False
    except OSError:
        warnings.warn(
            "Your terminal does not support raw input. Not adding scan. To add "
            "the scan use `add_scan=True`"
        )
        return False


def _get_input_variable(question: str, var_type: Any) -> Union[None, Any]:
    """Get variable input from user, returning the variable.

    Parameters
    ----------
    question
        Question to ask user.
    var_type
        Type of variable to return.

    """

    try:
        answer = input(question)
        while type(answer) != var_type:
            try:
                answer = var_type(answer)
            except (TypeError, ValueError):
                print(f"Please enter a variable of type {var_type}:\n")
                answer = input(question)
        return answer
    except OSError:
        warnings.warn(
            "Your terminal does not support raw input. Not adding scan. To add "
            "the scan use `add_scan=True`"
        )
        return None
