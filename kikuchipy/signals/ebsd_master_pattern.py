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

from typing import Optional, Union, List

from hyperspy._signals.signal2d import Signal2D
from hyperspy._lazy_signals import LazySignal2D
from hyperspy.misc.utils import DictionaryTreeBrowser
import numpy as np

from kikuchipy.signals.util._metadata import (
    ebsd_master_pattern_metadata,
    metadata_nodes,
    _update_phase_info,
    _write_parameters_to_dictionary,
)
from kikuchipy.signals._common_image import CommonImage


class EBSDMasterPattern(CommonImage, Signal2D):
    """Simulated Electron Backscatter Diffraction (EBSD) master pattern.

    This class extends HyperSpy's Signal2D class for EBSD master
    patterns.

    Methods inherited from HyperSpy can be found in the HyperSpy user
    guide.

    See the docstring of :class:`hyperspy.signal.BaseSignal` for a list
    of attributes.

    """

    _signal_type = "EBSDMasterPattern"
    _alias_signal_types = ["ebsd_master_pattern", "master_pattern"]
    _lazy = False

    def __init__(self, *args, **kwargs):
        """Create an :class:`~kikuchipy.signals.EBSDMasterPattern`
        object from a :class:`hyperspy.signals.Signal2D` or a
        :class:`numpy.ndarray`.

        """

        Signal2D.__init__(self, *args, **kwargs)

        # Update metadata if object is initialized from numpy array or
        # with set_signal_type()
        if not self.metadata.has_item(metadata_nodes("ebsd_master_pattern")):
            md = self.metadata.as_dictionary()
            md.update(ebsd_master_pattern_metadata().as_dictionary())
            self.metadata = DictionaryTreeBrowser(md)
        if not self.metadata.has_item("Sample.Phases"):
            self.set_phase_parameters()

    def set_simulation_parameters(
        self,
        complete_cutoff: Union[None, int, float] = None,
        depth_step: Union[None, int, float] = None,
        energy_step: Union[None, int, float] = None,
        hemisphere: Union[None, str] = None,
        incident_beam_energy: Union[None, int, float] = None,
        max_depth: Union[None, int, float] = None,
        min_beam_energy: Union[None, int, float] = None,
        mode: Optional[str] = None,
        number_of_electrons: Optional[int] = None,
        pixels_along_x: Optional[int] = None,
        projection: Union[None, str] = None,
        sample_tilt: Union[None, int, float] = None,
        smallest_interplanar_spacing: Union[None, int, float] = None,
        strong_beam_cutoff: Union[None, int, float] = None,
        weak_beam_cutoff: Union[None, int, float] = None,
    ):
        """Set simulated parameters in signal metadata.

        Parameters
        ----------
        complete_cutoff
            Bethe parameter c3.
        depth_step
            Material penetration depth step size, in nm.
        energy_step
            Energy bin size, in keV.
        hemisphere
            Which hemisphere(s) the data contains.
        incident_beam_energy
            Incident beam energy, in keV.
        max_depth
            Maximum material penetration depth, in nm.
        min_beam_energy
            Minimum electron energy to consider, in keV.
        mode
            Simulation mode, e.g. Continuous slowing down
            approximation (CSDA) used by EMsoft.
        number_of_electrons
            Total number of incident electrons.
        pixels_along_x
            Pixels along horizontal direction.
        projection
            Which projection the pattern is in.
        sample_tilt
            Sample tilte angle from horizontal, in degrees.
        smallest_interplanar_spacing
            Smallest interplanar spacing, d-spacing, taken into account
            in the computation of the electrostatic lattice potential,
            in nm.
        strong_beam_cutoff
            Bethe parameter c1.
        weak_beam_cutoff
            Bethe parameter c2.

        See Also
        --------
        set_phase_parameters

        Examples
        --------
        >>> import kikuchipy as kp
        >>> ebsd_mp_node = kp.signals.util.metadata_nodes(
        ...     "ebsd_master_pattern")
        >>> s.metadata.get_item(ebsd_mp_node + '.incident_beam_energy')
        15.0
        >>> s.set_simulated_parameters(incident_beam_energy=20.5)
        >>> s.metadata.get_item(ebsd_mp_node + '.incident_beam_energy')
        20.5
        """
        md = self.metadata
        ebsd_mp_node = metadata_nodes("ebsd_master_pattern")
        _write_parameters_to_dictionary(
            {
                "BSE_simulation": {
                    "depth_step": depth_step,
                    "energy_step": energy_step,
                    "incident_beam_energy": incident_beam_energy,
                    "max_depth": max_depth,
                    "min_beam_energy": min_beam_energy,
                    "mode": mode,
                    "number_of_electrons": number_of_electrons,
                    "pixels_along_x": pixels_along_x,
                    "sample_tilt": sample_tilt,
                },
                "Master_pattern": {
                    "Bethe_parameters": {
                        "complete_cutoff": complete_cutoff,
                        "strong_beam_cutoff": strong_beam_cutoff,
                        "weak_beam_cutoff": weak_beam_cutoff,
                    },
                    "smallest_interplanar_spacing": smallest_interplanar_spacing,
                    "projection": projection,
                    "hemisphere": hemisphere,
                },
            },
            md,
            ebsd_mp_node,
        )

    def set_phase_parameters(
        self,
        number: int = 1,
        atom_coordinates: Optional[dict] = None,
        formula: Optional[str] = None,
        info: Optional[str] = None,
        lattice_constants: Union[
            None, np.ndarray, List[float], List[int]
        ] = None,
        laue_group: Optional[str] = None,
        material_name: Optional[str] = None,
        point_group: Optional[str] = None,
        setting: Optional[int] = None,
        source: Optional[str] = None,
        space_group: Optional[int] = None,
        symmetry: Optional[int] = None,
    ):
        """Set parameters for one phase in signal metadata.

        A phase node with default values is created if none is present
        in the metadata when this method is called.

        Parameters
        ----------
        number
            Phase number.
        atom_coordinates
            Dictionary of dictionaries with one or more of the atoms in
            the unit cell, on the form `{'1': {'atom': 'Ni',
            'coordinates': [0, 0, 0], 'site_occupation': 1,
            'debye_waller_factor': 0}, '2': {'atom': 'O',... etc.`
            `debye_waller_factor` in units of nm^2, and
            `site_occupation` in range [0, 1].
        formula
            Phase formula, e.g. 'Fe2' or 'Ni'.
        info
            Whatever phase info the user finds relevant.
        lattice_constants
            Six lattice constants a, b, c, alpha, beta, gamma.
        laue_group
            Phase Laue group.
        material_name
            Name of material.
        point_group
            Phase point group.
        setting
            Space group's origin setting.
        source
            Literature reference for phase data.
        space_group
            Number between 1 and 230.
        symmetry
            Phase symmetry.

        See Also
        --------
        set_simulation_parameters

        Examples
        --------
        >>> s.metadata.Sample.Phases.Number_1.atom_coordinates.Number_1
        ├── atom =
        ├── coordinates = array([0., 0., 0.])
        ├── debye_waller_factor = 0.0
        └── site_occupation = 0.0
        >>> s.set_phase_parameters(
        ...     number=1, atom_coordinates={
        ...         '1': {'atom': 'Ni', 'coordinates': [0, 0, 0],
        ...         'site_occupation': 1,
        ...         'debye_waller_factor': 0.0035}})
        >>> s.metadata.Sample.Phases.Number_1.atom_coordinates.Number_1
        ├── atom = Ni
        ├── coordinates = array([0., 0., 0.])
        ├── debye_waller_factor = 0.0035
        └── site_occupation = 1
        """
        # Ensure atom coordinates are numpy arrays
        if atom_coordinates is not None:
            for phase, val in atom_coordinates.items():
                atom_coordinates[phase]["coordinates"] = np.array(
                    atom_coordinates[phase]["coordinates"]
                )

        inputs = {
            "atom_coordinates": atom_coordinates,
            "formula": formula,
            "info": info,
            "lattice_constants": lattice_constants,
            "laue_group": laue_group,
            "material_name": material_name,
            "point_group": point_group,
            "setting": setting,
            "source": source,
            "space_group": space_group,
            "symmetry": symmetry,
        }

        # Remove None values
        phase = {k: v for k, v in inputs.items() if v is not None}
        _update_phase_info(self.metadata, phase, number)


class LazyEBSDMasterPattern(EBSDMasterPattern, LazySignal2D):
    """Lazy implementation of the :class:`EBSDMasterPattern` class.

    This class extends HyperSpy's LazySignal2D class for EBSD master
    patterns.

    Methods inherited from HyperSpy can be found in the HyperSpy user
    guide.

    See docstring of :class:`EBSDMasterPattern` for attributes and
    methods.

    """

    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
