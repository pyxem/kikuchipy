# Copyright 2019-2022 The kikuchipy developers
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

from __future__ import annotations
from typing import Optional, Tuple, Union

import numpy as np
from orix.crystal_map import Phase

from kikuchipy.signals._kikuchi_master_pattern import KikuchiMasterPattern
from kikuchipy.signals._kikuchipy_signal import LazyKikuchipySignal2D


class ECPMasterPattern(KikuchiMasterPattern):
    """Simulated Electron Channeling Pattern (ECP) master pattern.

    This class extends HyperSpy's Signal2D class for ECP master
    patterns.

    See the documentation of
    :class:`~hyperspy._signals.signal2d.Signal2D` for the list of
    inherited attributes and methods.

    Parameters
    ----------
    *args
        See :class:`~hyperspy._signals.signal2d.Signal2D`.
    hemisphere : str
        Which hemisphere the data contains, either ``"upper"``,
        ``"lower"``, or ``"both"``.
    phase : ~orix.crystal_map.Phase
        The phase describing the crystal structure used in the master
        pattern simulation.
    projection : str
        Which projection the pattern is in, ``"stereographic"`` or
        ``"lambert"``.
    **kwargs
        See :class:`~hyperspy._signals.signal2d.Signal2D`.
    """

    _signal_type = "ECPMasterPattern"
    _alias_signal_types = ["ecp_master_pattern"]

    # --- Inherited methods and properties from KikuchiMasterPattern
    # overwritten. If the inherited properties or methods are not
    # altered, they are included for documentation purposes. Have to
    # include both getters and setters to include the getter docstring.

    @property
    def _has_multiple_energies(self) -> bool:
        return super()._has_multiple_energies

    @property
    def hemisphere(self) -> str:
        return super().hemisphere

    @hemisphere.setter
    def hemisphere(self, value: str):
        super(ECPMasterPattern, type(self)).hemisphere.fset(self, value)

    @property
    def phase(self) -> Phase:
        return super().phase

    @phase.setter
    def phase(self, value: Phase):
        super(ECPMasterPattern, type(self)).phase.fset(self, value)

    @property
    def projection(self) -> str:
        return super().projection

    @projection.setter
    def projection(self, value: str):
        super(ECPMasterPattern, type(self)).projection.fset(self, value)

    def as_lambert(self, show_progressbar: Optional[bool] = None) -> ECPMasterPattern:
        return super().as_lambert(show_progressbar=show_progressbar)

    def plot_spherical(
        self,
        energy: Union[int, float, None] = None,
        return_figure: bool = False,
        style: str = "surface",
        plotter_kwargs: Union[dict] = None,
        show_kwargs: Union[dict] = None,
    ) -> "pyvista.Plotter":
        return super().plot_spherical(
            energy=energy,
            return_figure=return_figure,
            style=style,
            plotter_kwargs=plotter_kwargs,
            show_kwargs=show_kwargs,
        )

    # --- Inherited methods from CommonImage overwritten

    def normalize_intensity(
        self,
        num_std: int = 1,
        divide_by_square_root: bool = False,
        dtype_out: Union[str, np.dtype, type, None] = None,
        show_progressbar: Optional[bool] = None,
    ) -> None:
        super().normalize_intensity(
            num_std=num_std,
            divide_by_square_root=divide_by_square_root,
            dtype_out=dtype_out,
            show_progressbar=show_progressbar,
        )

    def rescale_intensity(
        self,
        relative: bool = False,
        in_range: Union[Tuple[int, int], Tuple[float, float], None] = None,
        out_range: Union[Tuple[int, int], Tuple[float, float], None] = None,
        dtype_out: Union[
            str, np.dtype, type, Tuple[int, int], Tuple[float, float], None
        ] = None,
        percentiles: Union[Tuple[int, int], Tuple[float, float], None] = None,
        show_progressbar: Optional[bool] = None,
    ) -> None:
        super().rescale_intensity(
            relative=relative,
            in_range=in_range,
            out_range=out_range,
            dtype_out=dtype_out,
            percentiles=percentiles,
            show_progressbar=show_progressbar,
        )

    # --- Inherited methods from Signal2D overwritten

    def deepcopy(self) -> ECPMasterPattern:
        return super().deepcopy()


class LazyECPMasterPattern(LazyKikuchipySignal2D, ECPMasterPattern):
    """Lazy implementation of
    :class:`~kikuchipy.signals.ECPMasterPattern`.

    See the documentation of ``ECPMasterPattern`` for attributes and
    methods.

    This class extends HyperSpy's
    :class:`~hyperspy._signals.signal2d.LazySignal2D` class for ECP
    master patterns. See the documentation of that class for how to
    create this signal and the list of inherited attributes and methods.
    """

    def compute(self, *args, **kwargs) -> None:
        super().compute(*args, **kwargs)
