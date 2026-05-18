#
# Copyright 2019-2026 the kikuchipy developers
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
#

"""Widget helpers for interactive EBSD detector plots.

Requires :mod:`ipywidgets`.
"""

from kikuchipy._constants import verify_dependency_or_raise

verify_dependency_or_raise("ipywidgets", "Interactive detector plots")

import abc
from typing import Any, Literal

from diffsims.crystallography import ReciprocalLatticeVector
import ipywidgets
import matplotlib.axes as maxes
import matplotlib.collections as mcollections
import matplotlib.figure as mfigure
import matplotlib.pyplot as plt
import numpy as np
import orix.quaternion as oqu

from kikuchipy.detectors._ebsd_detector import EBSDDetector
from kikuchipy.draw._ebsd_detector_plot import (
    DETECTOR_PLOT_FORMATS,
    set_up_figure_axis,
    update_detector_plane,
    update_detector_sample_geometry_side_view,
    update_detector_sample_geometry_top_view,
)
from kikuchipy.signals.ebsd_master_pattern import EBSDMasterPattern
from kikuchipy.signals.util._master_pattern import (
    _get_direction_cosines_from_detector,
    _project_single_pattern_from_master_pattern,
)
from kikuchipy.simulations._geometrical_simulation import (
    _get_geometrical_simulation,
    _get_zone_axes_from_hkl,
)


def get_detector_value_range(
    value: float, vmin: float, vmax: float
) -> tuple[float, float]:
    """Return an appropriate range containing *value*."""
    margin = max((vmax - vmin) * 0.1, 0.1)
    if value < vmin:
        vmin = value - margin
    if value > vmax:
        vmax = value + margin
    return vmin, vmax


def get_slider_style() -> dict[str, str]:
    return {"description_width": "initial"}


def get_sample_tilt_slider(detector: EBSDDetector) -> ipywidgets.FloatSlider:
    """Return an :mod:`ipywidgets` slider from a *detector*."""
    stilt = detector.sample_tilt
    stilt_min, stilt_max = get_detector_value_range(stilt, -90, 180)

    style = get_slider_style()
    widget = ipywidgets.FloatSlider(
        value=stilt,
        min=stilt_min,
        max=stilt_max,
        step=0.1,
        description="Sample tilt",
        style=style,
    )

    return widget


def get_detector_tilt_slider(detector: EBSDDetector) -> ipywidgets.FloatSlider:
    """Return an :mod:`ipywidgets` slider from a *detector*."""
    tilt = detector.tilt
    tilt_min, tilt_max = get_detector_value_range(tilt, 0, 180)

    style = get_slider_style()
    widget = ipywidgets.FloatSlider(
        value=tilt,
        min=tilt_min,
        max=tilt_max,
        step=0.1,
        description="Detector tilt",
        style=style,
    )

    return widget


def get_detector_azimuthal_slider(detector: EBSDDetector) -> ipywidgets.FloatSlider:
    """Return an :mod:`ipywidgets` slider from a *detector*."""
    azim = detector.azimuthal
    azim_min, azim_max = get_detector_value_range(azim, -10, 10)

    style = get_slider_style()
    widget = ipywidgets.FloatSlider(
        value=azim,
        min=azim_min,
        max=azim_max,
        step=0.01,
        description="Azimuthal",
        style=style,
    )

    return widget


def get_detector_twist_slider(detector: EBSDDetector) -> ipywidgets.FloatSlider:
    """Return an :mod:`ipywidgets` slider from a *detector*."""
    twist = detector.twist
    twist_min, twist_max = get_detector_value_range(twist, -10, 10)

    style = get_slider_style()
    widget = ipywidgets.FloatSlider(
        value=twist,
        min=twist_min,
        max=twist_max,
        step=0.1,
        description="Twist",
        style=style,
    )

    return widget


def get_pcx_slider(detector: EBSDDetector) -> ipywidgets.FloatSlider:
    """Return an :mod:`ipywidgets` slider from a *detector*."""
    pcx = detector.pc_average[0]
    pcx_min, pcx_max = get_detector_value_range(pcx, 0, 1)

    style = get_slider_style()
    widget = ipywidgets.FloatSlider(
        value=pcx,
        min=pcx_min,
        max=pcx_max,
        step=0.01,
        description="PCx",
        style=style,
    )

    return widget


def get_pcy_slider(detector: EBSDDetector) -> ipywidgets.FloatSlider:
    """Return an :mod:`ipywidgets` slider from a *detector*."""
    pcy = detector.pc_average[1]
    pcy_min, pcy_max = get_detector_value_range(pcy, -0.5, 1.5)

    style = get_slider_style()
    widget = ipywidgets.FloatSlider(
        value=pcy,
        min=pcy_min,
        max=pcy_max,
        step=0.01,
        description="PCy",
        style=style,
    )

    return widget


def get_pcz_slider(detector: EBSDDetector) -> ipywidgets.FloatSlider:
    """Return an :mod:`ipywidgets` slider from a *detector*."""
    pcz = detector.pc_average[2]
    pcz_min, pcz_max = get_detector_value_range(pcz, 0.2, 1)

    style = get_slider_style()
    widget = ipywidgets.FloatSlider(
        value=pcz,
        min=pcz_min,
        max=pcz_max,
        step=0.01,
        description="PCz",
        style=style,
    )

    return widget


def get_phi1_slider(angle: float) -> ipywidgets.FloatSlider:
    r"""Return an :mod:`ipywidgets` slider from the :math:`\phi_1` Euler
    angle in degrees.
    """
    style = get_slider_style()
    widget = ipywidgets.FloatSlider(
        value=angle,
        min=0.0,
        max=360.0,
        step=0.1,
        description="\u03c61 (\u00b0)",
        style=style,
    )

    return widget


def get_Phi_slider(angle: float) -> ipywidgets.FloatSlider:
    r"""Return an :mod:`ipywidgets` slider from the :math:`\Phi` Euler
    angle in degrees.
    """
    style = get_slider_style()
    widget = ipywidgets.FloatSlider(
        value=angle,
        min=0.0,
        max=180.0,
        step=0.1,
        description="\u03a6 (\u00b0)",
        style=style,
    )

    return widget


def get_phi2_slider(angle: float) -> ipywidgets.FloatSlider:
    r"""Return an :mod:`ipywidgets` slider from the :math:`\phi_2` Euler
    angle in degrees.
    """
    style = get_slider_style()
    widget = ipywidgets.FloatSlider(
        value=angle,
        min=0.0,
        max=360.0,
        step=0.1,
        description="\u03c62 (\u00b0)",
        style=style,
    )

    return widget


def combine_widgets(
    detector_widgets: list[ipywidgets.FloatSlider],
    rotation_widgets: list[ipywidgets.FloatSlider],
    simulation_checkbox: ipywidgets.Checkbox | None,
) -> ipywidgets.VBox | ipywidgets.HBox:
    det_col = ipywidgets.VBox(
        [ipywidgets.HTML("<b>Detector-sample geometry</b>")] + detector_widgets
    )
    if simulation_checkbox is not None:
        rot_col = ipywidgets.VBox(
            [ipywidgets.HTML("<b>Crystal orientation</b>")] + rotation_widgets
        )
        sim_col = ipywidgets.VBox(
            [ipywidgets.HTML("<b>Simulation</b>"), simulation_checkbox]
        )
        controls = ipywidgets.HBox([det_col, rot_col, sim_col])
    else:
        controls = det_col
    return controls


def plot_detector_sample_geometry_side_view_interactive(
    detector: EBSDDetector,
    ax: maxes.Axes | None = None,
    legend: bool = False,
    dimensionless: bool = True,
    **kwargs,
) -> tuple[ipywidgets.VBox, mfigure.Figure | mfigure.SubFigure]:
    """See the docstring of
    :meth:`~kikuchipy.detectors.EBSDDetector.plot_side_view` for
    details.
    """
    detector.pc = detector.pc_average
    sample_tilt_slider = get_sample_tilt_slider(detector)
    detector_tilt_slider = get_detector_tilt_slider(detector)
    pcy_slider = get_pcy_slider(detector)
    pcz_slider = get_pcz_slider(detector)
    sliders = [
        sample_tilt_slider,
        detector_tilt_slider,
        pcy_slider,
        pcz_slider,
    ]

    fig, ax = set_up_figure_axis(ax=ax, **kwargs)
    ax.set_title("Side view")

    def redraw(*args: Any) -> None:
        update_detector_sample_geometry_side_view(
            detector, ax, legend=legend, dimensionless=dimensionless
        )
        fig.canvas.draw_idle()

    def update_detector_from_sliders():
        detector.sample_tilt = sample_tilt_slider.value
        detector.tilt = detector_tilt_slider.value
        detector.pcy = pcy_slider.value
        detector.pcz = pcz_slider.value

    if detector._has_signals:

        def on_slider_change(change: Any = None) -> None:
            # Block to redraw only once
            with (
                detector._sample_tilt_changed.blocked(),
                detector._tilt_changed.blocked(),
                detector._azimuthal_changed.blocked(),
                detector._pc_changed.blocked(),
            ):
                update_detector_from_sliders()
            redraw()
    else:

        def on_slider_change(change: Any = None) -> None:
            update_detector_from_sliders()
            redraw()

    for slider in sliders:
        slider.observe(on_slider_change, names="value")

    redraw()  # Initial draw

    controls = ipywidgets.VBox(sliders)

    return controls, fig


def plot_detector_sample_geometry_top_view_interactive(
    detector: EBSDDetector,
    ax: maxes.Axes | None = None,
    legend: bool = False,
    dimensionless: bool = True,
    **kwargs,
) -> tuple[ipywidgets.VBox, mfigure.Figure | mfigure.SubFigure]:
    """See the docstring of
    :meth:`~kikuchipy.detectors.EBSDDetector.plot_top_view` for
    details.
    """
    detector.pc = detector.pc_average
    azimuthal_slider = get_detector_azimuthal_slider(detector)
    pcx_slider = get_pcx_slider(detector)
    pcz_slider = get_pcz_slider(detector)
    sliders = [
        azimuthal_slider,
        pcx_slider,
        pcz_slider,
    ]

    fig, ax = set_up_figure_axis(ax=ax, **kwargs)
    ax.set_title("Top view")

    def redraw(*args: Any) -> None:
        update_detector_sample_geometry_top_view(
            detector, ax, legend=legend, dimensionless=dimensionless
        )
        fig.canvas.draw_idle()

    def update_detector_from_sliders():
        detector.azimuthal = azimuthal_slider.value
        detector.pcx = pcx_slider.value
        detector.pcz = pcz_slider.value

    if detector._has_signals:
        detector._azimuthal_changed.connect(redraw)
        detector._pc_changed.connect(redraw)

        def on_slider_change(change: Any = None) -> None:
            # Block to redraw only once
            with (
                detector._azimuthal_changed.blocked(),
                detector._pc_changed.blocked(),
            ):
                update_detector_from_sliders()
            redraw()
    else:

        def on_slider_change(change: Any = None) -> None:
            update_detector_from_sliders()
            redraw()

    for slider in sliders:
        slider.observe(on_slider_change, names="value")

    redraw()  # Initial draw

    controls = ipywidgets.VBox(sliders)

    return controls, fig


class EBSDDetectorOverlay(abc.ABC):
    """Abstract base class for overlays drawn on the EBSD detector.

    Subclasses supply a column of controls (via :meth:`setup`) and know
    how to paint themselves onto a Matplotlib axis (via :meth:`draw`).
    """

    @property
    @abc.abstractmethod
    def column_header(self) -> str:
        """Header text shown above this overlay's control column."""

    @abc.abstractmethod
    def setup(self, detector: EBSDDetector) -> list[ipywidgets.Widget]:
        """Set up overlay state depending on *detector* and return its
        widgets.

        Called once by :meth:`EBSDDetectorPlot.show` before the figure
        is displayed. The returned widgets are placed in the controls
        box under :attr:`column_header`.

        Parameters
        ----------
        detector
            The detector with a single PC that will be used for all
            subsequent :meth:`draw` calls.

        Returns
        -------
        widgets
            Ordered list of widgets to display in the control column.
        """

    @abc.abstractmethod
    def draw(
        self,
        ax: maxes.Axes,
        detector: EBSDDetector,
        rotation: oqu.Rotation,
        coords_fmt: DETECTOR_PLOT_FORMATS,
    ) -> None:
        """Draw the overlay on *ax* for the current *detector* and
        *rotation*.

        Called every time a slider changes value. The axes have already
        been cleared and the detector background redrawn before this
        method is invoked.

        Parameters
        ----------
        ax
            Detector panel axis.
        detector
            Current detector state.
        rotation
            Current crystal orientation (single rotation).
        coords_fmt
            Coordinate system in use: "gnomonic" or "detector".
        """


class GeometricalSimulationOverlay(EBSDDetectorOverlay):
    """Overlay that draws geometrical Kikuchi bands and zone axes.

    Parameters
    ----------
    reflectors
        Reciprocal lattice vectors defining the Kikuchi bands.
    bands_kwargs
        Keyword arguments forwarded to
        :class:`~matplotlib.collections.LineCollection` for the band
        traces.
    zone_axes_kwargs
        Keyword arguments forwarded to
        :meth:`~matplotlib.axes.Axes.scatter` for the zone axis markers.
    """

    def __init__(
        self,
        reflectors: "ReciprocalLatticeVector",
        bands_kwargs: dict | None = None,
        zone_axes_kwargs: dict | None = None,
    ) -> None:
        self._reflectors = reflectors

        if bands_kwargs is None:
            bands_kwargs = {}
        default_bands_kwds = {"color": "r", "linewidth": 1, "zorder": 2}
        for k, v in default_bands_kwds.items():
            bands_kwargs.setdefault(k, v)
        self._bands_kwargs = bands_kwargs

        if zone_axes_kwargs is None:
            zone_axes_kwargs = {}
        default_za_kwds = {"fc": "w", "ec": "k", "zorder": 3}
        for k, v in default_za_kwds.items():
            zone_axes_kwargs.setdefault(k, v)
        self._zone_axes_kwargs = zone_axes_kwargs

        self._uvw = _get_zone_axes_from_hkl(self._reflectors.hkl.astype(float))

        self._show_checkbox = ipywidgets.Checkbox(
            value=True, description="Geometric", indent=False
        )

    @property
    def column_header(self) -> str:
        return "Simulation"

    def setup(self, detector: EBSDDetector) -> list[ipywidgets.Widget]:
        self._zone_axes_kwargs["s"] = max(np.pi * (0.01 * detector.nrows) ** 2, 6)
        return [self._show_checkbox]

    def draw(
        self,
        ax: maxes.Axes,
        detector: EBSDDetector,
        rotation: oqu.Rotation,
        coords_fmt: DETECTOR_PLOT_FORMATS,
    ) -> None:
        if not self._show_checkbox.value:
            return
        band_coords, za_coords = _get_geometrical_simulation(
            reflectors=self._reflectors,
            detector=detector,
            rotation=rotation,
            uvw=self._uvw,
            coords_fmt=coords_fmt,
        )
        if band_coords.shape[0] > 0:
            segments = list(band_coords.reshape(-1, 2, 2))
            lines = mcollections.LineCollection(segments, **self._bands_kwargs)
            ax.add_collection(lines)
        if za_coords.shape[0] > 0:
            ax.scatter(za_coords[:, 0], za_coords[:, 1], **self._zone_axes_kwargs)


class MasterPatternOverlay(EBSDDetectorOverlay):
    """Overlay that projects a master pattern onto the detector.

    Parameters
    ----------
    master_pattern
        EBSD master pattern signal in the square Lambert projection.
    energy
        Acceleration voltage in kV used to select the master pattern
        slice.  If *None*, the highest energy slice is used.
    **kwargs
        Keyword arguments forwarded to
        :meth:`~matplotlib.axes.Axes.imshow`.
    """

    def __init__(
        self,
        master_pattern: EBSDMasterPattern,
        energy: float | None = None,
        **kwargs: Any,
    ) -> None:
        master_pattern._is_suitable_for_projection(True)
        self._master_pattern = master_pattern
        self._energy = energy
        self._kwargs = kwargs
        npx, npy = self._master_pattern.axes_manager.signal_shape
        self._npx = int(npx)
        self._npy = int(npy)
        self._scale = float((npx - 1) / 2)
        self._master_upper, self._master_lower = (
            self._master_pattern._get_master_pattern_arrays_from_energy(self._energy)
        )

        self._show_checkbox = ipywidgets.Checkbox(
            value=True, description="Simulation", indent=False
        )

        kwargs.setdefault("cmap", "gray")
        self._imshow_kwargs = kwargs

    @property
    def column_header(self) -> str:
        return "Simulation"

    def setup(self, detector: EBSDDetector) -> list[ipywidgets.Widget]:
        return [self._show_checkbox]

    def draw(
        self,
        ax: maxes.Axes,
        detector: EBSDDetector,
        rotation: oqu.Rotation,
        coords_fmt: DETECTOR_PLOT_FORMATS,
    ) -> None:
        if not self._show_checkbox.value:
            return

        dc = _get_direction_cosines_from_detector(detector)
        pattern_1d = _project_single_pattern_from_master_pattern(
            rotation=rotation.data.squeeze(),
            direction_cosines=dc,
            master_upper=self._master_upper,
            master_lower=self._master_lower,
            npx=self._npx,
            npy=self._npy,
            scale=self._scale,
            rescale=True,
            out_min=0.0,
            out_max=1.0,
            dtype_out=np.float32,
        )
        pattern = pattern_1d.reshape(detector.shape)

        extent = (
            detector._average_gnomonic_bounds
            if coords_fmt == "gnomonic"
            else detector.bounds
        )
        ax.imshow(pattern, extent=extent, **self._imshow_kwargs)


class EBSDDetectorPlotter:
    """Interactive EBSD detector plot with optional overlays.

    Shows the detector-sample geometry (side view, top view) and an EBSD
    detector panel with interactive slider controls. Overlays can be
    added via :meth:`add_geometrical_simulation` and
    :meth:`add_master_pattern` before calling :meth:`show`.

    .. warning::

        This plotter is highly experimental, and breaking API changes
        can be made between releases without prior notice.

        The plotter is considered as stable as the rest of the code base
        once this warning is removed.

    Parameters
    ----------
    detector
        EBSD detector to visualize.
    rotation
        Initial crystal orientation as a single
        :class:`~orix.quaternion.Rotation`. Required when any overlay is
        added.
    inplace
        Whether interactive changes affect *detector* inplace. Default
        is False (a deep copy is used). If True, the projection center
        will be overwritten with the average projection center (required
        for the plot).
    legend
        Whether to show a legend in the side and top views. Default is
        False.
    dimensionless
        Whether to ignore
        :attr:`~kikuchipy.detectors.EBSDDetector.px_size` when drawing
        the side-view plot axes. Default is True.
    coords_fmt
        Detector panel coordinate format: "gnomonic" (default) or
        "detector".
    """

    def __init__(
        self,
        detector: EBSDDetector,
        rotation: oqu.Rotation | None = None,
        *,
        inplace: bool = False,
        legend: bool = False,
        dimensionless: bool = True,
        coords_fmt: DETECTOR_PLOT_FORMATS = "gnomonic",
    ) -> None:
        if not inplace:
            detector = detector.deepcopy()
        self._detector = detector
        self._detector.pc = self._detector.pc_average
        self._rotation = rotation
        self._legend = legend
        self._dimensionless = dimensionless
        self._coords_fmt: DETECTOR_PLOT_FORMATS = coords_fmt
        self._overlays: list[EBSDDetectorOverlay] = []

    # --------------------------- Methods ---------------------------- #

    def set_geometrical_simulation(
        self,
        reflectors: "ReciprocalLatticeVector",
        bands_kwargs: dict | None = None,
        zone_axes_kwargs: dict | None = None,
    ) -> None:
        """Set the geometrical Kikuchi band simulation overlay.

        Valid if:

        - A rotation was given upon creation of the plotter
        - Plotter has no other geometrical simulation

        Parameters
        ----------
        reflectors
            Reciprocal lattice vectors defining the Kikuchi bands.
        bands_kwargs
            Keyword arguments for
            :class:`~matplotlib.collections.LineCollection`.
        zone_axes_kwargs
            Keyword arguments for
            :meth:`~matplotlib.axes.Axes.scatter`.
        """
        self._raise_if_no_rotation()
        self._overlays.append(
            GeometricalSimulationOverlay(reflectors, bands_kwargs, zone_axes_kwargs)
        )

    def set_master_pattern(
        self,
        master_pattern: "EBSDMasterPattern",
        energy: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Set the master pattern from which a simulated pattern is
        projected onto the detector.

        Valid if:

        - A rotation was given upon creation of the plotter
        - Plotter has no other master pattern added

        Parameters
        ----------
        master_pattern
            EBSD master pattern in the square Lambert projection.
        energy
            Acceleration voltage in kV.  If *None*, the highest energy
            slice is used.
        **kwargs
            Keyword arguments forwarded to
            :meth:`~matplotlib.axes.Axes.imshow`.
        """
        self._raise_if_no_rotation()
        self._overlays.append(MasterPatternOverlay(master_pattern, energy, **kwargs))

    def show(
        self, **figure_kwargs: Any
    ) -> "tuple[mfigure.Figure, ipywidgets.VBox | ipywidgets.HBox]":
        """Build the interactive figure and return it with its controls.

        Parameters
        ----------
        **figure_kwargs
            Keyword arguments passed to
            :func:`~matplotlib.pyplot.figure`.

        Returns
        -------
        fig
            Matplotlib figure containing the three-panel plot.
        controls
            Widget box containing the slider controls. Display with
            ``IPython.display.display(controls)`` alongside *fig* in a
            Jupyter notebook.

        Notes
        -----
        Requires :mod:`ipywidgets`. If :mod:`psygnal` is installed and
        the detector was not deep-copied (*inplace=True*), the side and
        top views are also driven by signals emitted from the detector
        properties.
        """
        det = self._detector

        # Set up overlay controls (call setup once, store widgets)
        pc_checkbox = ipywidgets.Checkbox(value=True, description="PC", indent=False)
        gnomonic_circles_checkbox = ipywidgets.Checkbox(
            value=self._coords_fmt == "gnomonic",
            description="Gnomonic circles",
            indent=False,
        )
        overlay_widget_lists: list[list] = []
        all_overlay_widgets: list[ipywidgets.Widget] = [
            pc_checkbox,
            gnomonic_circles_checkbox,
        ]
        for overlay in self._overlays:
            widgets = overlay.setup(det)
            overlay_widget_lists.append(widgets)
            all_overlay_widgets.extend(widgets)

        # Rotation sliders (only when overlays exist and rotation given)
        if self._overlays and self._rotation is not None:
            euler0 = self._rotation.to_euler(degrees=True).squeeze()
            phi1_slider = get_phi1_slider(float(euler0[0]))
            Phi_slider = get_Phi_slider(float(euler0[1]))
            phi2_slider = get_phi2_slider(float(euler0[2]))
            rotation_sliders = [phi1_slider, Phi_slider, phi2_slider]
        else:
            rotation_sliders = []

        # Detector sliders
        sample_tilt_slider = get_sample_tilt_slider(det)
        detector_tilt_slider = get_detector_tilt_slider(det)
        azimuthal_slider = get_detector_azimuthal_slider(det)
        twist_slider = get_detector_twist_slider(det)
        pcx_slider = get_pcx_slider(det)
        pcy_slider = get_pcy_slider(det)
        pcz_slider = get_pcz_slider(det)
        det_sliders = [
            sample_tilt_slider,
            detector_tilt_slider,
            azimuthal_slider,
            twist_slider,
            pcx_slider,
            pcy_slider,
            pcz_slider,
        ]
        all_sliders = det_sliders + rotation_sliders

        # Create figure
        w, h = plt.rcParams["figure.figsize"]
        figure_kwargs.setdefault("layout", "constrained")
        figure_kwargs.setdefault("figsize", (3 * w, h))
        fig = plt.figure(**figure_kwargs)
        ax_side, ax_top, ax_det = fig.subplots(1, 3)

        def get_rotation() -> oqu.Rotation | None:
            if rotation_sliders:
                eu = [phi1_slider.value, Phi_slider.value, phi2_slider.value]
                return oqu.Rotation.from_euler(eu, degrees=True)
            return self._rotation

        def redraw_side(*args: Any) -> None:
            update_detector_sample_geometry_side_view(
                det, ax_side, legend=self._legend, dimensionless=self._dimensionless
            )
            ax_side.set_title("Side view")

        def redraw_top(*args: Any) -> None:
            update_detector_sample_geometry_top_view(
                det, ax_top, legend=self._legend, dimensionless=self._dimensionless
            )
            ax_top.set_title("Top view")

        def redraw_det(*args: Any) -> None:
            update_detector_plane(
                det,
                ax_det,
                coords_fmt=self._coords_fmt,
                zoom=1,
                draw_gnomonic_circles=gnomonic_circles_checkbox.value,
                show_pc=pc_checkbox.value,
            )
            ax_det.set_title("Detector")
            rot = get_rotation()
            if rot is not None:
                for overlay in self._overlays:
                    overlay.draw(ax_det, det, rot, self._coords_fmt)

        def redraw_all() -> None:
            redraw_side()
            redraw_top()
            redraw_det()
            fig.canvas.draw_idle()

        def update_detector_from_sliders() -> None:
            det.sample_tilt = sample_tilt_slider.value
            det.tilt = detector_tilt_slider.value
            det.azimuthal = azimuthal_slider.value
            det.twist = twist_slider.value
            det.pc = [pcx_slider.value, pcy_slider.value, pcz_slider.value]

        if det._has_signals:
            # Affecting side view
            det._sample_tilt_changed.connect(redraw_side)
            det._tilt_changed.connect(redraw_side)
            det._pc_changed.connect(redraw_side)

            # Affecting top view
            det._sample_tilt_changed.connect(redraw_top)
            det._tilt_changed.connect(redraw_top)
            det._azimuthal_changed.connect(redraw_top)
            det._pc_changed.connect(redraw_top)

            # Affecting detector panel
            det._pc_changed.connect(redraw_det)
            det._twist_changed.connect(redraw_det)

            # Canvas flush
            det._sample_tilt_changed.connect(lambda *_: fig.canvas.draw_idle())
            det._tilt_changed.connect(lambda *_: fig.canvas.draw_idle())
            det._azimuthal_changed.connect(lambda *_: fig.canvas.draw_idle())
            det._twist_changed.connect(lambda *_: fig.canvas.draw_idle())
            det._pc_changed.connect(lambda *_: fig.canvas.draw_idle())

            def on_slider_change(change: Any = None) -> None:
                with (
                    det._sample_tilt_changed.blocked(),
                    det._tilt_changed.blocked(),
                    det._azimuthal_changed.blocked(),
                    det._twist_changed.blocked(),
                    det._pc_changed.blocked(),
                ):
                    update_detector_from_sliders()
                redraw_all()
        else:

            def on_slider_change(change: Any = None) -> None:
                update_detector_from_sliders()
                redraw_all()

        # Wire sliders and overlay widgets
        for slider in all_sliders:
            slider.observe(on_slider_change, names="value")
        for widget in all_overlay_widgets:
            widget.observe(on_slider_change, names="value")

        redraw_all()  # Initial draw

        # Build controls layout
        det_col = ipywidgets.VBox(
            [ipywidgets.HTML("<b>Detector-sample geometry</b>")] + det_sliders
        )
        cols: list[ipywidgets.VBox | ipywidgets.HBox] = [det_col]
        if rotation_sliders:
            rot_col = ipywidgets.VBox(
                [ipywidgets.HTML("<b>Crystal orientation</b>")] + rotation_sliders
            )
            cols.append(rot_col)
        if all_overlay_widgets:
            overlay_col = ipywidgets.VBox(
                [ipywidgets.HTML("<b>Overlay</b>")] + all_overlay_widgets
            )
            cols.append(overlay_col)
        controls = ipywidgets.HBox(cols) if len(cols) > 1 else det_col

        return fig, controls

    def _raise_if_no_rotation(self) -> None:
        if self._rotation is None:
            raise RuntimeError("Plotter must be made with a rotation")

    def _raise_if_has_overlay(
        self, overlay: Literal["geometrical", "master_pattern"]
    ) -> None:
        overlay_types = list(map(type, self._overlays))
        if overlay == "geometrical" and GeometricalSimulationOverlay in overlay_types:
            raise ValueError("Plotter already has a geometrical simulation")
        if overlay == "master_pattern" and MasterPatternOverlay in overlay_types:
            raise ValueError("Plotter already has a matter pattern")
