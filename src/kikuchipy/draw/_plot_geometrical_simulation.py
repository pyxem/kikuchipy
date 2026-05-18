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

"""Plotting of geometrical Kikuchi pattern simulations."""

from typing import TYPE_CHECKING, Any

from diffsims.crystallography import ReciprocalLatticeVector
import matplotlib.axes as maxes
import matplotlib.collections as mcollections
import matplotlib.figure as mfigure
import numpy as np
import orix.quaternion as oqu

from kikuchipy._constants import verify_dependency_or_raise
from kikuchipy.detectors._ebsd_detector import EBSDDetector

if TYPE_CHECKING:
    import ipywidgets


def _plot_geometrical_simulation_on_detector(
    band_coords: np.ndarray,
    zone_axes_coords: np.ndarray,
    detector: EBSDDetector,
    ax: maxes.Axes | None = None,
    bands_kwargs: dict | None = None,
    zone_axes_kwargs: dict | None = None,
) -> mfigure.Figure | mfigure.SubFigure:
    """Plot Kikuchi bands and zone axes on an EBSD detector.

    Parameters
    ----------
    band_coords
        Plane-trace endpoints (x0, y0, x1, y1) in uncalibrated
        detector-pixel coordinates, shape (k, 4).
    zone_axes_coords
        Zone axis (x, y) positions in uncalibrated detector-pixel
        coordinates, shape (l, 2).
    detector
        EBSD detector. Used to set up the detector background when *ax*
        is not given.
    ax
        Existing Matplotlib axes to add to. If not given, a new figure
        is created via
        :func:`~kikuchipy.draw._ebsd_detector_plot.plot_ebsd_detector`.
    bands_kwargs
        Keyword arguments passed to
        :class:`~matplotlib.collections.LineCollection` for the band
        traces.
    zone_axes_kwargs
        Keyword arguments passed to
        :meth:`~matplotlib.axes.Axes.scatter` for the zone axis markers.

    Returns
    -------
    fig
        Matplotlib figure containing the plot.
    """
    if ax is None:
        from kikuchipy.draw._ebsd_detector_plot import plot_ebsd_detector

        fig = plot_ebsd_detector(
            detector=detector,
            coords_fmt="detector",
            zoom=1.0,
            show_pc=True,
            draw_gnomonic_circles=False,
            pattern=None,
            pattern_kwargs={},
            pc_kwargs={},
            gnomonic_circles_kwargs={},
            gnomonic_angles=None,
        )
        ax = fig.axes[0]
    else:
        fig = ax.figure

    if band_coords.shape[0] > 0:
        segments = band_coords.reshape(-1, 2, 2)
        kw = {"color": "r", "linewidth": 1, "zorder": 2}
        if bands_kwargs is not None:
            kw.update(bands_kwargs)
        ax.add_collection(mcollections.LineCollection(segments, **kw))

    if zone_axes_coords.shape[0] > 0:
        # Scale marker area to ~1 % of detector height (matching the
        # existing GeometricalKikuchiPatternSimulation convention)
        r_px = 0.01 * detector.nrows
        s = np.pi * r_px**2  # area in data units² → reasonable default
        kw = {"fc": "w", "ec": "k", "zorder": 3, "s": max(s, 6)}
        if zone_axes_kwargs is not None:
            kw.update(zone_axes_kwargs)
        ax.scatter(zone_axes_coords[:, 0], zone_axes_coords[:, 1], **kw)

    return fig


def _plot_geometrical_simulation_on_detector_interactive(
    reflectors: ReciprocalLatticeVector,
    detector: EBSDDetector,
    rotation: oqu.Rotation,
    inplace: bool = False,
    bands_kwargs: dict | None = None,
    zone_axes_kwargs: dict | None = None,
    **kwargs: Any,
) -> "tuple[mfigure.Figure | mfigure.SubFigure, ipywidgets.VBox]":
    """Interactively plot Kikuchi bands and zone axes on an EBSD
    detector.

    Sliders control both the crystal orientation (ZXZ Bunge Euler angles)
    and all EBSD detector geometry parameters, so the effect of each on
    the geometrical simulation can be explored in real time.

    Parameters
    ----------
    reflectors
        Reciprocal lattice vectors defining the Kikuchi bands.
    detector
        EBSD detector. Averaged to a single PC before use. Modified
        inplace if *inplace* is True, otherwise a deep copy is used.
    rotation
        A single initial crystal orientation.
    inplace
        Whether to modify *detector* inplace. Default is False.
    bands_kwargs
        Keyword arguments passed to
        :class:`~matplotlib.collections.LineCollection` for the band
        traces.
    zone_axes_kwargs
        Keyword arguments passed to
        :meth:`~matplotlib.axes.Axes.scatter` for the zone axis markers.
    **kwargs
        Keyword arguments passed to :func:`~matplotlib.pyplot.figure`.

    Returns
    -------
    fig
        Matplotlib figure containing the detector plot.
    controls
        Slider controls. Display with
        ``IPython.display.display(controls)`` or alongside *fig* in a
        Jupyter notebook.

    Notes
    -----
    Requires :mod:`ipywidgets`. If :mod:`psygnal` is installed the plot
    is driven by signals emitted from the detector property setters.
    """
    verify_dependency_or_raise("ipywidgets", "Interactive detector plots")
    verify_dependency_or_raise("IPython", "Interactive detector plots")

    import ipywidgets

    from kikuchipy.draw._ebsd_detector_plot import (
        plot_ebsd_detector,
        set_up_figure_axis,
    )
    from kikuchipy.draw._ebsd_detector_plot_widgets import (
        get_detector_azimuthal_slider,
        get_detector_tilt_slider,
        get_pcx_slider,
        get_pcy_slider,
        get_pcz_slider,
        get_sample_tilt_slider,
        get_slider_style,
    )
    from kikuchipy.simulations._geometrical_simulation import (
        _get_geometrical_simulation,
        _get_zone_axes_from_hkl,
    )

    # Pre-compute zone axes once; they depend only on hkl indices which
    # never change during interactive exploration.
    uvw = _get_zone_axes_from_hkl(reflectors.hkl.astype(float))

    if inplace:
        det = detector
    else:
        det = detector.deepcopy()
    det.pc = det.pc_average

    # Crystal orientation sliders (ZXZ Bunge Euler angles, degrees)
    euler0 = rotation.to_euler(degrees=True).squeeze()
    style = get_slider_style()
    phi1_slider = ipywidgets.FloatSlider(
        value=float(euler0[0]),
        min=0.0,
        max=360.0,
        step=0.1,
        description="\u03c61 (\u00b0)",
        style=style,
    )
    Phi_slider = ipywidgets.FloatSlider(
        value=float(euler0[1]),
        min=0.0,
        max=180.0,
        step=0.1,
        description="\u03a6 (\u00b0)",
        style=style,
    )
    phi2_slider = ipywidgets.FloatSlider(
        value=float(euler0[2]),
        min=0.0,
        max=360.0,
        step=0.1,
        description="\u03c62 (\u00b0)",
        style=style,
    )

    # Detector parameter sliders
    sample_tilt_slider = get_sample_tilt_slider(det)
    detector_tilt_slider = get_detector_tilt_slider(det)
    azimuthal_slider = get_detector_azimuthal_slider(det)
    pcx_slider = get_pcx_slider(det)
    pcy_slider = get_pcy_slider(det)
    pcz_slider = get_pcz_slider(det)

    all_sliders = [
        phi1_slider,
        Phi_slider,
        phi2_slider,
        sample_tilt_slider,
        detector_tilt_slider,
        azimuthal_slider,
        pcx_slider,
        pcy_slider,
        pcz_slider,
    ]

    fig, ax = set_up_figure_axis(**kwargs)

    def redraw(*args: Any) -> None:
        rot = oqu.Rotation.from_euler(
            [phi1_slider.value, Phi_slider.value, phi2_slider.value],
            degrees=True,
        )
        band_coords, za_coords = _get_geometrical_simulation(
            reflectors, det, rot, uvw=uvw
        )
        ax.clear()
        plot_ebsd_detector(
            det,
            coords_fmt="detector",
            zoom=1.0,
            show_pc=True,
            draw_gnomonic_circles=False,
            pattern=None,
            pattern_kwargs={},
            pc_kwargs={},
            gnomonic_circles_kwargs={},
            gnomonic_angles=None,
            ax=ax,
        )
        if band_coords.shape[0] > 0:
            segments = band_coords.reshape(-1, 2, 2)
            kw: dict = {"color": "r", "linewidth": 1, "zorder": 2}
            if bands_kwargs is not None:
                kw.update(bands_kwargs)
            ax.add_collection(mcollections.LineCollection(segments, **kw))
        if za_coords.shape[0] > 0:
            r_px = 0.01 * det.nrows
            s = np.pi * r_px**2
            kw = {"fc": "w", "ec": "k", "zorder": 3, "s": max(s, 6)}
            if zone_axes_kwargs is not None:
                kw.update(zone_axes_kwargs)
            ax.scatter(za_coords[:, 0], za_coords[:, 1], **kw)
        fig.canvas.draw_idle()

    def update_detector_from_sliders() -> None:
        det.sample_tilt = sample_tilt_slider.value
        det.tilt = detector_tilt_slider.value
        det.azimuthal = azimuthal_slider.value
        det.pc = [pcx_slider.value, pcy_slider.value, pcz_slider.value]

    if det._has_signals:

        def on_slider_change(change: Any = None) -> None:
            with (
                det._sample_tilt_changed.blocked(),
                det._tilt_changed.blocked(),
                det._azimuthal_changed.blocked(),
                det._pc_changed.blocked(),
            ):
                update_detector_from_sliders()
            redraw()

    else:

        def on_slider_change(change: Any = None) -> None:
            update_detector_from_sliders()
            redraw()

    for slider in all_sliders:
        slider.observe(on_slider_change, names="value")

    redraw()  # Initial draw

    controls = ipywidgets.VBox(all_sliders)

    return fig, controls
