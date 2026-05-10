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

"""Plotting an EBSD detector and projection centers.

The interactive widgets require an optional dependency
:mod:`ipywidgets`. It must only be imported if we know it's installed.
"""

from typing import TYPE_CHECKING, Any, Literal

import matplotlib.axes as maxes
import matplotlib.figure as mfigure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from kikuchipy._constants import dependency_version, verify_dependency_or_raise
from kikuchipy.detectors._ebsd_detector import EBSDDetector

if TYPE_CHECKING:
    if dependency_version["ipywidgets"] is not None:
        import ipywidgets

# Repeated in detector module: keep in sync!
DETECTOR_PLOT_FORMATS = Literal["detector", "gnomonic"]
PROJECTION_CENTER_PLOT_MODES = Literal["map", "scatter", "3d"]

BEAM_COLOR = "C2"
SAMPLE_COLOR = "C0"
PC_COLOR = "C1"
DETECTOR_COLOR = "C7"


def plot_ebsd_detector(
    detector: EBSDDetector,
    coords_fmt: DETECTOR_PLOT_FORMATS,
    zoom: float,
    show_pc: bool,
    draw_gnomonic_circles: bool,
    pattern: np.ndarray | None,
    pattern_kwargs: dict,
    pc_kwargs: dict,
    gnomonic_circles_kwargs: dict,
    gnomonic_angles: np.ndarray | list | None,
    ax: maxes.Axes | None = None,
) -> mfigure.Figure | mfigure.SubFigure:
    sy, sx = detector.shape
    pcx, pcy = detector.pc_average[:2]
    bounds = detector.bounds

    if coords_fmt == "detector":
        pcy *= sy - 1
        pcx *= sx - 1
        bounds[2:] = bounds[2:][::-1]
        x_label = "x detector"
        y_label = "y detector"
    else:
        pcy, pcx = (0, 0)
        bounds = detector._average_gnomonic_bounds
        x_label = "x gnomonic"
        y_label = "y gnomonic"

    fig, axis = set_up_figure_axis(ax=ax)
    axis.axis(zoom * bounds)
    axis.set(xlabel=x_label, ylabel=y_label, aspect="equal")

    # Plot a pattern on the detector
    if isinstance(pattern, np.ndarray):
        if pattern.shape != (sy, sx):
            raise ValueError(
                f"Pattern shape {pattern.shape} must equal the detector shape "
                f"{(sy, sx)}"
            )
        pattern_kwargs.setdefault("cmap", "gray")
        axis.imshow(pattern, extent=bounds, **pattern_kwargs)
    else:
        origin = (bounds[0], bounds[2])
        width = np.diff(bounds[:2])[0]
        height = np.diff(bounds[2:])[0]
        axis.add_artist(
            mpatches.Rectangle(origin, width, height, fc=DETECTOR_COLOR, zorder=-1)
        )

    if show_pc:
        default_params_pc = {
            "s": 300,
            "facecolor": PC_COLOR,
            "marker": "+",
            "zorder": 10,
        }
        for k, v in default_params_pc.items():
            pc_kwargs.setdefault(k, v)
        axis.scatter(x=pcx, y=pcy, **pc_kwargs)

    # Draw gnomonic circles centered on the projection center
    if draw_gnomonic_circles:
        if gnomonic_circles_kwargs is None:
            gnomonic_circles_kwargs = {}
        default_params_gnomonic = {
            "alpha": 0.4,
            "edgecolor": "k",
            "facecolor": "None",
            "linewidth": 3,
        }
        for k, v in default_params_gnomonic.items():
            gnomonic_circles_kwargs.setdefault(k, v)
        if gnomonic_angles is None:
            gnomonic_angles = np.arange(1, 9) * 10
        for angle in gnomonic_angles:
            axis.add_patch(
                mpatches.Circle(
                    (pcx, pcy), np.tan(np.deg2rad(angle)), **gnomonic_circles_kwargs
                )
            )

    return fig


def plot_xtilt_estimate(
    pcy: np.ndarray,
    pcz: np.ndarray,
    x_tilt: float,
    pcy_fit: np.ndarray,
    pcz_fit: np.ndarray,
    is_outlier: np.ndarray | None,
    **kwargs,
) -> mfigure.Figure:
    """Plot projection centers and the estimated line.

    See :meth:`kikuchipy.detectors.EBSDDetector.estimate_xtilt` for
    details.
    """
    kwargs.setdefault("layout", "tight")
    fig = plt.figure(**kwargs)

    ax = fig.add_subplot()
    ax.scatter(pcz, pcy, c="yellowgreen", ec="k", label="Data")

    if is_outlier is not None:
        ax.scatter(pcz[is_outlier], pcy[is_outlier], c="gold", label="Outliers")

    x_tilt_deg = np.rad2deg(x_tilt)

    fit_label = "Fit, tilt = " + f"{x_tilt_deg:.2f}" + r"$^{\circ}$"
    ax.plot(pcz_fit, pcy_fit, label=fit_label, c="C1")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    ax.set(aspect="equal", xlabel="PCz", ylabel="PCy")

    return fig


def plot_projection_center_fit(
    pc: np.ndarray,
    pc_fit: np.ndarray,
    fit_intercept: float,
    fit_slope: float,
    **kwargs,
) -> mfigure.Figure:
    pcx, pcy, pcz = pc.T
    pcx_fit_2d, pcy_fit_2d, pcz_fit_2d = pc_fit.T
    pcx_fit = pcx_fit_2d.ravel()
    pcy_fit = pcy_fit_2d.ravel()
    pcz_fit = pcz_fit_2d.ravel()

    pcy_fit_line = fit_intercept + fit_slope * pcz_fit

    data_kw = {"s": 25, "c": "k"}
    fit_kw = {"s": 50, "fc": "gray", "alpha": 0.5, "ec": "k"}

    w, h = plt.rcParams["figure.figsize"]
    kwargs.setdefault("layout", "compressed")
    kwargs.setdefault("figsize", (w, h))

    fig = plt.figure(**kwargs)

    # PCx v PCy
    ax0 = fig.add_subplot(221)
    ax0.set(xlabel="PCx", ylabel="PCy", aspect="equal")
    ax0.scatter(pcx_fit, pcy_fit, **fit_kw)
    ax0.scatter(pcx, pcy, **data_kw)
    ax0.invert_xaxis()

    # PCx v PCz
    ax1 = fig.add_subplot(222)
    ax1.set(xlabel="PCx", ylabel="PCz", aspect="equal")
    ax1.scatter(pcx, pcz, **data_kw)
    ax1.scatter(pcx_fit, pcz_fit, **fit_kw)
    ax1.invert_xaxis()
    ax1.invert_yaxis()

    # PCz v PCy
    ax2 = fig.add_subplot(223)
    ax2.set(xlabel="PCz", ylabel="PCy", aspect="equal")
    ax2.scatter(pcz, pcy, label="Data", **data_kw)
    ax2.scatter(pcz_fit, pcy_fit, label="Fit", **fit_kw)
    ax2.plot(pcz_fit, pcy_fit_line, "r-", label="Linear fit")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

    # Plot PC values in 3D
    ax3: Axes3D = fig.add_subplot(224, projection="3d")
    ax3.scatter(pcx, pcz, pcy, **data_kw)
    ax3.scatter(pcx_fit, pcz_fit, pcy_fit, **fit_kw)
    ax3.set(
        xlabel="PCx",
        ylabel="PCz",
        zlabel="PCy",
        xlim=[np.min([pcx, pcx_fit]), np.max([pcx, pcx_fit])],
        ylim=[np.min([pcz, pcz_fit]), np.max([pcz, pcz_fit])],
        zlim=[np.min([pcy, pcy_fit]), np.max([pcy, pcy_fit])],
    )
    ax3.invert_zaxis()

    # Add 3D plane
    if pcx_fit_2d.ndim == 2:
        ax3.plot_surface(pcx_fit_2d, pcz_fit_2d, pcy_fit_2d, color="r", alpha=0.5)

    return fig


def plot_all_projection_centers(
    pc: np.ndarray,
    navigation_size: int,
    mode: PROJECTION_CENTER_PLOT_MODES,
    orientation: Literal["horizontal", "vertical"],
    annotate: bool,
    figure_kwargs: dict[str, Any],
    **kwargs,
) -> mfigure.Figure:
    # Prepare keyword arguments common to at least two modes
    figure_kwargs.setdefault("layout", "tight")
    if mode in ["map", "scatter"]:
        w, h = plt.rcParams["figure.figsize"]
        k = max(w, h) / 3
        if orientation == "horizontal":
            figure_kwargs.setdefault("figsize", (6 * k, 2 * k))
            subplots_kw = {"ncols": 3}
        else:
            figure_kwargs.setdefault("figsize", (2 * k, 6 * k))
            subplots_kw = {"nrows": 3}
    if mode in ["scatter", "3d"]:
        kwargs.setdefault("c", np.arange(navigation_size))
        kwargs.setdefault("ec", "k")
        kwargs.setdefault("clip_on", False)

    labels = ["PCx", "PCy", "PCz"]

    fig = plt.figure(**figure_kwargs)
    if mode == "map":
        plot_all_projection_centers_in_map(
            pc=pc, labels=labels, fig=fig, subplots_kw=subplots_kw, **kwargs
        )
    elif mode == "scatter":
        plot_all_projection_centers_as_scatter(
            pc=pc,
            labels=labels,
            fig=fig,
            subplots_kw=subplots_kw,
            annotate=annotate,
            **kwargs,
        )
    else:
        plot_all_projection_centers_in_3d(
            pc=pc, labels=labels, fig=fig, annotate=annotate, **kwargs
        )

    return fig


def plot_all_projection_centers_in_map(
    pc: np.ndarray,
    labels: list[str],
    fig: mfigure.Figure,
    subplots_kw: dict[str, Any],
    **kwargs,
) -> None:
    axes = fig.subplots(**subplots_kw)
    for i, ax in enumerate(axes):
        ax.set(xlabel="Column", ylabel="Row", aspect="equal")
        im = ax.imshow(pc[..., i], **kwargs)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position="right", size="5%", pad=0.1)
        fig.colorbar(im, cax=cax, label=labels[i])


def plot_all_projection_centers_as_scatter(
    pc: np.ndarray,
    labels: list[str],
    fig: mfigure.Figure,
    subplots_kw: dict,
    annotate: bool,
    **kwargs,
) -> None:
    pc_flattened = pc.reshape(-1, 3)
    axes = fig.subplots(**subplots_kw)
    for i, (j, k) in enumerate([[0, 1], [0, 2], [2, 1]]):
        x_coord = pc_flattened[:, j]
        y_coord = pc_flattened[:, k]
        axes[i].scatter(x_coord, y_coord, **kwargs)
        axes[i].set(xlabel=labels[j], ylabel=labels[k], aspect="equal")
        if annotate:
            for num, (x, y) in enumerate(zip(x_coord, y_coord)):
                axes[i].text(x, y, num, ha="left", va="bottom")
    axes[0].invert_xaxis()
    axes[1].invert_xaxis()
    axes[1].invert_yaxis()


def plot_all_projection_centers_in_3d(
    pc: np.ndarray,
    labels: list[str],
    fig: mfigure.Figure,
    annotate: bool,
    **kwargs,
) -> None:
    nav_shape = pc.shape[: pc.ndim - 1]
    nav_dim = len(nav_shape)
    pcx, pcy, pcz = pc.reshape(-1, 3).T

    ax: Axes3D = fig.add_subplot(projection="3d")
    ax.scatter(pcx, pcz, pcy, **kwargs)
    nav_axes = tuple(np.arange(len(pc.shape))[:nav_dim])
    extent_min = np.min(pc, axis=nav_axes)
    extent_max = np.max(pc, axis=nav_axes)
    ax.set(
        xlabel=labels[0],
        ylabel=labels[2],
        zlabel=labels[1],
        xlim=[extent_min[0], extent_max[0]],
        ylim=[extent_min[2], extent_max[2]],
        zlim=[extent_min[1], extent_max[1]],
    )
    ax.invert_zaxis()

    if annotate:
        for i, (x, z, y) in enumerate(zip(pcx, pcz, pcy)):
            ax.text(x, z, y, i)


def plot_detector_sample_geometry_side_view(
    detector: EBSDDetector,
    ax: maxes.Axes | None = None,
    annotate: bool = False,
    dimensionless: bool = False,
    **kwargs,
) -> mfigure.Figure | mfigure.SubFigure:
    """See the docstring of the EBSD detector method using this
    function.

    Coordinates are calculated in microns relative to the beam-sample
    interaction volume (0, 0).

    The figure plane is given by the microscope reference frame: Y west
    and Z north. Coordinates for elements are transformed to this
    reference frame after they are defined.
    """
    fig, ax = set_up_figure_axis(ax=ax, **kwargs)

    # Dimensions in mm and angles in radians
    height = detector.height
    L = detector.specimen_scintillator_distance[0]
    beam_length = height * 0.5
    half_sample_length = height * 0.3
    sigma = np.deg2rad(detector.sample_tilt)
    theta = np.deg2rad(detector.tilt)

    # Beam
    beam_end = np.zeros(2, dtype=np.float64)
    beam_start = np.array([0, -beam_length], dtype=np.float64)

    # Sample
    dx_sample = half_sample_length * np.cos(sigma)
    dy_sample = half_sample_length * np.sin(sigma)
    sample_end = np.array([dx_sample, dy_sample], dtype=np.float64)
    sample_start = -sample_end

    # Detector screen
    detector_z = np.array([np.cos(theta), np.sin(theta)])
    P = L * detector_z
    detector_y = np.array([-np.sin(theta), np.cos(theta)])

    # Screen extent
    pcy = detector.pc_average[1]
    screen_top = P - pcy * height * detector_y
    screen_bottom = P + (1 - pcy) * detector.height * detector_y

    # Negate Z so the Z axis points upward (opposite beam direction)
    def trans(point: np.ndarray) -> np.ndarray:
        p = point * 1e-3
        p[1] = -p[1]
        return p

    beam_start = trans(beam_start)
    beam_end = trans(beam_end)
    sample_start = trans(sample_start)
    sample_end = trans(sample_end)
    detector_start = trans(screen_top)
    detector_end = trans(screen_bottom)
    pc_pos = trans(P)
    source_pos = trans(np.zeros(2, dtype=np.float64))

    ax.annotate(
        "",
        xy=beam_end,
        xytext=beam_start,
        arrowprops={"arrowstyle": "->", "lw": 2, "color": BEAM_COLOR},
        zorder=5,
    )
    ax.plot(
        [sample_start[0], sample_end[0]],
        [sample_start[1], sample_end[1]],
        c=SAMPLE_COLOR,
        lw=6,
        label="Sample",
        zorder=4,
    )
    ax.plot(
        [detector_start[0], detector_end[0]],
        [detector_start[1], detector_end[1]],
        c=DETECTOR_COLOR,
        lw=4,
        label="Detector",
        zorder=4,
    )
    ax.plot(
        pc_pos[0],
        pc_pos[1],
        marker="+",
        ms=15,
        mec=PC_COLOR,
        linestyle="None",
        label="PC",
        zorder=5,
    )
    ax.plot(
        [source_pos[0], pc_pos[0]],
        [source_pos[1], pc_pos[1]],
        linestyle=":",
        color=BEAM_COLOR,
        alpha=0.5,
        zorder=0,
    )

    # Handle axis orientation: Y axis as x-coordinate (inverted so
    # detector appears on the left), Z as y-coordinate pointing up
    ax.set_aspect("equal")

    # Fix axis limits
    pad = get_axis_limit_pad(height)
    ax.set_xlim(pad, -pad)
    ax.set_ylim(-pad, pad)

    if annotate:
        ax.text(
            beam_start[0],
            beam_start[1],
            "Beam",
            ha="center",
            va="bottom",
            label="beam_annotation",
        )
        ax.text(
            sample_start[0],
            sample_start[1],
            "Sample",
            ha="right",
            va="center",
            label="sample_annotation",
        )
        ax.annotate(
            "Detector",
            xy=(detector_start[0], detector_start[1]),
            xytext=(0, -12),
            textcoords="offset points",
            ha="center",
            va="top",
            label="detector_annotation",
        )
        ax.annotate(
            "PC",
            xy=(pc_pos[0], pc_pos[1]),
            xytext=(5, 5),
            textcoords="offset points",
            ha="left",
            va="bottom",
            label="pc_annotation",
        )

    if dimensionless:
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        unit = ""
    else:
        unit = " [mm]"
    ax.set_xlabel(f"Microscope Y{unit}")
    ax.set_ylabel(f"Microscope Z{unit}")

    return fig


def update_detector_sample_geometry_side_view(
    detector: EBSDDetector,
    ax: maxes.Axes,
    annotate: bool = False,
    dimensionless: bool = True,
) -> None:
    """Clear *ax* and redraw the side view for *detector*."""
    ax.clear()
    plot_detector_sample_geometry_side_view(
        detector, ax=ax, annotate=annotate, dimensionless=dimensionless
    )


def plot_detector_sample_geometry_top_view(
    detector: EBSDDetector,
    ax: maxes.Axes | None = None,
    annotate: bool = False,
    dimensionless: bool = False,
    **kwargs,
) -> mfigure.Figure | mfigure.SubFigure:
    """See the docstring of the EBSD detector method using this function
    for further details.

    Coordinates are calculated in microns relative to the beam-sample
    interaction volume (0, 0).

    The figure plane is given by the microscope reference frame: X west
    and Y south. Coordinates for elements are transformed to this
    reference frame after they are defined.
    """
    fig, ax = set_up_figure_axis(ax=ax, **kwargs)

    # Dimensions in microns and angles in radians
    width = detector.width
    L = detector.specimen_scintillator_distance[0]
    sample_length = width * 0.6
    azimuthal = np.deg2rad(detector.azimuthal)

    beam = np.zeros(2)

    # PC in microscope X-Y coordinates, with the screen normal pointing
    # towards the interaction volume
    det_normal = np.array([np.sin(azimuthal), np.cos(azimuthal)])
    pc_pos = L * det_normal

    # Unit vector along the detector width, perpendicular to the normal
    width_dir = np.array([-np.cos(azimuthal), np.sin(azimuthal)])

    # Detector endpoints
    pcx = detector.pc_average[0]
    left_extent = pcx * width
    right_extent = (1 - pcx) * width
    det_left = pc_pos - left_extent * width_dir
    det_right = pc_pos + right_extent * width_dir

    # Sample surface: a line along X at Y=0
    sample_start = np.array([-sample_length / 2, 0])
    sample_end = np.array([sample_length / 2, 0])

    to_mm = 1e-3

    ax.plot(
        beam[0] * to_mm,
        beam[1] * to_mm,
        marker="o",
        ms=8,
        color=BEAM_COLOR,
        zorder=5,
        label="Beam",
    )
    ax.plot(
        [sample_start[0] * to_mm, sample_end[0] * to_mm],
        [sample_start[1] * to_mm, sample_end[1] * to_mm],
        c=SAMPLE_COLOR,
        lw=6,
        label="Sample",
        zorder=4,
    )
    ax.plot(
        [det_left[0] * to_mm, det_right[0] * to_mm],
        [det_left[1] * to_mm, det_right[1] * to_mm],
        c=DETECTOR_COLOR,
        lw=4,
        label="Detector",
        zorder=4,
    )
    ax.plot(
        pc_pos[0] * to_mm,
        pc_pos[1] * to_mm,
        marker="+",
        ms=15,
        mec=PC_COLOR,
        linestyle="None",
        label="PC",
        zorder=5,
    )
    ax.plot(
        [beam[0] * to_mm, pc_pos[0] * to_mm],
        [beam[1] * to_mm, pc_pos[1] * to_mm],
        linestyle=":",
        color=BEAM_COLOR,
        alpha=0.5,
        zorder=0,
    )

    ax.set_aspect("equal")

    # Fix axis limits
    pad = get_axis_limit_pad(width)
    ax.set_xlim(pad, -pad)
    ax.set_ylim(pad, -pad)

    if annotate:
        ax.text(
            beam[0] * to_mm,
            beam[1] * to_mm,
            " Beam",
            ha="left",
            va="bottom",
            label="beam_annotation",
        )
        ax.text(
            sample_end[0] * to_mm,
            sample_end[1] * to_mm,
            " Sample",
            ha="left",
            va="center",
            label="sample_annotation",
        )
        ax.annotate(
            "Detector",
            xy=(pc_pos[0] * to_mm, pc_pos[1] * to_mm),
            xytext=(0, -12),
            textcoords="offset points",
            ha="center",
            va="top",
            label="detector_annotation",
        )
        ax.annotate(
            "PC",
            xy=(pc_pos[0] * to_mm, pc_pos[1] * to_mm),
            xytext=(5, 5),
            textcoords="offset points",
            ha="left",
            va="bottom",
            label="pc_annotation",
        )

    if dimensionless:
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        unit = ""
    else:
        unit = " [mm]"
    ax.set_xlabel(f"x microscope{unit}")
    ax.set_ylabel(f"y microscope{unit}")

    return fig


def get_axis_limit_pad(width: float) -> float:
    """Return sufficient axis limits for the detector to move around
    without moving outside the axis.
    """
    width_mm = width * 1e-3
    pad = 1.1 * width_mm
    return pad


def update_detector_sample_geometry_top_view(
    detector: EBSDDetector,
    ax: maxes.Axes,
    annotate: bool = False,
    dimensionless: bool = True,
) -> None:
    """Clear *ax* and redraw the top view for *detector*."""
    ax.clear()
    plot_detector_sample_geometry_top_view(
        detector, ax=ax, annotate=annotate, dimensionless=dimensionless
    )


def update_detector_plane(
    detector: EBSDDetector,
    ax: maxes.Axes,
    coords_fmt: DETECTOR_PLOT_FORMATS = "detector",
    zoom: float = 1.0,
) -> None:
    """Clear *ax* and redraw the detector plane for *detector*."""
    ax.clear()
    plot_ebsd_detector(
        detector,
        coords_fmt=coords_fmt,
        zoom=zoom,
        show_pc=True,
        draw_gnomonic_circles=False,
        pattern=None,
        pattern_kwargs={},
        pc_kwargs={},
        gnomonic_circles_kwargs={},
        gnomonic_angles=None,
        ax=ax,
    )


def plot_detector_sample_geometry_side_view_interactive(
    detector: EBSDDetector,
    ax: maxes.Axes | None = None,
    annotate: bool = False,
    dimensionless: bool = True,
    **kwargs,
) -> "tuple[ipywidgets.VBox, mfigure.Figure | mfigure.SubFigure]":
    """See the docstring of
    :meth:`~kikuchipy.detectors.EBSDDetector.plot_side_view` for
    details.
    """
    verify_dependency_or_raise("ipywidgets", "Interactive detector plots")
    verify_dependency_or_raise("IPython", "Interactive detector plots")

    import ipywidgets

    detector.pc = detector.pc_average
    sample_tilt_slider = get_sample_tilt_slider(detector)
    detector_tilt_slider = get_detector_tilt_slider(detector)
    azimuthal_slider = get_detector_azimuthal_slider(detector)
    pcy_slider = get_pcy_slider(detector)
    pcz_slider = get_pcz_slider(detector)
    sliders = [
        sample_tilt_slider,
        detector_tilt_slider,
        azimuthal_slider,
        pcy_slider,
        pcz_slider,
    ]

    fig, ax = set_up_figure_axis(ax=ax, **kwargs)
    ax.set_title("Side view")

    def redraw(*args: Any) -> None:
        update_detector_sample_geometry_side_view(
            detector, ax, annotate=annotate, dimensionless=dimensionless
        )
        fig.canvas.draw_idle()

    def update_detector_from_sliders():
        detector.sample_tilt = sample_tilt_slider.value
        detector.tilt = detector_tilt_slider.value
        detector.azimuthal = azimuthal_slider.value
        detector.pcy = pcy_slider.value
        detector.pcz = pcz_slider.value

    if detector._has_signals:
        detector._sample_tilt_changed.connect(redraw)
        detector._tilt_changed.connect(redraw)
        detector._azimuthal_changed.connect(redraw)
        detector._pc_changed.connect(redraw)

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
    annotate: bool = False,
    dimensionless: bool = True,
    **kwargs,
) -> "tuple[ipywidgets.VBox, mfigure.Figure | mfigure.SubFigure]":
    """See the docstring of
    :meth:`~kikuchipy.detectors.EBSDDetector.plot_top_view` for
    details.
    """
    verify_dependency_or_raise("ipywidgets", "Interactive detector plots")
    verify_dependency_or_raise("IPython", "Interactive detector plots")

    import ipywidgets

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
            detector, ax, annotate=annotate, dimensionless=dimensionless
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


def plot_detector_sample_geometry_interactive(
    detector: EBSDDetector,
    inplace: bool = False,
    annotate: bool = False,
    dimensionless: bool = True,
    coords_fmt: DETECTOR_PLOT_FORMATS = "detector",
    zoom: float = 1.0,
    **kwargs,
) -> "tuple[mfigure.Figure, ipywidgets.VBox]":
    """Plot the side view, top view, and detector plane side by side
    with interactive controls.

    Parameters
    ----------
    detector
        EBSD detector.
    inplace
        Whether to edit the given *detector* inplace. The given detector
        is not affected by default.
    annotate
        Whether to annotate the side-view components.
        Default is False.
    dimensionless
        Whether to ignore
        :attr:`~kikuchipy.detectors.EBSDDetector.px_size` when
        drawing the side-view plot axes. Default is True.
    coords_fmt
        Detector plane coordinate format: ``"detector"`` (default) or
        ``"gnomonic"``.
    zoom
        Zoom factor for the detector plane. Default is 1.0.
    **kwargs
        Keyword arguments passed to :func:`~matplotlib.pyplot.figure`.

    Returns
    -------
    fig
        Matplotlib figure.
    controls
        ipywidgets VBox containing slider controls.

    Notes
    -----
    Requires that :mod:`ipywidgets` is installed. If :mod:`psygnal` is
    installed, the plots are driven by signals emitted from the detector
    property setters.
    """
    verify_dependency_or_raise("ipywidgets", "Interactive detector plots")
    verify_dependency_or_raise("IPython", "Interactive detector plots")

    import ipywidgets

    if inplace:
        det = detector
    else:
        det = detector.deepcopy()
    det.pc = det.pc_average

    sample_tilt_slider = get_sample_tilt_slider(detector)
    detector_tilt_slider = get_detector_tilt_slider(detector)
    azimuthal_slider = get_detector_azimuthal_slider(detector)
    pcx_slider = get_pcx_slider(detector)
    pcy_slider = get_pcy_slider(detector)
    pcz_slider = get_pcz_slider(detector)
    sliders = [
        sample_tilt_slider,
        detector_tilt_slider,
        azimuthal_slider,
        pcx_slider,
        pcy_slider,
        pcz_slider,
    ]

    w, h = plt.rcParams["figure.figsize"]
    kwargs.setdefault("layout", "constrained")
    kwargs.setdefault("figsize", (3 * w, h))
    fig = plt.figure(**kwargs)
    ax_side, ax_top, ax_det = fig.subplots(1, 3)

    def redraw_side(*args: Any) -> None:
        update_detector_sample_geometry_side_view(
            det, ax_side, annotate=annotate, dimensionless=dimensionless
        )
        ax_side.set_title("Side view")

    def redraw_top(*args: Any) -> None:
        update_detector_sample_geometry_top_view(
            det, ax_top, annotate=annotate, dimensionless=dimensionless
        )
        ax_top.set_title("Top view")

    def redraw_det(*args: Any) -> None:
        update_detector_plane(det, ax_det, coords_fmt=coords_fmt, zoom=zoom)
        ax_det.set_title("Detector")

    def redraw_all() -> None:
        redraw_side()
        redraw_top()
        redraw_det()
        fig.canvas.draw_idle()

    def update_detector_from_sliders():
        det.sample_tilt = sample_tilt_slider.value
        det.tilt = detector_tilt_slider.value
        det.azimuthal = azimuthal_slider.value
        det.pc = [pcx_slider.value, pcy_slider.value, pcz_slider.value]

    if det._has_signals:
        det._sample_tilt_changed.connect(redraw_side)
        det._sample_tilt_changed.connect(redraw_top)
        det._sample_tilt_changed.connect(lambda *_: fig.canvas.draw_idle())
        det._tilt_changed.connect(redraw_side)
        det._tilt_changed.connect(redraw_top)
        det._tilt_changed.connect(lambda *_: fig.canvas.draw_idle())
        det._azimuthal_changed.connect(redraw_top)
        det._azimuthal_changed.connect(lambda *_: fig.canvas.draw_idle())
        det._pc_changed.connect(redraw_side)
        det._pc_changed.connect(redraw_top)
        det._pc_changed.connect(redraw_det)
        det._pc_changed.connect(lambda *_: fig.canvas.draw_idle())

        def on_slider_change(change: Any = None) -> None:
            with (
                det._sample_tilt_changed.blocked(),
                det._tilt_changed.blocked(),
                det._azimuthal_changed.blocked(),
                det._pc_changed.blocked(),
            ):
                update_detector_from_sliders()
            redraw_all()
    else:

        def on_slider_change(change: Any = None) -> None:
            update_detector_from_sliders()
            redraw_all()

    for slider in sliders:
        slider.observe(on_slider_change, names="value")

    redraw_all()  # Initial draw

    controls = ipywidgets.VBox(sliders)

    return fig, controls


def set_up_figure_axis(
    ax: maxes.Axes | None = None, **kwargs
) -> tuple[mfigure.Figure | mfigure.SubFigure, maxes.Axes]:
    if ax is None:
        w, h = plt.rcParams["figure.figsize"]
        kwargs.setdefault("layout", "constrained")
        kwargs.setdefault("figsize", (w, h))
        fig = plt.figure(**kwargs)
        ax = fig.add_subplot()
    else:
        fig = ax.figure
    return fig, ax


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


def get_sample_tilt_slider(detector: EBSDDetector) -> "ipywidgets.FloatSlider":
    """Return an :mod:`ipywidgets` slider from a *detector*."""
    verify_dependency_or_raise("ipywidgets", "Interactive sliders require")

    import ipywidgets

    stilt = detector.sample_tilt
    stilt_min, stilt_max = get_detector_value_range(stilt, 0, 180)

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


def get_detector_tilt_slider(detector: EBSDDetector) -> "ipywidgets.FloatSlider":
    """Return an :mod:`ipywidgets` slider from a *detector*."""
    verify_dependency_or_raise("ipywidgets", "Interactive sliders require")

    import ipywidgets

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


def get_detector_azimuthal_slider(detector: EBSDDetector) -> "ipywidgets.FloatSlider":
    """Return an :mod:`ipywidgets` slider from a *detector*."""
    verify_dependency_or_raise("ipywidgets", "Interactive sliders require")

    import ipywidgets

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


def get_pcx_slider(detector: EBSDDetector) -> "ipywidgets.FloatSlider":
    """Return an :mod:`ipywidgets` slider from a *detector*."""
    verify_dependency_or_raise("ipywidgets", "Interactive sliders require")

    import ipywidgets

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


def get_pcy_slider(detector: EBSDDetector) -> "ipywidgets.FloatSlider":
    """Return an :mod:`ipywidgets` slider from a *detector*."""
    verify_dependency_or_raise("ipywidgets", "Interactive sliders require")

    import ipywidgets

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


def get_pcz_slider(detector: EBSDDetector) -> "ipywidgets.FloatSlider":
    """Return an :mod:`ipywidgets` slider from a *detector*."""
    verify_dependency_or_raise("ipywidgets", "Interactive sliders require")

    import ipywidgets

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
