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

"""Plotting an EBSD detector and projection centers."""

from typing import TYPE_CHECKING, Any, Literal

import matplotlib.axes as maxes
import matplotlib.figure as mfigure
from matplotlib.markers import MarkerStyle
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from kikuchipy._constants import dependency_version
from kikuchipy.detectors._ebsd_detector import EBSDDetector

if TYPE_CHECKING:
    if dependency_version["ipywidgets"] is not None:
        import ipywidgets as widgets

DETECTOR_PLOT_FORMATS = Literal["detector", "gnomonic"]
PROJECTION_CENTER_PLOT_MODES = Literal["map", "scatter", "3d"]


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
) -> mfigure.Figure:
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

    fig, ax = plt.subplots()
    ax.axis(zoom * bounds)
    ax.set(xlabel=x_label, ylabel=y_label, aspect="equal")

    # Plot a pattern on the detector
    if isinstance(pattern, np.ndarray):
        if pattern.shape != (sy, sx):
            raise ValueError(
                f"Pattern shape {pattern.shape} must equal the detector shape "
                f"{(sy, sx)}"
            )
        pattern_kwargs.setdefault("cmap", "gray")
        ax.imshow(pattern, extent=bounds, **pattern_kwargs)
    else:
        origin = (bounds[0], bounds[2])
        width = np.diff(bounds[:2])[0]
        height = np.diff(bounds[2:])[0]
        ax.add_artist(
            mpatches.Rectangle(origin, width, height, fc=(0.5,) * 3, zorder=-1)
        )

    # Show the projection center
    if show_pc:
        default_params_pc = dict(
            s=300,
            facecolor="gold",
            edgecolor="k",
            marker=MarkerStyle(marker="*", fillstyle="full"),
            zorder=10,
        )
        for k, v in default_params_pc.items():
            pc_kwargs.setdefault(k, v)
        ax.scatter(x=pcx, y=pcy, **pc_kwargs)

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
            ax.add_patch(
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
        fig = plot_all_projection_centers_in_map(
            pc=pc, labels=labels, fig=fig, subplots_kw=subplots_kw, **kwargs
        )
    elif mode == "scatter":
        fig = plot_all_projection_centers_as_scatter(
            pc=pc,
            labels=labels,
            fig=fig,
            subplots_kw=subplots_kw,
            annotate=annotate,
            **kwargs,
        )
    else:
        fig = plot_all_projection_centers_in_3d(
            pc=pc, labels=labels, fig=fig, annotate=annotate, **kwargs
        )

    return fig


def plot_all_projection_centers_in_map(
    pc: np.ndarray,
    labels: list[str],
    fig: mfigure.Figure,
    subplots_kw: dict[str, Any],
    **kwargs,
) -> mfigure.Figure:
    axes = fig.subplots(**subplots_kw)
    for i, ax in enumerate(axes):
        ax.set(xlabel="Column", ylabel="Row", aspect="equal")
        im = ax.imshow(pc[..., i], **kwargs)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position="right", size="5%", pad=0.1)
        fig.colorbar(im, cax=cax, label=labels[i])
    return fig


def plot_all_projection_centers_as_scatter(
    pc: np.ndarray,
    labels: list[str],
    fig: mfigure.Figure,
    subplots_kw: dict,
    annotate: bool,
    **kwargs,
) -> mfigure.Figure:
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
    return fig


def plot_all_projection_centers_in_3d(
    pc: np.ndarray,
    labels: list[str],
    fig: mfigure.Figure,
    annotate: bool,
    **kwargs,
) -> mfigure.Figure:
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

    return fig


def plot_ebsd_detector_geometry_side_view(
    detector: EBSDDetector,
    ax: maxes.Axes | None = None,
    annotate: bool = False,
    dimensionless: bool = False,
    **kwargs,
) -> mfigure.Figure | mfigure.SubFigure:
    """See the docstring of the EBSD detector method using this
    function.
    """
    if ax is None:
        w, h = plt.rcParams["figure.figsize"]
        kwargs.setdefault("layout", "constrained")
        kwargs.setdefault("figsize", (w, h))
        fig = plt.figure(**kwargs)
        ax = fig.add_subplot()
    else:
        fig = ax.figure

    # Dimensions in microns
    height = detector.height
    L = detector.specimen_scintillator_distance[0]
    beam_length = L * 0.5
    sample_length = L * 1.5

    # Coordinates are calculated in microns relative to the beam-sample
    # interaction volume (0, 0)

    # Electron beam
    beam_start = np.array([0, -beam_length], dtype=np.float64)
    beam_end = np.zeros(2, dtype=np.float64)

    # Sample surface
    sigma = np.deg2rad(detector.sample_tilt)
    dx_sample = (sample_length / 2) * np.cos(sigma)
    dz_sample = (sample_length / 2) * np.sin(sigma)
    sample_start = np.array([-dx_sample, -dz_sample])
    sample_end = np.array([dx_sample, dz_sample])

    # Detector screen (Z = normal, Y = down)
    theta = np.deg2rad(detector.tilt)
    detector_z = np.array([np.cos(theta), np.sin(theta)])
    P = L * detector_z
    detector_y = np.array([-np.sin(theta), np.cos(theta)])

    # Screen extent
    pcy = detector.pc_average[1]
    screen_top = P - pcy * height * detector_y
    screen_bottom = P + (1 - pcy) * detector.height * detector_y

    # Shift coordinates to top left corner of detector and convert to mm
    def trans(point):
        return (point - screen_top) * 1e-3

    beam_start = trans(beam_start)
    beam_end = trans(beam_end)
    sample_start = trans(sample_start)
    sample_end = trans(sample_end)
    detector_start = trans(screen_top)
    detector_end = trans(screen_bottom)
    pc_pos = trans(P)
    source_pos = trans(np.zeros(2, dtype=np.float64))

    beam_color = "#c4761c"
    ax.annotate(
        "",
        xy=beam_end,
        xytext=beam_start,
        arrowprops={"arrowstyle": "->", "lw": 2, "color": beam_color},
        zorder=5,
    )
    ax.plot(
        [sample_start[0], sample_end[0]],
        [sample_start[1], sample_end[1]],
        c="gray",
        lw=3,
        label="Sample",
        zorder=4,
    )
    ax.plot(
        [detector_start[0], detector_end[0]],
        [detector_start[1], detector_end[1]],
        c="purple",
        lw=4,
        label="Detector",
        zorder=4,
    )
    ax.plot(
        pc_pos[0],
        pc_pos[1],
        marker="*",
        ms=15,
        mfc="gold",
        mec="k",
        linestyle="None",
        label="PC",
        zorder=5,
    )
    ax.plot(
        [source_pos[0], pc_pos[0]],
        [source_pos[1], pc_pos[1]],
        linestyle=":",
        color=beam_color,
        alpha=0.5,
        zorder=0,
    )

    # Handle axis orientation: Y axis as x-coordinate, Z as
    # y-coordinate, Z pointing down
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.invert_xaxis()

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
            sample_end[1],
            "Sample",
            ha="right",
            va="center",
            label="sample_annotation",
        )
        ax.annotate(
            "Detector",
            xy=(detector_start[0], detector_start[1]),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            va="top",
            label="detector_annotation",
        )
        ax.annotate(
            "PC",
            xy=(pc_pos[0], pc_pos[1]),
            xytext=(5, -5),
            textcoords="offset points",
            ha="left",
            va="top",
            label="pc_annotation",
        )

    if dimensionless:
        ax.axis("off")
    else:
        unit = "mm"
        ax.set_xlabel(f"Microscope Y axis [{unit}]")
        ax.set_ylabel(f"Microscope Z axis [{unit}]")

    return fig


def plot_ebsd_detector_geometry_side_view_interactive(
    detector: EBSDDetector,
    inplace: bool = False,
    ax: maxes.Axes | None = None,
    annotate: bool = False,
    dimensionless: bool = True,
    **kwargs,
) -> "widgets.VBox":
    """Plot an interactive EBSD detector geometry side view.

    The plot allows interactive modification of the detector and sample
    tilts and the projection center (PC) values.

    Parameters
    ----------
    detector
        EBSD detector.
    inplace
        Whether to edit the given *detector* inplace. The given detector
        is not affected by default.
    ax
        The Matplotlib axis to plot in. If not given, a new figure
        and axis are created.
    annotate
        Whether to annotate the various components of the geometry.
        Default is False.
    dimensionless
        Whether to ignore the
        :attr:`~kikuchipy.detectors.EBSDDetector.px_size` when
        drawing the plot axes. Default is True.
    **kwargs
        Keyword arguments passed to
        :func:`~matplotlib.pyplot.figure` if *ax* is not given.

    Returns
    -------
    widgets
        The widget containing the sliders. Required to display the
        interactive controls.

    Notes
    -----
    Requires that :mod:`ipywidgets` is installed.
    """
    if dependency_version["ipywidgets"] is None:
        raise ImportError("Requires that ipywidgets is installed")
    if dependency_version["IPython"] is None:
        raise ImportError("Requires that IPython is installed")

    import ipywidgets as widgets

    if inplace:
        det = detector
    else:
        det = detector.deepcopy()
    pcx, pcy, pcz = det.pc_average
    det.pc = [pcx, pcy, pcz]

    if ax is None:
        w, h = plt.rcParams["figure.figsize"]
        kwargs.setdefault("layout", "constrained")
        kwargs.setdefault("figsize", (w, h))
        fig = plt.figure(**kwargs)
        ax = fig.add_subplot()
    else:
        fig = ax.figure

    def get_range(val, default_range):
        vmin, vmax = default_range
        margin = max((vmax - vmin) * 0.1, 0.1)
        if val < vmin:
            vmin = val - margin
        if val > vmax:
            vmax = val + margin
        return vmin, vmax

    px_min, px_max = get_range(pcx, (0, 1))
    py_min, py_max = get_range(pcy, (0, 1))
    pz_min, pz_max = get_range(pcz, (0, 1))

    style = {"description_width": "initial"}
    s_st = widgets.FloatSlider(
        value=det.sample_tilt,
        min=0,
        max=180,
        step=0.1,
        description="Sample tilt",
        style=style,
    )
    s_dt = widgets.FloatSlider(
        value=det.tilt,
        min=0,
        max=180,
        step=0.1,
        description="Detector tilt",
        style=style,
    )
    s_px = widgets.FloatSlider(
        value=pcx, min=px_min, max=px_max, step=0.01, description="PCx", style=style
    )
    s_py = widgets.FloatSlider(
        value=pcy, min=py_min, max=py_max, step=0.01, description="PCy", style=style
    )
    s_pz = widgets.FloatSlider(
        value=pcz, min=pz_min, max=pz_max, step=0.01, description="PCz", style=style
    )

    def update(change: Any = None) -> None:
        det.sample_tilt = s_st.value
        det.tilt = s_dt.value
        det.pc = [s_px.value, s_py.value, s_pz.value]

        # TODO: Look into blitting instead of clearing everything
        ax.clear()

        plot_ebsd_detector_geometry_side_view(
            det, ax=ax, annotate=annotate, dimensionless=dimensionless
        )

        fig.canvas.draw_idle()

    for s in [s_st, s_dt, s_px, s_py, s_pz]:
        s.observe(update, names="value")

    update()

    widget = widgets.VBox([s_st, s_dt, s_px, s_py, s_pz])

    return widget
