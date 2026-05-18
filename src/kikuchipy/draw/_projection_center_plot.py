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

"""Plotting projection centers."""

from typing import Any, Literal

import matplotlib.figure as mfigure
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

PROJECTION_CENTER_PLOT_MODES = Literal["map", "scatter", "3d"]


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
