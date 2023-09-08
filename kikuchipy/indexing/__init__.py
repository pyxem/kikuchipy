# Copyright 2019-2023 The kikuchipy developers
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

"""Tools for indexing of EBSD patterns by matching to a dictionary of
simulated patterns.

Some of these tools are used in
:meth:`~kikuchipy.signals.EBSD.dictionary_indexing`.
"""

__all__ = [
    "NormalizedCrossCorrelationMetric",
    "NormalizedDotProductMetric",
    "SimilarityMetric",
    "PCAIndexer",
    "DIIndexer",
    "compute_refine_orientation_projection_center_results",
    "compute_refine_orientation_results",
    "compute_refine_projection_center_results",
    "merge_crystal_maps",
    "orientation_similarity_map",
    "xmap_from_hough_indexing_data",
]


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    _import_mapping = {
        "NormalizedCrossCorrelationMetric": "similarity_metrics",
        "NormalizedDotProductMetric": "similarity_metrics",
        "SimilarityMetric": "similarity_metrics",
        "PCAIndexer": "di_indexers",
        "DIIndexer": "di_indexers",
        "compute_refine_orientation_projection_center_results": "_refinement._refinement",
        "compute_refine_orientation_results": "_refinement._refinement",
        "compute_refine_projection_center_results": "_refinement._refinement",
        "merge_crystal_maps": "_merge_crystal_maps",
        "orientation_similarity_map": "_orientation_similarity_map",
        "xmap_from_hough_indexing_data": "_hough_indexing",
    }
    if name in __all__:
        import importlib

        if name in _import_mapping.keys():
            import_path = f"{__name__}.{_import_mapping.get(name)}"
            return getattr(importlib.import_module(import_path), name)
        else:  # pragma: no cover
            return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
