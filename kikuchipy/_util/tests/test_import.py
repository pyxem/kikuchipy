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

import pytest


class TestImport:
    def test_import_version(self):
        from kikuchipy import __version__

        assert isinstance(__version__, str)

    def test_import(self):
        import kikuchipy

        for obj_name in kikuchipy.__all__:
            _ = getattr(kikuchipy, obj_name)
        with pytest.raises(
            AttributeError, match="module 'kikuchipy' has no attribute 'foo'"
        ):
            _ = kikuchipy.foo

    def test_import_pyvista_installed(self):
        from kikuchipy import _pyvista_installed

        assert isinstance(_pyvista_installed, bool)

    def test_import_data(self):
        import kikuchipy.data

        for obj_name in kikuchipy.data.__all__:
            _ = getattr(kikuchipy.data, obj_name)
        with pytest.raises(
            AttributeError, match="module 'kikuchipy.data' has no attribute 'foo'"
        ):
            _ = kikuchipy.data.foo

    def test_import_detectors(self):
        import kikuchipy.detectors

        for obj_name in kikuchipy.detectors.__all__:
            _ = getattr(kikuchipy.detectors, obj_name)
        with pytest.raises(
            AttributeError, match="module 'kikuchipy.detectors' has no attribute 'foo'"
        ):
            _ = kikuchipy.detectors.foo

    def test_import_draw(self):
        import kikuchipy.draw

        for obj_name in kikuchipy.draw.__all__:
            _ = getattr(kikuchipy.draw, obj_name)
        with pytest.raises(
            AttributeError, match="module 'kikuchipy.draw' has no attribute 'foo'"
        ):
            _ = kikuchipy.draw.foo

    def test_import_filters(self):
        import kikuchipy.filters

        for obj_name in kikuchipy.filters.__all__:
            _ = getattr(kikuchipy.filters, obj_name)
        with pytest.raises(
            AttributeError, match="module 'kikuchipy.filters' has no attribute 'foo'"
        ):
            _ = kikuchipy.filters.foo

    def test_import_imaging(self):
        import kikuchipy.imaging

        for obj_name in kikuchipy.imaging.__all__:
            _ = getattr(kikuchipy.imaging, obj_name)
        with pytest.raises(
            AttributeError, match="module 'kikuchipy.imaging' has no attribute 'foo'"
        ):
            _ = kikuchipy.imaging.foo

    def test_import_indexing(self):
        import kikuchipy.indexing

        for obj_name in kikuchipy.indexing.__all__:
            _ = getattr(kikuchipy.indexing, obj_name)
        with pytest.raises(
            AttributeError, match="module 'kikuchipy.indexing' has no attribute 'foo'"
        ):
            _ = kikuchipy.indexing.foo

    def test_import_io(self):
        import kikuchipy.io

        for obj_name in kikuchipy.io.__all__:
            _ = getattr(kikuchipy.io, obj_name)
        with pytest.raises(
            AttributeError, match="module 'kikuchipy.io' has no attribute 'foo'"
        ):
            _ = kikuchipy.io.foo

    def test_import_io_plugins(self):
        import kikuchipy.io.plugins

        for obj_name in kikuchipy.io.plugins.__all__:
            plugin_module = getattr(kikuchipy.io.plugins, obj_name)
            for obj_name2 in plugin_module.__all__:
                _ = getattr(plugin_module, obj_name2)
            plugin_module_name = plugin_module.__name__
            with pytest.raises(
                AttributeError, match=f"module '{plugin_module_name}' has no attribute"
            ):
                _ = getattr(plugin_module, "foo")

        with pytest.raises(
            AttributeError, match="module 'kikuchipy.io.plugins' has no attribute 'foo'"
        ):
            _ = kikuchipy.io.plugins.foo

    def test_import_load(self):
        from kikuchipy import load

        assert callable(load)

    def test_import_pattern(self):
        import kikuchipy.pattern

        for obj_name in kikuchipy.pattern.__all__:
            _ = getattr(kikuchipy.pattern, obj_name)
        with pytest.raises(
            AttributeError, match="module 'kikuchipy.pattern' has no attribute 'foo'"
        ):
            _ = kikuchipy.pattern.foo

    def test_import_signals(self):
        import kikuchipy.signals

        for obj_name in kikuchipy.signals.__all__:
            _ = getattr(kikuchipy.signals, obj_name)
        with pytest.raises(
            AttributeError, match="module 'kikuchipy.signals' has no attribute 'foo'"
        ):
            _ = kikuchipy.signals.foo

    def test_import_signals_util(self):
        import kikuchipy.signals.util

        for obj_name in kikuchipy.signals.util.__all__:
            _ = getattr(kikuchipy.signals.util, obj_name)
        with pytest.raises(
            AttributeError,
            match="module 'kikuchipy.signals.util' has no attribute 'foo'",
        ):
            _ = kikuchipy.signals.util.foo

    def test_import_simulations(self):
        import kikuchipy.simulations

        for obj_name in kikuchipy.simulations.__all__:
            _ = getattr(kikuchipy.simulations, obj_name)
        with pytest.raises(
            AttributeError,
            match="module 'kikuchipy.simulations' has no attribute 'foo'",
        ):
            _ = kikuchipy.simulations.foo

    def test_dir(self):
        import kikuchipy

        assert dir(kikuchipy) == [
            "__version__",
            "_pyebsdindex_installed",
            "_pyvista_installed",
            "data",
            "detectors",
            "draw",
            "filters",
            "imaging",
            "indexing",
            "io",
            "load",
            "pattern",
            "release",
            "set_log_level",
            "signals",
            "simulations",
        ]

    def test_dir_data(self):
        import kikuchipy.data

        assert dir(kikuchipy.data) == [
            "ebsd_master_pattern",
            "ni_gain",
            "ni_gain_calibration",
            "nickel_ebsd_large",
            "nickel_ebsd_master_pattern_small",
            "nickel_ebsd_small",
            "si_ebsd_moving_screen",
            "si_wafer",
        ]

    def test_dir_detectors(self):
        import kikuchipy.detectors

        assert dir(kikuchipy.detectors) == ["EBSDDetector", "PCCalibrationMovingScreen"]

    def test_dir_draw(self):
        import kikuchipy.draw

        assert dir(kikuchipy.draw) == [
            "get_rgb_navigator",
            "plot_pattern_positions_in_map",
        ]

    def test_dir_filters(self):
        import kikuchipy.filters

        assert dir(kikuchipy.filters) == [
            "Window",
            "distance_to_origin",
            "highpass_fft_filter",
            "lowpass_fft_filter",
            "modified_hann",
        ]

    def test_dir_imaging(self):
        import kikuchipy.imaging

        assert dir(kikuchipy.imaging) == [
            "VirtualBSEImager",
        ]

    def test_dir_indexing(self):
        import kikuchipy.indexing

        assert dir(kikuchipy.indexing) == [
            "NormalizedCrossCorrelationMetric",
            "NormalizedDotProductMetric",
            "SimilarityMetric",
            "compute_refine_orientation_projection_center_results",
            "compute_refine_orientation_results",
            "compute_refine_projection_center_results",
            "merge_crystal_maps",
            "orientation_similarity_map",
            "xmap_from_hough_indexing_data",
        ]

    def test_dir_io(self):
        import kikuchipy.io

        assert dir(kikuchipy.io) == ["plugins"]

    def test_dir_io_plugins(self):
        import kikuchipy.io.plugins
        from kikuchipy.io._io import plugins as plugins_list

        assert dir(kikuchipy.io.plugins) == [
            "bruker_h5ebsd",
            "ebsd_directory",
            "edax_binary",
            "edax_h5ebsd",
            "emsoft_ebsd",
            "emsoft_ebsd_master_pattern",
            "emsoft_ecp_master_pattern",
            "emsoft_tkd_master_pattern",
            "kikuchipy_h5ebsd",
            "nordif",
            "nordif_calibration_patterns",
            "oxford_binary",
            "oxford_h5ebsd",
        ]

        for spec in plugins_list:
            plugin_name = spec["name"]
            if plugin_name in ["HSPY", "ZSPY"]:
                continue
            plugin_dir = ["file_reader"]
            if spec["writes"]:
                plugin_dir.append("file_writer")
            plugin_module = getattr(kikuchipy.io.plugins, plugin_name)
            assert dir(plugin_module) == plugin_dir

    def test_dir_pattern(self):
        import kikuchipy.pattern

        assert dir(kikuchipy.pattern) == [
            "chunk",
            "fft",
            "fft_filter",
            "fft_frequency_vectors",
            "fft_spectrum",
            "get_dynamic_background",
            "get_image_quality",
            "ifft",
            "normalize_intensity",
            "remove_dynamic_background",
            "rescale_intensity",
        ]

    def test_dir_signals(self):
        import kikuchipy.signals

        assert dir(kikuchipy.signals) == [
            "EBSD",
            "EBSDMasterPattern",
            "ECPMasterPattern",
            "LazyEBSD",
            "LazyEBSDMasterPattern",
            "LazyECPMasterPattern",
            "LazyVirtualBSEImage",
            "VirtualBSEImage",
            "util",
        ]

    def test_dir_signals_util(self):
        import kikuchipy.signals.util

        assert dir(kikuchipy.signals.util) == [
            "get_chunking",
            "get_dask_array",
            "grid_indices",
        ]

    def test_dir_simulations(self):
        import kikuchipy.simulations

        assert dir(kikuchipy.simulations) == [
            "GeometricalKikuchiPatternSimulation",
            "KikuchiPatternSimulator",
        ]
