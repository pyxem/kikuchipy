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

import logging

import numpy as np
import pytest

import kikuchipy as kp
from kikuchipy import _constants
from kikuchipy.pattern._processing_optimization import (
    _get_parameters,
    _get_searched_parameter,
)


class TestOptimizePatternProcessing:
    def test_nelder_mead(self, nickel_pattern_and_reference):
        """Optimization with Nelder-Mead improves the similarity to the
        reference, and the result is complete and consistent.
        """
        pattern, reference = nickel_pattern_and_reference
        n_calls = 60
        result = kp.pattern.optimize_pattern_processing(
            pattern,
            reference,
            method="nelder-mead",
            n_calls=n_calls,
            random_state=0,
        )

        # Steps
        steps = ("remove_dynamic_background", "bandpass_filter")
        assert result["steps"] == steps
        assert list(result["parameters"]) == list(steps)

        # Results
        assert list(result["patterns"]) == ["raw"] + list(steps)
        for processed in result["patterns"].values():
            assert processed.shape == pattern.shape
            assert processed.dtype == pattern.dtype
        assert result["image_quality"].shape == (3,)
        assert result["signal_mask"] is None
        # Optimized pattern is more similar to the reference than the
        # raw pattern
        ncc = result["normalized_cross_correlation"]
        assert ncc.shape == (3,)
        assert result["score"] == ncc[-1]
        assert result["score"] > ncc[0]

        # Evaluations are shared between the start and the restarts
        assert len(result["optimizer"]) == 3
        n_evaluations = 0
        for start_result in result["optimizer"]:
            n_evaluations += start_result.nfev
        assert n_evaluations <= n_calls + 3 * 2  # May exceed by a few

    def test_nelder_mead_reproducible(self, nickel_pattern_and_reference):
        pattern, reference = nickel_pattern_and_reference
        kwargs = dict(method="nelder-mead", n_calls=40, random_state=1)
        result1 = kp.pattern.optimize_pattern_processing(pattern, reference, **kwargs)
        result2 = kp.pattern.optimize_pattern_processing(pattern, reference, **kwargs)
        assert result1["parameters"] == result2["parameters"]
        assert result1["score"] == result2["score"]

    @pytest.mark.skipif_bayesian_optimization_not_installed
    def test_bayesian(self, nickel_pattern_and_reference):
        """Bayesian optimization (default) improves the similarity to
        the reference, evaluates the given number of calls, and is
        reproducible.
        """
        pattern, reference = nickel_pattern_and_reference
        kwargs = dict(n_calls=12, n_initial_points=5, random_state=0)
        result1 = kp.pattern.optimize_pattern_processing(pattern, reference, **kwargs)
        result2 = kp.pattern.optimize_pattern_processing(pattern, reference, **kwargs)

        ncc = result1["normalized_cross_correlation"]
        assert result1["score"] > ncc[0]
        assert len(result1["optimizer"].res) == 12
        assert result1["parameters"] == result2["parameters"]

    def test_steps_in_given_order(self, nickel_pattern_and_reference):
        pattern, reference = nickel_pattern_and_reference
        steps = [
            "bandpass_filter",
            "adaptive_histogram_equalization",
            "remove_dynamic_background",
        ]
        result = kp.pattern.optimize_pattern_processing(
            pattern, reference, steps=steps, method="nelder-mead", n_calls=30
        )
        assert result["steps"] == tuple(steps)
        assert list(result["patterns"]) == ["raw"] + steps

    def test_fixed_and_searched_parameters(self, nickel_pattern_and_reference):
        """Fixed parameters are kept, and searched parameters are within
        their search space and of the expected type.
        """
        pattern, reference = nickel_pattern_and_reference
        result = kp.pattern.optimize_pattern_processing(
            pattern,
            reference,
            steps=["remove_dynamic_background", "adaptive_histogram_equalization"],
            method="nelder-mead",
            n_calls=40,
            remove_dynamic_background={
                "std": [4.0, 8.0],
                "truncate": 4.0,
                "operation": "divide",
            },
            adaptive_histogram_equalization={
                "kernel_size": (10, 20),
                "clip_limit": (1e-3, 1e-2, "log"),
                "nbins": 128,
            },
        )
        params_dbr = result["parameters"]["remove_dynamic_background"]
        assert params_dbr["std"] in [4.0, 8.0]
        assert params_dbr["truncate"] == 4.0
        assert params_dbr["operation"] == "divide"
        assert params_dbr["filter_domain"] == "frequency"

        params_ahe = result["parameters"]["adaptive_histogram_equalization"]
        assert isinstance(params_ahe["kernel_size"], int)
        assert 10 <= params_ahe["kernel_size"] <= 20
        assert isinstance(params_ahe["clip_limit"], float)
        assert 1e-3 <= params_ahe["clip_limit"] <= 1e-2
        assert params_ahe["nbins"] == 128

    def test_signal_mask(self, nickel_pattern_and_reference):
        pattern, reference = nickel_pattern_and_reference
        signal_mask = np.zeros(pattern.shape, dtype=bool)
        signal_mask[:10] = True
        result = kp.pattern.optimize_pattern_processing(
            pattern,
            reference,
            signal_mask=signal_mask,
            method="nelder-mead",
            n_calls=30,
        )
        assert np.array_equal(result["signal_mask"], signal_mask)

    @pytest.mark.parametrize(
        "func, step",
        [
            (
                kp.pattern.optimize_remove_dynamic_background,
                "remove_dynamic_background",
            ),
            (
                kp.pattern.optimize_adaptive_histogram_equalization,
                "adaptive_histogram_equalization",
            ),
            (kp.pattern.optimize_bandpass_filter, "bandpass_filter"),
        ],
    )
    def test_single_step(self, nickel_pattern_and_reference, func, step):
        pattern, reference = nickel_pattern_and_reference
        result = func(pattern, reference, method="nelder-mead", n_calls=30)
        assert result["steps"] == (step,)
        assert list(result["parameters"]) == [step]
        ncc = result["normalized_cross_correlation"]
        assert result["score"] >= ncc[0]

    def test_logging(self, nickel_pattern_and_reference, caplog):
        pattern, reference = nickel_pattern_and_reference
        logger_name = "kikuchipy.pattern._processing_optimization"
        with caplog.at_level(logging.INFO, logger=logger_name):
            kp.pattern.optimize_pattern_processing(
                pattern, reference, method="nelder-mead", n_calls=30
            )
        assert "Optimize 4 parameters of steps" in caplog.text
        assert "best NCC" in caplog.text
        assert "Best NCC" in caplog.text


class TestOptimizePatternProcessingRaises:
    def test_unknown_method(self, nickel_pattern_and_reference):
        pattern, reference = nickel_pattern_and_reference
        with pytest.raises(ValueError, match="Unknown optimization method 'powell'"):
            kp.pattern.optimize_pattern_processing(pattern, reference, method="powell")

    def test_bayesian_requires_dependency(
        self, nickel_pattern_and_reference, monkeypatch
    ):
        pattern, reference = nickel_pattern_and_reference
        monkeypatch.setitem(
            _constants.dependency_version, "bayesian-optimization", None
        )
        with pytest.raises(ImportError, match="requires that 'bayesian-optimization'"):
            kp.pattern.optimize_pattern_processing(pattern, reference)

    @pytest.mark.skipif_bayesian_optimization_not_installed
    @pytest.mark.parametrize("n_initial_points", [0, 11])
    def test_invalid_n_initial_points(
        self, nickel_pattern_and_reference, n_initial_points
    ):
        pattern, reference = nickel_pattern_and_reference
        with pytest.raises(ValueError, match="Number of initial points "):
            kp.pattern.optimize_pattern_processing(
                pattern, reference, n_calls=10, n_initial_points=n_initial_points
            )

    def test_invalid_n_restarts(self, nickel_pattern_and_reference):
        pattern, reference = nickel_pattern_and_reference
        with pytest.raises(ValueError, match="Number of restarts -1 must be at least"):
            kp.pattern.optimize_pattern_processing(
                pattern, reference, method="nelder-mead", n_restarts=-1
            )

    def test_too_few_calls_for_nelder_mead(self, nickel_pattern_and_reference):
        pattern, reference = nickel_pattern_and_reference
        # 4 parameters with 2 restarts require at least 3 * (4 + 2) calls
        with pytest.raises(ValueError, match="Number of calls 10 must be at least 18"):
            kp.pattern.optimize_pattern_processing(
                pattern, reference, method="nelder-mead", n_calls=10
            )

    @pytest.mark.parametrize("pattern_shape", [(60, 59), (1, 60, 60)])
    def test_invalid_pattern_shape(self, nickel_pattern_and_reference, pattern_shape):
        _, reference = nickel_pattern_and_reference
        pattern = np.zeros(pattern_shape, dtype=np.uint8)
        with pytest.raises(ValueError, match="Pattern shape .* and reference shape"):
            kp.pattern.optimize_pattern_processing(
                pattern, reference, method="nelder-mead"
            )

    def test_invalid_signal_mask_shape(self, nickel_pattern_and_reference):
        pattern, reference = nickel_pattern_and_reference
        signal_mask = np.zeros((3, 3), dtype=bool)
        with pytest.raises(ValueError, match=r"Signal mask shape \(3, 3\) and pattern"):
            kp.pattern.optimize_pattern_processing(
                pattern, reference, signal_mask=signal_mask, method="nelder-mead"
            )

    def test_all_parameters_fixed(self, nickel_pattern_and_reference):
        pattern, reference = nickel_pattern_and_reference
        with pytest.raises(ValueError, match="No parameters to optimize"):
            kp.pattern.optimize_bandpass_filter(
                pattern,
                reference,
                highpass_cutoff=2,
                lowpass_cutoff=20,
                method="nelder-mead",
            )

    def test_unknown_step(self, nickel_pattern_and_reference):
        pattern, reference = nickel_pattern_and_reference
        with pytest.raises(ValueError, match="Unknown processing step 'binning'"):
            kp.pattern.optimize_pattern_processing(
                pattern, reference, steps=["binning"], method="nelder-mead"
            )

    def test_duplicate_steps(self, nickel_pattern_and_reference):
        pattern, reference = nickel_pattern_and_reference
        with pytest.raises(ValueError, match="cannot contain duplicates"):
            kp.pattern.optimize_pattern_processing(
                pattern,
                reference,
                steps=["bandpass_filter", "bandpass_filter"],
                method="nelder-mead",
            )

    def test_parameters_for_step_not_in_steps(self, nickel_pattern_and_reference):
        pattern, reference = nickel_pattern_and_reference
        with pytest.raises(
            ValueError,
            match="Parameters given for step 'adaptive_histogram_equalization'",
        ):
            kp.pattern.optimize_pattern_processing(
                pattern,
                reference,
                method="nelder-mead",
                adaptive_histogram_equalization={"nbins": 128},
            )

    def test_unknown_parameter(self, nickel_pattern_and_reference):
        pattern, reference = nickel_pattern_and_reference
        with pytest.raises(ValueError, match=r"Unknown parameters \['cutoff'\]"):
            kp.pattern.optimize_pattern_processing(
                pattern,
                reference,
                method="nelder-mead",
                bandpass_filter={"cutoff": 10},
            )

    def test_float_pattern_with_equalization_warns(self, nickel_pattern_and_reference):
        pattern, reference = nickel_pattern_and_reference
        with pytest.warns(UserWarning, match="Equalization of patterns with floating"):
            kp.pattern.optimize_adaptive_histogram_equalization(
                pattern.astype(np.float32) / 255,  # Within [0, 1] for skimage
                reference,
                method="nelder-mead",
                n_calls=20,
                n_restarts=0,
            )


class TestSearchedParameter:
    def test_fixed(self):
        assert _get_searched_parameter("a", "b", 2) is None
        assert _get_searched_parameter("a", "b", 2.5) is None
        assert _get_searched_parameter("a", "b", "subtract") is None

    def test_integer_range(self):
        param = _get_searched_parameter("a", "b", (7, 30))
        assert param.key == "a.b"
        assert param.pbound == (6.5, 30.5)
        # Rounded and clipped to the integer range
        assert param.to_value(6.5) == 7
        assert param.to_value(18.4) == 18
        assert param.to_value(30.5) == 30
        assert isinstance(param.to_value(18.4), int)

        param2 = _get_searched_parameter("a", "b", (np.int64(7), 30))
        assert param2.integer

    def test_float_range(self):
        param = _get_searched_parameter("a", "b", (1.0, 15.0))
        assert param.pbound == (1.0, 15.0)
        value = param.to_value(np.float64(3.5))
        assert value == 3.5
        assert type(value) is float

    def test_log_range(self):
        param = _get_searched_parameter("a", "b", (1e-4, 1e-2, "log"))
        assert np.allclose(param.pbound, (-4, -2))
        assert np.isclose(param.to_value(-4), 1e-4)
        assert np.isclose(param.to_value(-3), 1e-3)

        param_int = _get_searched_parameter("a", "b", (1, 100, "log"))
        assert param_int.to_value(1) == 10
        assert isinstance(param_int.to_value(1), int)

    def test_candidates(self):
        param = _get_searched_parameter("a", "b", np.array([128, 256, 512]))
        assert param.pbound == (0.0, 3.0)
        assert param.to_value(0) == 128
        assert param.to_value(1.5) == 256
        assert param.to_value(3.0) == 512  # Upper bound is the last candidate
        # Plain Python types
        assert type(param.to_value(0)) is int

    @pytest.mark.parametrize("spec", [[128], [128, 128]])
    def test_invalid_candidates(self, spec):
        with pytest.raises(ValueError, match="must be at least two unique values"):
            _get_searched_parameter("a", "b", spec)

    def test_invalid_range(self):
        with pytest.raises(ValueError, match=r"must be \(low, high\) or \(low, high, "):
            _get_searched_parameter("a", "b", (1, 2, "log", 4))

    def test_invalid_scale(self):
        with pytest.raises(
            ValueError, match="Scale of 'a.b' must be 'linear' or 'log'"
        ):
            _get_searched_parameter("a", "b", (1, 2, "log-uniform"))

    def test_invalid_log_range(self):
        with pytest.raises(ValueError, match="on a log scale must be positive"):
            _get_searched_parameter("a", "b", (0, 2, "log"))


class TestGetParameters:
    def test_size_defaults(self):
        """Parameters not given or given as None depend on the pattern
        width, while given parameters are kept.
        """
        width = 60
        steps = ("remove_dynamic_background", "bandpass_filter")
        params = _get_parameters(
            steps, {"bandpass_filter": {"lowpass_cutoff": None}}, width
        )
        assert params["remove_dynamic_background"]["std"] == (width / 32, width / 4)
        assert params["remove_dynamic_background"]["truncate"] == (2.0, 10.0)
        assert params["bandpass_filter"]["lowpass_cutoff"] == (width / 4, width / 2)
        assert params["bandpass_filter"]["highpass_cutoff_width"] == width / 30

        params2 = _get_parameters(
            ("adaptive_histogram_equalization",),
            {"adaptive_histogram_equalization": {"kernel_size": 16}},
            width,
        )
        assert params2["adaptive_histogram_equalization"]["kernel_size"] == 16
        assert params2["adaptive_histogram_equalization"]["clip_limit"] == (
            1e-4,
            5e-3,
            "log",
        )
