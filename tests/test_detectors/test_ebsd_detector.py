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

from copy import deepcopy

import matplotlib.collections as mcollections
import matplotlib.pyplot as plt
import numpy as np
from orix.crystal_map import PhaseList
import orix.quaternion as oqu
import orix.vector as ove
import pytest

import kikuchipy as kp
from kikuchipy._constants import dependency_version

skipif_pyebsdindex_installed = pytest.mark.skipif(
    dependency_version["pyebsdindex"] is not None, reason="pyebsdindex is installed"
)
skipif_pyebsdindex_not_installed = pytest.mark.skipif(
    dependency_version["pyebsdindex"] is None, reason="pyebsdindex is not installed"
)


class TestEBSDDetector:
    def test_init(self, pc1):
        """Initialization works."""
        shape = (1, 2)
        px_size = 3
        binning = 4
        tilt = 5
        det = kp.detectors.EBSDDetector(
            shape=shape, px_size=px_size, binning=binning, tilt=tilt, pc=pc1
        )
        assert det.shape == shape
        assert det.aspect_ratio == 2
        assert np.issubdtype(det.pc.dtype, np.floating)
        for attr in [det.sample_tilt, det.tilt, det.azimuthal, det.px_size]:
            assert type(attr) is float

    def test_set_detector_shape(self):
        det = kp.detectors.EBSDDetector()
        with pytest.raises(ValueError, match="Invalid shape 2. Must be an iterable of"):
            det.shape = 2
        with pytest.raises(ValueError, match=r"Invalid shape \(2,\). Must be an "):
            det.shape = (2,)
        with pytest.raises(ValueError, match=r"Invalid shape \(2, 'a'\). Must be an "):
            det.shape = (2, "a")

    @pytest.mark.parametrize(
        "nav_shape, desired_nav_shape, desired_nav_dim",
        [
            ((), (1,), 1),
            ((1,), (1,), 1),
            ((10, 1), (10,), 1),
            ((10, 10, 1), (10, 10), 2),
        ],
    )
    def test_nav_shape_dim(self, pc1, nav_shape, desired_nav_shape, desired_nav_dim):
        """Navigation shape and dimension is derived correctly from PC shape."""
        det = kp.detectors.EBSDDetector(pc=np.tile(pc1, nav_shape))
        assert det.navigation_shape == desired_nav_shape
        assert det.navigation_dimension == desired_nav_dim

    @pytest.mark.parametrize("pc_type", [list, tuple, np.asarray])
    def test_pc_initialization(self, pc1, pc_type):
        """Initialize PC of valid types."""
        det = kp.detectors.EBSDDetector(pc=pc_type(pc1))
        assert isinstance(det.pc, np.ndarray)

    @pytest.mark.parametrize(
        (
            "shape, px_size, binning, pc, ssd, width, height, size, "
            "shape_unbinned, px_size_binned"
        ),
        [
            # fmt: off
            ((60, 60), 70, 8, [1, 1, 0.5], 16800, 33600, 33600, 3600, (480, 480), 560),
            ((60, 60), 70, 8, [1, 1, 0.7], 23520, 33600, 33600, 3600, (480, 480), 560),
            (
                (480, 460),
                70,
                0.5,
                [1, 1, 0.7],
                11760,
                16100,
                16800,
                220800,
                (240, 230),
                35,
            ),
            (
                (340, 680),
                40,
                2,
                [1, 1, 0.7],
                19040,
                54400,
                27200,
                231200,
                (680, 1360),
                80,
            ),
            # fmt: on
        ],
    )
    def test_detector_dimensions(
        self,
        shape,
        px_size,
        binning,
        pc,
        ssd,
        width,
        height,
        size,
        shape_unbinned,
        px_size_binned,
    ):
        """Initialization yields expected derived values."""
        det = kp.detectors.EBSDDetector(
            shape=shape, px_size=px_size, binning=binning, pc=pc
        )
        print(det)
        assert det.specimen_scintillator_distance == ssd
        assert det.width == width
        assert det.height == height
        assert det.size == size
        assert det.unbinned_shape == shape_unbinned
        assert det.px_size_binned == px_size_binned

    def test_detector_repr(self, pc1):
        """Expected string representation."""
        det = kp.detectors.EBSDDetector(
            shape=(1, 2), px_size=3, binning=4, tilt=5, azimuthal=2, twist=1.02, pc=pc1
        )
        assert repr(det) == (
            "EBSDDetector\n"
            "  shape (Ny, Nx):     (1, 2)\n"
            "  pc (PCx, PCy, PCz): (0.421, 0.779, 0.505)\n"
            "  sample_tilt:        70.0\N{DEGREE SIGN}\n"
            "  tilt:               5.0\N{DEGREE SIGN}\n"
            "  azimuthal:          2.0\N{DEGREE SIGN}\n"
            "  twist:              1.02\N{DEGREE SIGN}\n"
            "  binning:            4\n"
            "  px_size:            3.0 um"
        )

    def test_deepcopy(self, pc1):
        """Yields the expected parameters and an actual deep copy."""
        detector1 = kp.detectors.EBSDDetector(pc=pc1)
        detector2 = detector1.deepcopy()
        detector1.pcx += 0.1
        assert np.allclose(detector1.pcx, 0.521)
        assert np.allclose(detector2.pcx, 0.421)

    def test_set_pc_coordinates(self, pc1):
        """Returns desired arrays with desired shapes."""
        ny, nx = (2, 3)
        nav_shape = (ny, nx)
        n = ny * nx
        detector = kp.detectors.EBSDDetector(pc=np.tile(pc1, nav_shape + (1,)))
        assert detector.navigation_shape == nav_shape

        new_pc = np.zeros(nav_shape + (3,))
        new_pc[..., 0] = pc1[0] * 0.01 * np.arange(n).reshape(nav_shape)
        new_pc[..., 1] = pc1[1] * 0.01 * np.arange(n).reshape(nav_shape)
        new_pc[..., 2] = pc1[2] * 0.01 * np.arange(n).reshape(nav_shape)
        detector.pcx = new_pc[..., 0]
        detector.pcy = new_pc[..., 1]
        detector.pcz = new_pc[..., 2]
        assert np.allclose(detector.pc, new_pc)

    @pytest.mark.parametrize(
        "pc, desired_pc_average",
        [
            ([0.1234, 0.1235, 0.1234], [0.1230, 0.1240, 0.1230]),
            (np.arange(30).reshape((2, 5, 3)), [13.5, 14.5, 15.5]),
            (np.arange(30).reshape((10, 3)), [13.5, 14.5, 15.5]),
        ],
    )
    def test_pc_average(self, pc, desired_pc_average):
        """Calculation of PC average."""
        assert np.allclose(
            kp.detectors.EBSDDetector(pc=pc).pc_average,
            desired_pc_average,
            atol=1e-3,
        )

    @pytest.mark.parametrize(
        "pc, desired_nav_shape, desired_nav_ndim",
        [
            (np.arange(30).reshape((2, 5, 3)), (5, 2), 2),
            (np.arange(30).reshape((5, 2, 3)), (10, 1), 2),
            (np.arange(30).reshape((2, 5, 3)), (10,), 1),
        ],
    )
    def test_set_navigation_shape(self, pc, desired_nav_shape, desired_nav_ndim):
        """Change shape of PC array."""
        detector = kp.detectors.EBSDDetector(pc=pc)
        detector.navigation_shape = desired_nav_shape
        assert detector.navigation_shape == desired_nav_shape
        assert detector.navigation_dimension == desired_nav_ndim
        assert detector.pc.shape == desired_nav_shape + (3,)

    def test_set_navigation_shape_raises(self, pc1):
        """Desired error message."""
        detector = kp.detectors.EBSDDetector(pc=pc1)
        with pytest.raises(ValueError, match="A maximum dimension of 2"):
            detector.navigation_shape = (1, 2, 3)

    @pytest.mark.parametrize(
        "shape, desired_x_range, desired_y_range",
        [
            ((60, 60), [-0.833828, 1.146762], [-0.436918, 1.543672]),
            ((510, 510), [-0.833828, 1.146762], [-0.436918, 1.543672]),
            ((1, 1), [-0.833828, 1.146762], [-0.436918, 1.543672]),
            ((480, 640), [-1.111771, 1.529016], [-0.436918, 1.543672]),
        ],
    )
    def test_gnomonic_range(self, pc1, shape, desired_x_range, desired_y_range):
        """Gnomonic x/y range, x depends on aspect ratio."""
        detector = kp.detectors.EBSDDetector(shape=shape, pc=pc1)
        assert np.allclose(detector.x_range, desired_x_range, atol=1e-6)
        assert np.allclose(detector.y_range, desired_y_range, atol=1e-6)

    @pytest.mark.parametrize(
        "shape, desired_x_scale, desired_y_scale",
        [
            ((60, 60), 0.033569, 0.033569),
            ((510, 510), 0.003891, 0.003891),
            ((1, 1), 1.980590, 1.980590),
            ((480, 640), 0.004133, 0.004135),
        ],
    )
    def test_gnomonic_scale(self, pc1, shape, desired_x_scale, desired_y_scale):
        """Gnomonic (x, y) scale."""
        detector = kp.detectors.EBSDDetector(shape=shape, pc=pc1)
        assert np.allclose(detector.x_scale, desired_x_scale, atol=1e-6)
        assert np.allclose(detector.y_scale, desired_y_scale, atol=1e-6)

    @pytest.mark.parametrize(
        "tilt, azimuthal, twist, sample_tilt, expected_rotation",
        [
            (0, 0, 0, 90.0, [0.7071, 0.0, 0.0, -0.7071]),
            (0, 0, 0, 70.0, [0.6964, -0.1228, -0.1228, -0.6964]),
            (8.3, 4.7, -1.02, 70.0, [0.6861, -0.2021, -0.1428, -0.6841]),
        ],
    )
    def test_sample_to_detector(
        self, tilt, azimuthal, twist, sample_tilt, expected_rotation
    ):
        det = kp.detectors.EBSDDetector(
            tilt=tilt, azimuthal=azimuthal, twist=twist, sample_tilt=sample_tilt
        )
        rot_sample_to_det = det.sample_to_detector
        assert isinstance(rot_sample_to_det, oqu.Rotation)
        assert np.allclose(rot_sample_to_det.data, expected_rotation, atol=1e-4)

    def test_sample_to_detector_numpy_comparison(self):
        det = kp.detectors.EBSDDetector()
        v = ove.Vector3d.random(4)

        R_s2d = det.sample_to_detector
        om_s2d = R_s2d.to_matrix().squeeze()
        assert np.allclose((R_s2d * v).data, (om_s2d @ v.data.T).T, atol=1e-4)
        assert np.allclose((R_s2d * v).data, np.dot(om_s2d, v.data.T).T, atol=1e-4)

        R_d2s = ~R_s2d
        om_d2s = R_d2s.to_matrix().squeeze()
        assert np.allclose((R_d2s * v).data, (om_d2s @ v.data.T).T, atol=1e-4)
        assert np.allclose((R_d2s * v).data, np.dot(om_d2s, v.data.T).T, atol=1e-4)

    @pytest.mark.parametrize("sample_tilt", [0.0, 70.0])
    def test_sample_to_detector_azimuthal_about_detector_y(self, sample_tilt):
        det = kp.detectors.EBSDDetector(
            sample_tilt=sample_tilt,
            tilt=40.0,
            azimuthal=0.0,
            twist=0.0,
        )
        y0 = det.sample_to_detector.to_matrix().squeeze()[1]

        for azimuthal in [20.0, -40.0]:
            det.azimuthal = azimuthal
            y = det.sample_to_detector.to_matrix().squeeze()[1]
            assert np.allclose(y, y0, atol=1e-8)

    @pytest.mark.parametrize(
        "coords, detector_index, desired_coords",
        [
            (
                np.array([[12.7, 36.2]]),
                None,
                np.array(
                    [
                        [[[0.00664684, 0.36223463]], [[-0.00304659, 0.357628]]],
                        [[[0.00973462, 0.36432453]], [[0.00567801, 0.35219232]]],
                    ]
                ),
            ),
            (
                np.array([[12.7, 36.2], [43.7, 2.5], [27.7, 8.2]]),
                (0, 1),
                np.array(
                    [
                        [-0.00304659, 0.35762801],
                        [-1.03422601, -0.76336381],
                        [-0.50200438, -0.57375985],
                    ]
                ),
            ),
        ],
    )
    def test_coordinate_conversions(self, coords, detector_index, desired_coords):
        """Converting from pixel to gnomonic coords and back."""
        pc = np.array(
            [
                [
                    [0.4214844, 0.21500351, 0.50201974],
                    [0.42414583, 0.21014019, 0.50104439],
                ],
                [
                    [0.42088203, 0.2165417, 0.50079336],
                    [0.42725023, 0.21450546, 0.49996293],
                ],
            ]
        )
        det = kp.detectors.EBSDDetector(shape=(60, 60), pc=pc)
        cds_out = det.to_gnomonic_coords(coords, detector_index)
        cds_back = det.to_pixel_coords(cds_out, detector_index)
        assert np.allclose(cds_out, desired_coords)
        assert np.allclose(cds_back[..., :], coords[..., :])

    @pytest.mark.parametrize(
        "method_name, coords",
        [
            ("to_gnomonic_coords", np.array([[12.7, 36.2]])),
            ("to_pixel_coords", np.array([[0.01, 0.35]])),
        ],
    )
    def test_coordinate_conversions_detector_index_validation(
        self, method_name, coords
    ):
        pc_1d = np.array(
            [
                [0.4214844, 0.21500351, 0.50201974],
                [0.42414583, 0.21014019, 0.50104439],
                [0.42088203, 0.2165417, 0.50079336],
            ]
        )
        pc_2d = np.array(
            [
                [
                    [0.4214844, 0.21500351, 0.50201974],
                    [0.42414583, 0.21014019, 0.50104439],
                ],
                [
                    [0.42088203, 0.2165417, 0.50079336],
                    [0.42725023, 0.21450546, 0.49996293],
                ],
            ]
        )

        det_1d = kp.detectors.EBSDDetector(shape=(60, 60), pc=pc_1d)
        method_1d = getattr(det_1d, method_name)
        assert np.allclose(method_1d(coords, 1), method_1d(coords, (1,)))

        with pytest.raises(ValueError, match="navigation dimension is 2"):
            method_1d(coords, (0, 1))

        with pytest.raises(TypeError, match="single integer"):
            method_1d(coords, (slice(None),))

        det_2d = kp.detectors.EBSDDetector(shape=(60, 60), pc=pc_2d)
        method_2d = getattr(det_2d, method_name)

        with pytest.raises(ValueError, match="navigation dimension is 1"):
            method_2d(coords, 0)

        with pytest.raises(TypeError, match="Sample position"):
            method_2d(coords, [0, 1])

        with pytest.raises(ValueError, match="length 1 or 2"):
            method_2d(coords, (0, 1, 2))

        cds_out_multi = method_2d(coords, (slice(None), 1))
        assert cds_out_multi.shape == (2,) + coords.shape
        assert np.allclose(cds_out_multi[0], method_2d(coords, (0, 1)))
        assert np.allclose(cds_out_multi[1], method_2d(coords, (1, 1)))

    @pytest.mark.parametrize(
        "shape, desired_shapes",
        [
            (
                (1,),  # PC
                [
                    (4,),  # extent
                    (1,),  # x_min
                    (1,),  # y_min
                    (1, 2),  # x_range
                    (1, 2),  # y_range
                    (1,),  # x_scale
                    (1,),  # y_scale
                    (1, 4),  # extent_gnomonic
                ],
            ),
            (
                (10,),
                [
                    (4,),
                    (10,),
                    (10,),
                    (10, 2),
                    (10, 2),
                    (10,),
                    (10,),
                    (10, 4),
                ],
            ),
            (
                (10, 10),
                [
                    (4,),
                    (10, 10),
                    (10, 10),
                    (10, 10, 2),
                    (10, 10, 2),
                    (10, 10),
                    (10, 10),
                    (10, 10, 4),
                ],
            ),
            (
                (1, 10),
                [
                    (4,),
                    (1, 10),
                    (1, 10),
                    (1, 10, 2),
                    (1, 10, 2),
                    (1, 10),
                    (1, 10),
                    (1, 10, 4),
                ],
            ),
            (
                (10, 1),
                [
                    (4,),
                    (10, 1),
                    (10, 1),
                    (10, 1, 2),
                    (10, 1, 2),
                    (10, 1),
                    (10, 1),
                    (10, 1, 4),
                ],
            ),
        ],
    )
    def test_property_shapes(self, shape, desired_shapes):
        """Expected property shapes when varying navigation shape."""
        det = kp.detectors.EBSDDetector(pc=np.ones(shape + (3,)))
        assert det.bounds.shape == desired_shapes[0]
        assert det.x_min.shape == desired_shapes[1]
        assert det.y_min.shape == desired_shapes[2]
        assert det.x_range.shape == desired_shapes[3]
        assert det.y_range.shape == desired_shapes[4]
        assert det.x_scale.shape == desired_shapes[5]
        assert det.y_scale.shape == desired_shapes[6]
        assert det.gnomonic_bounds.shape == desired_shapes[7]

    def test_crop(self):
        det = kp.detectors.EBSDDetector((6, 6), pc=[3 / 6, 2 / 6, 0.5])
        det2 = det.crop((1, 5, 2, 6))
        assert det2.shape == (4, 4)
        assert np.allclose(det2.pc, [0.25, 0.25, 0.75])

        # "Real" example
        s = kp.data.nickel_ebsd_small()
        det3 = s.detector
        det4 = det3.crop((-10, 50, 20, 70))  # (0, 50, 20, 60)
        assert det4.shape == (50, 40)

    def test_crop_raises(self):
        det = kp.detectors.EBSDDetector((6, 6), pc=[3 / 6, 2 / 6, 0.5])
        with pytest.raises(ValueError):
            _ = det.crop((1.0, 5, 2, 6))
        with pytest.raises(ValueError):
            _ = det.crop((5, 1, 2, 6))
        with pytest.raises(ValueError):
            _ = det.crop((1, 5, 6, 2))

    def test_crop_simulated(self):
        s = kp.data.nickel_ebsd_small()

        det2 = s.detector.crop((0, 50, 20, 60))

        mp = kp.data.nickel_ebsd_master_pattern_small(projection="lambert")
        rot = s.xmap.rotations.reshape(*s.xmap.shape)

        kwds = {"compute": True, "dtype_out": "uint8"}
        sim1 = mp.get_patterns(rot, s.detector, **kwds)
        sim2 = mp.get_patterns(rot, det2, **kwds)

        assert np.allclose(sim2.data, sim1.isig[20:60, :50].data)

    def test_set_detector_binning_raises(self):
        det = kp.detectors.EBSDDetector()
        with pytest.raises(ValueError, match=r"Invalid binning \(3, 4\). Must be an "):
            det.binning = (3, 4)

    def test_invalid_values_raises(self):
        det = kp.detectors.EBSDDetector()

        with pytest.raises(ValueError, match="Invalid sample tilt "):
            det.sample_tilt = "a"

        with pytest.raises(ValueError, match="Invalid detector tilt "):
            det.tilt = "a"

        with pytest.raises(ValueError, match="Invalid azimuthal "):
            det.azimuthal = "a"

        with pytest.raises(ValueError, match="Invalid twist "):
            det.twist = "a"

        with pytest.raises(ValueError, match="Invalid pixel size "):
            det.px_size = "a"


class TestEBSDDetectorPCConventions:
    @pytest.mark.parametrize(
        "shape, pc, px_size, binning, version, desired_pc",
        [
            (
                (60, 60),
                [3.4848, 114.2016, 15767.7],
                59.2,
                8,
                4,
                [0.50726, 0.26208, 0.55488809122],
            ),
            (
                (60, 60),
                [-3.4848, 114.2016, 15767.7],
                59.2,
                8,
                5,
                [0.50726, 0.26208, 0.55489],
            ),
            (
                (61, 61),
                [-10.6320, 145.5187, 19918.9],
                59.2,
                8,
                5,
                [0.52178688525, 0.20180594262, 0.68948341272],
            ),
            (
                (61, 61),
                [10.632, 145.5187, 19918.9],
                59.2,
                8,
                5,
                [0.47821, 0.20181, 0.68948],
            ),
            (
                (80, 60),
                [-0.55, -13.00, 16075.2],
                50,
                6,
                5,
                [0.50153, 0.52708, 0.66980],
            ),
            (
                (80, 60),
                [0.55, -13.00, 16075.2],
                50,
                6,
                4,
                [0.50153, 0.52708, 0.66980],
            ),
            ((480, 640), [0, 0, 15000], 50, 1, 5, [0.5, 0.5, 0.625]),
        ],
    )
    def test_set_pc_from_emsoft(self, shape, pc, px_size, binning, version, desired_pc):
        """PC EMsoft -> Bruker -> EMsoft, also checking to_tsl(),
        to_oxford(), and to_bruker().
        """
        det = kp.detectors.EBSDDetector(
            shape=shape,
            pc=pc,
            px_size=px_size,
            binning=binning,
            convention=f"emsoft{version}",
        )

        assert np.allclose(det.pc, desired_pc, atol=1e-5)
        assert np.allclose(det.pc_emsoft(version=version), pc, atol=1e-5)
        assert np.allclose(det.pc_bruker(), desired_pc, atol=1e-5)

        # EDAX
        pc_tsl = deepcopy(det.pc)
        pc_tsl[..., 1] = 1 - pc_tsl[..., 1]
        pc_tsl[..., 2] /= min([det.nrows, det.ncols]) / det.nrows
        assert np.allclose(det.pc_tsl(), pc_tsl, atol=1e-5)

        # Oxford
        pc_oxford = deepcopy(det.pc)
        pc_oxford[..., 1] = 1 - pc_oxford[..., 1]
        pc_oxford[..., 1:] /= det.aspect_ratio
        assert np.allclose(det.pc_oxford(), pc_oxford, atol=1e-5)

    def test_set_pc_from_emsoft_no_version(self):
        """PC EMsoft -> Bruker, no EMsoft version specified gives v5."""
        assert np.allclose(
            kp.detectors.EBSDDetector(
                shape=(60, 60),
                pc=[3.4848, 114.2016, 15767.7],
                px_size=59.2,
                binning=8,
                convention="emsoft",
            ).pc,
            [0.49274, 0.26208, 0.55489],
            atol=1e-5,
        )

    @pytest.mark.parametrize(
        "shape, pc, convention, desired_pc",
        [
            ((60, 60), [0.35, 1, 0.65], "tsl", [0.35, 0, 0.65]),
            ((60, 80), [0.35, 1, 0.65], "tsl", [0.35, 0, 0.65]),
            ((60, 60), [0.1, 0.2, 0.3], "amatek", [0.1, 0.8, 0.3]),
            ((60, 80), [0.1, 0.2, 0.3], "amatek", [0.1, 0.8, 0.3]),
            ((60, 60), [0.6, 0.6, 0.6], "edax", [0.6, 0.4, 0.6]),
            ((60, 80), [0.6, 0.6, 0.6], "edax", [0.6, 0.4, 0.6]),
        ],
    )
    def test_set_pc_from_tsl(self, shape, pc, convention, desired_pc):
        """PC TSL -> Bruker -> TSL."""
        det = kp.detectors.EBSDDetector(shape=shape, pc=pc, convention=convention)
        assert np.allclose(det.pc, desired_pc, atol=1e-2)
        assert np.allclose(det.pc_tsl(), pc, atol=1e-3)
        assert np.allclose(
            kp.detectors.EBSDDetector(pc=det.pc_tsl(), convention="tsl").pc_tsl(),
            pc,
            atol=1e-2,
        )

    @pytest.mark.parametrize(
        "shape, pc, desired_pc",
        [
            ((60, 60), [0.25, 0, 0.75], [0.25, 1, 0.75]),
            ((60, 80), [0.25, 0, 0.75], [0.25, 1, 1]),
        ],
    )
    def test_set_pc_from_oxford(self, shape, pc, desired_pc):
        """PC Oxford -> Bruker -> Oxford."""
        det = kp.detectors.EBSDDetector(shape=shape, pc=pc, convention="oxford")
        assert np.allclose(det.pc, desired_pc, atol=1e-2)
        assert np.allclose(det.pc_oxford(), pc, atol=1e-3)
        assert np.allclose(
            kp.detectors.EBSDDetector(
                pc=det.pc_oxford(), convention="oxford"
            ).pc_oxford(),
            pc,
            atol=1e-2,
        )

    @pytest.mark.parametrize(
        "pc, convention",
        [
            ([0.1, 0.2, 0.3], "Bruker"),
            ([0.6, 0.6, 0.6], "bruker"),
        ],
    )
    def test_set_pc_from_bruker(self, pc, convention):
        """PC Bruker returns Bruker PC, which is the default."""
        det = kp.detectors.EBSDDetector(pc=pc, convention=convention)
        assert np.allclose(det.pc, pc)

    def test_set_pc_convention_raises(self, pc1):
        """Wrong convention raises."""
        with pytest.raises(ValueError, match="Invalid projection/pattern center "):
            kp.detectors.EBSDDetector(pc=pc1, convention="nordif")

    # TODO: Remove test after 0.13 has been released
    def test_convention_none_warns(self):
        pc0 = [0.5, 0.4, 0.3]
        with pytest.warns(UserWarning, match="Passing None"):
            det = kp.detectors.EBSDDetector(pc=pc0, convention=None)
        assert np.allclose(det.pc, pc0)


class TestEstimateTilts:
    det0 = kp.detectors.EBSDDetector(
        shape=(480, 480),
        pc=(0.5, 0.3, 0.5),
        sample_tilt=70,
        tilt=0,
        px_size=70,
    )

    def test_estimate_xtilt_raises(self):
        with pytest.raises(ValueError, match="Estimation requires more than one "):
            _ = self.det0.estimate_xtilt()

    def test_estimate_xtilt(self):
        det = self.det0.extrapolate_pc(
            pc_indices=[0, 0],
            navigation_shape=(15, 20),
            step_sizes=(50, 50),
        )

        xtilt, fig = det.estimate_xtilt(degrees=True, return_figure=True)
        assert np.isclose(xtilt, 90 - self.det0.sample_tilt + self.det0.tilt)
        assert not any(
            ["Outliers" in t.get_text() for t in fig.axes[0].get_legend().texts]
        )

        xtilt, is_outliers = det.estimate_xtilt(return_outliers=True)
        assert isinstance(is_outliers, np.ndarray)
        assert is_outliers.sum() == 0

        plt.close("all")

    def test_estimate_xtilt_outliers(self):
        det = self.det0.extrapolate_pc(
            pc_indices=[0, 0],
            navigation_shape=(15, 20),
            step_sizes=(50, 50),
        )
        det.pc[0, 0] = (0.5, 0.5, 0.5)

        _, is_outliers, fig = det.estimate_xtilt(
            return_outliers=True, return_figure=True
        )
        assert isinstance(is_outliers, np.ndarray)
        assert is_outliers.shape == det.navigation_shape
        assert is_outliers.sum() == 1
        assert np.allclose(np.where(is_outliers)[0], [0, 0])
        assert any(["Outliers" in t.get_text() for t in fig.axes[0].get_legend().texts])

    def test_estimate_xtilt_ztilt(self):
        det1 = self.det0.extrapolate_pc(
            pc_indices=[0, 0],
            navigation_shape=(15, 20),
            step_sizes=(1, 1),
        )
        xtilt, ztilt = det1.estimate_xtilt_ztilt(degrees=True)
        assert np.isclose(xtilt, 20)
        assert np.isclose(ztilt, 0)

        # Add outliers
        det2 = det1.deepcopy()
        outlier_idx = [[0, 0], [0, 10]]
        det2.pc[tuple(outlier_idx)] = (0.5, 0.4, 0.5)

        np.random.seed(42)

        xtilt2, ztilt2 = det2.estimate_xtilt_ztilt(degrees=True)
        assert np.isclose(xtilt2, 0.169, atol=1e-3)
        assert np.isclose(ztilt2, -74.339, atol=1e-3)

        is_outlier = np.ravel_multi_index(outlier_idx, det1.navigation_shape)
        xtilt3, ztilt3 = det2.estimate_xtilt_ztilt(degrees=True, is_outlier=is_outlier)
        assert np.isclose(xtilt3, 20)
        assert np.isclose(ztilt3, 0)

    def test_estimate_xtilt_ztilt_raises(self):
        with pytest.raises(ValueError, match="Estimation requires more than one "):
            _ = self.det0.estimate_xtilt_ztilt()


class TestExtrapolatePC:
    det0 = kp.detectors.EBSDDetector(
        shape=(240, 240),
        pc=(0.5, 0.3, 0.5),
        sample_tilt=70,
        tilt=0,
        px_size=70,
        binning=2,
    )

    def test_extrapolate_pc(self):
        det = self.det0.extrapolate_pc(
            pc_indices=[7, 15],
            navigation_shape=(15, 31),
            step_sizes=(50, 50),
        )
        assert det.navigation_shape == (15, 31)
        assert np.allclose(
            [
                self.det0.nrows,
                self.det0.ncols,
                self.det0.sample_tilt,
                self.det0.tilt,
                self.det0.px_size,
                self.det0._binning,
                self.det0.azimuthal,
            ],
            [
                det.nrows,
                det.ncols,
                det.sample_tilt,
                det.tilt,
                det.px_size,
                det.binning,
                det.azimuthal,
            ],
        )
        assert np.allclose(det.pc_average, self.det0.pc)
        assert np.allclose(det.pc_flattened.min(0), [0.4777, 0.2902, 0.4964], atol=1e-4)
        assert np.allclose(det.pc_flattened.max(0), [0.5223, 0.3098, 0.5036], atol=1e-4)

    def test_extrapolate_pc_multiple_indices(self):
        det1 = self.det0.deepcopy()
        # Specify PC values in four corners of the map, visualized here
        # as they would show up in the (PCx, PCy) scatter plot
        # fmt: off
        det1.pc = [
            [0.5, 0.3, 0.5],                [0.3, 0.3, 0.5],




            [0.5, 0.2, 0.6],                [0.3, 0.2, 0.6],
        ]
        # fmt: on

        det2 = det1.extrapolate_pc(
            pc_indices=[[0, 0], [0, 10], [20, 0], [20, 10]],
            navigation_shape=(11, 21),
            step_sizes=(11, 11),
        )
        assert np.allclose(det1.pc_average, det2.pc_average, atol=1e-2)

        det3 = det1.extrapolate_pc(
            pc_indices=[[0, 0], [0, 10], [20, 0], [20, 10]],
            navigation_shape=(11, 21),
            step_sizes=(5, 5),
            shape=(60, 60),
            binning=8,
            px_size=70 * 8,
        )
        assert np.allclose(det1.pc_average, det3.pc_average, atol=1e-2)
        assert det3.shape == (60, 60)
        assert det3._binning == 8
        assert det3.px_size == 70 * 8

    def test_extrapolate_pc_outliers(self):
        det1 = self.det0.deepcopy()
        det1.pc = [[0.5, 0.3, 0.5], [0.3, 0.3, 0.5], [0.5, 0.2, 0.6], [0.3, 0.2, 0.6]]
        pc_indices = np.array([[0, 0], [0, 10], [20, 0], [20, 10]]).T
        det2 = det1.extrapolate_pc(
            pc_indices=pc_indices,
            navigation_shape=(11, 21),
            step_sizes=(11, 11),
            is_outlier=[True, False, False, False],
        )
        assert np.allclose(det2.pc_average, [0.366, 0.233, 0.567], atol=1e-3)


class TestFitPC:
    def setup_method(self):
        """Create a plane of PCs with a known mean, add some noise,
        extract selected patterns, and try to 'reconstruct' the plane
        by fitting.
        """
        det0 = kp.detectors.EBSDDetector(
            shape=(240, 240),
            pc=(0.5, 0.3, 0.5),
            sample_tilt=70,
        )

        det = det0.extrapolate_pc(
            pc_indices=[7, 15], navigation_shape=(15, 31), step_sizes=(50, 50)
        )

        # Add noise
        rng = np.random.default_rng(42)
        v = 0.005
        det.pcy += rng.uniform(-v, v, det.navigation_size).reshape(det.navigation_shape)
        det.pcz += rng.uniform(-v, v, det.navigation_size).reshape(det.navigation_shape)

        self.det0 = det0
        self.det = det
        self.map_indices = np.stack(np.indices(det.navigation_shape))

    def test_fit_pc_corner_patterns(self):
        """Test projective fit."""
        pc_indices = [[0, 0, 14, 14], [0, 30, 0, 30]]
        det2 = self.det.deepcopy()
        det2.pc = det2.pc[tuple(pc_indices)].reshape((2, 2, 3))
        pc_indices = np.array(pc_indices).reshape((2, 2, 2))
        det_fit, fig = det2.fit_pc(
            pc_indices=pc_indices, map_indices=self.map_indices, return_figure=True
        )
        assert np.all(abs(det_fit.pc_flattened - self.det.pc_flattened).max(0) < 0.009)

        # We have a plane in the 3D plot
        assert isinstance(fig.axes[3].collections[2], mcollections.PolyCollection)

        plt.close("all")

    @pytest.mark.parametrize(
        "grid_shape, max_error", [((3, 3), 0.009), ((5, 5), 0.0091), ((7, 7), 0.0064)]
    )
    def test_fit_pc_grid_patterns_33(self, grid_shape, max_error):
        """Test projective fit."""
        pc_indices = kp.signals.util.grid_indices(grid_shape, self.det.navigation_shape)
        pc_indices = pc_indices.reshape(2, -1)
        det2 = self.det.deepcopy()
        det2.pc = det2.pc[tuple(pc_indices)]
        det_fit = det2.fit_pc(
            pc_indices=pc_indices, map_indices=self.map_indices, plot=False
        )
        assert np.all(
            abs(det_fit.pc_flattened - self.det.pc_flattened).max(0) < max_error
        )

    def test_fit_pc_affine_outliers(self):
        grid_shape = (7, 7)
        pc_indices = kp.signals.util.grid_indices(grid_shape, self.det.navigation_shape)
        pc_indices = pc_indices.reshape(2, -1)
        det2 = self.det.deepcopy()
        det2.pc = det2.pc[tuple(pc_indices)]

        # Add outliers to extracted PCs
        det2.pc = np.append(det2.pc, [[0.55, 0.15, 0.55], [0.6, 0.10, 0.6]], axis=0)
        is_outlier = np.zeros(det2.navigation_size, dtype=bool)
        is_outlier[[-2, -1]] = True
        pc_indices = np.append(pc_indices, [[1, 1], [1, 2]], axis=1)

        # Bad fit
        det_fit1 = det2.fit_pc(
            pc_indices=pc_indices,
            map_indices=self.map_indices,
            transformation="affine",
        )
        assert np.allclose(
            abs(det_fit1.pc_flattened - self.det.pc_flattened).max(0),
            [0.70, 0.35, 0.13],
            atol=1e-2,
        )

        # Good fit
        det_fit2, fig = det2.fit_pc(
            pc_indices=pc_indices,
            map_indices=self.map_indices,
            transformation="affine",
            is_outlier=is_outlier,
            return_figure=True,
        )
        assert np.all(abs(det_fit2.pc_flattened - self.det.pc_flattened).max(0) < 0.009)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4

        plt.close("all")

    def test_fit_pc_raises(self):
        grid_shape = (7, 5)
        pc_indices = kp.signals.util.grid_indices(grid_shape, self.det.navigation_shape)
        pc_indices = pc_indices.reshape(2, -1)
        det2 = self.det.deepcopy()
        det2.pc = det2.pc[tuple(pc_indices)]

        with pytest.raises(ValueError, match="Fitting requires multiple projection "):
            _ = self.det0.fit_pc(pc_indices, self.map_indices)

        with pytest.raises(ValueError, match=r"`pc_indices` array shape \(2, 7, 5\) "):
            _ = det2.fit_pc(pc_indices.reshape((2,) + grid_shape), self.map_indices)

        with pytest.raises(ValueError, match=r"`map_indices` array shape \(930,\) "):
            _ = det2.fit_pc(pc_indices, self.map_indices.ravel())

        det2.pc = det2.pc.reshape(grid_shape + (3,))
        with pytest.raises(ValueError, match=r"`pc_indices` array shape \(2, 35\) "):
            _ = det2.fit_pc(pc_indices, self.map_indices)
        det2.pc = det2.pc.reshape(-1, 3)

        is_outlier = np.zeros(det2.navigation_size - 1, dtype=bool)
        with pytest.raises(ValueError, match="`is_outlier` must be a boolean array of"):
            _ = det2.fit_pc(pc_indices, self.map_indices, is_outlier=is_outlier)


class TestGetIndexer:
    def setup_method(self):
        det0 = kp.detectors.EBSDDetector(
            shape=(10, 12),
            pc=(0.5, 0.3, 0.5),
            sample_tilt=69.5,
            tilt=10.5,
        )
        det = det0.extrapolate_pc(
            pc_indices=[0, 0], navigation_shape=(2, 3), step_sizes=(1, 1)
        )
        self.det = det

    @skipif_pyebsdindex_installed
    def test_get_indexer_raises(self):
        pl = PhaseList(names=["al", "si"], space_groups=[225, 227])
        with pytest.raises(ImportError, match="requires that 'pyebsdindex'"):
            _ = self.det.get_indexer(pl)

    @skipif_pyebsdindex_not_installed
    def test_get_indexer_invalid_phase_lists(self):
        # Not all phases have space groups
        pl = PhaseList(names=["a", "b"], point_groups=["m-3m", "432"])
        pl["a"].space_group = 225
        with pytest.raises(ValueError, match="Space group for each phase must be set,"):
            _ = self.det.get_indexer(pl)

    @skipif_pyebsdindex_not_installed
    def test_get_indexer(self):
        # fmt: off
        #               -1  2/m  222   -3   -3m   4/m   4/mmm   6/m  6/mmm    m-3  m-3m
        space_groups = [ 1,  15,  74,  75,  142,  143,    167,  168,   194,   195,  207]
        laue_codes =   [ 1,   2,  22,   4,   42,    3,     32,    6,    62,    23,   43]
        # fmt: on
        names = "abcdefghijk"

        pl = PhaseList(names=list(names), space_groups=space_groups)
        indexer = self.det.get_indexer(pl, nBands=6)
        assert indexer.vendor == "KIKUCHIPY"
        assert np.isclose(indexer.sampleTilt, self.det.sample_tilt)
        assert np.isclose(indexer.camElev, self.det.tilt)
        assert tuple(indexer.bandDetectPlan.patDim) == self.det.shape
        assert indexer.bandDetectPlan.nBands == 6
        assert np.allclose(indexer.PC, self.det.pc_flattened)
        for phase, sg, code in zip(indexer.phaselist, pl.space_groups, laue_codes):
            assert phase.spacegroup == sg.number
            assert phase.lauecode == code


class TestSaveLoadDetector:
    @pytest.mark.parametrize(
        "nav_shape, shape, convention, sample_tilt, tilt, px_size, binning, azimuthal, "
        "twist",
        [
            ((3, 4), (10, 20), "bruker", 70, 0, 70, 1, 0.1, 0),
            ((1, 5), (5, 5), "tsl", 69.5, 3.14, 57.2, 2, 3.7, 0.003),
            ((4, 3), (6, 7), "emsoft", -69.5, -3.14, 57.2, 2, -3.7, -1.23),
            ((3, 2), (5, 7), "oxford", 71.3, 1.2, 90.3, 3, 0.1, 0.0465),
        ],
    )
    def test_save_load_detector(
        self,
        tmp_path,
        nav_shape,
        shape,
        convention,
        sample_tilt,
        tilt,
        px_size,
        binning,
        azimuthal,
        twist,
    ):
        det0 = kp.detectors.EBSDDetector(
            shape=shape,
            pc=(0.4, 0.3, 0.6),
            sample_tilt=sample_tilt,
            tilt=tilt,
            px_size=px_size,
            binning=binning,
            azimuthal=azimuthal,
            twist=twist,
            convention=convention,
        )

        # Warns about unused non-zero azimuthal and twist angles
        with pytest.warns(UserWarning, match="azimuthal"):
            det1 = det0.extrapolate_pc(
                pc_indices=[0, 0],
                navigation_shape=nav_shape,
                step_sizes=(2, 2),
            )
        if any(i == 1 for i in nav_shape):
            det1.pc = det1.pc.reshape(-1, 3)
        fname = tmp_path / "det_temp.txt"
        det1.save(fname, convention=convention)

        det2 = kp.detectors.EBSDDetector.load(fname)

        assert det2.shape == det1.shape
        assert np.allclose(det2.pc, det1.pc, atol=1e-7)
        assert np.isclose(det2.sample_tilt, det1.sample_tilt)
        assert np.isclose(det2.tilt, det1.tilt)
        assert np.isclose(det2.px_size, det1.px_size)
        assert det2._binning == det1._binning
        assert np.isclose(det2.azimuthal, det1.azimuthal)
        assert np.isclose(det2.twist, det1.twist)

    def test_save_detector_raises(self, tmp_path):
        det = kp.detectors.EBSDDetector()
        with pytest.raises(ValueError, match="Invalid projection/pattern center "):
            det.save(tmp_path / "det_temp.txt", convention="EMsofts")
