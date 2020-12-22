import kikuchipy as kp
from orix import sampling, io


s_large = kp.data.nickel_ebsd_large()

s_large.remove_static_background()
s_large.remove_dynamic_background()
s_large.average_neighbour_patterns(window="gaussian", std=2)

mp = kp.data.nickel_ebsd_master_pattern_small(projection="lambert")

r = sampling.get_sample_fundamental(
    resolution=5, space_group=mp.phase.space_group.number
)

detector = kp.detectors.EBSDDetector(
    shape=s_large.axes_manager.signal_shape[::-1],
    pc=[0.421, 0.7794, 0.5049],
    convention="tsl",
    sample_tilt=70,
)

sim = mp.get_patterns(
    rotations=r, detector=detector, energy=20, dtype_out=np.uint8, compute=True
)

xmap = s_large.match_patterns(sim, keep_n=1, n_slices=10, metric="ncc")

io.save("/home/hakon/ni_large.h5", xmap)
