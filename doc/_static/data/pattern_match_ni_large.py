from orix import sampling, io
import kikuchipy as kp


s_large = kp.data.nickel_ebsd_large()
s_large.remove_static_background()
s_large.remove_dynamic_background()
s_large.average_neighbour_patterns(window="gaussian", std=2)

mp = kp.data.nickel_ebsd_master_pattern_small(projection="lambert")
r = sampling.get_sample_fundamental(
    resolution=4, space_group=mp.phase.space_group.number
)
detector = kp.detectors.EBSDDetector(
    shape=s_large.axes_manager.signal_shape[::-1],
    pc=[0.421, 0.7794, 0.5049],
    convention="tsl",
    sample_tilt=70,
)
sim = mp.get_patterns(rotations=r, detector=detector, energy=20, compute=False)

xmap = s_large.dictionary_indexing(sim, keep_n=1, metric="ncc")
io.save("/home/hakon/ni_large.h5", xmap)
