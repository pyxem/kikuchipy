import numpy as np


def _get_nav_shape(p):
    return {2: (), 3: (p.shape[0],), 4: (p.shape[:2])}[p.ndim]


def _get_sig_shape(p):
    return p.shape[-2:]


def _get_number_of_templates(t):
    if t.ndim == 3:
        return t.shape[0]
    else:
        return 1
