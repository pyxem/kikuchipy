================
Reference frames
================

All definitions are right handed.

.. warning::

    Use the detector class with caution, as it is not fully tested against all
    vendor projection center formats.

.. _detector:

Detector
========

Since HyperSpy's ``axes_manager`` does not allow the signal offset and scale
to vary in navigation space (per pattern), all EBSD detector information per
signal is assumed to be stored in the :class:`~kikuchipy.detectors.EBSDDetector`
class.

Also, since HyperSpy's vertical axis increases downwards, this convention is
adopted for the EBSD detector class.

.. figure:: _static/image/reference_frames/detector.jpg
    :align: center
    :width: 50%

    Frame of reference for a diffraction pattern, with the gnomonic reference
    frame :math:`(y_g, x_g)` origin located at the projection/pattern center
    (0, 0), while the detector reference frame :math:`(y_d, x_d)` origin (0, 0)
    is located at the upper left of the detector.
