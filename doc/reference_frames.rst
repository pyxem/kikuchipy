================
Reference frames
================

:numref:`fig-detector-sample-geometry` and :numref:`fig-detector-coordinates`
show EBSD reference frames used in kikuchipy, all of which are right handed.
They are based on definitions presented in [Britton2016]_ and the accompanying
supplementary material.

.. _fig-detector-sample-geometry:

.. figure:: _static/image/reference_frames/detector_sample_geometry.png
    :align: center
    :width: 100%

    **(a)** An EBSD experiment showing the orientation of the crystal reference
    frame, :math:`x_{euler}-y_{euler}-z_{euler}`, attached to the sample. The
    RD-TD-ND crystal reference frame used by EDAX TSL is shown for reference.
    An EBSD pattern on the detector screen is viewed from behind the screen
    towards the sample. **(b)** How the EBSD map appears within the data
    collection software, with the crystal reference frame and the scanning
    reference frame, :math:`x_{scan}-y_{scan}-z_{scan}`, attached. **(c)** The
    relationship between the crystal reference frame and the detector reference
    frame, :math:`x_{detector}-y_{detector}-z_{detector}`, with the projection
    center highlighted. The detector tilt :math:`\theta` and sample tilt
    :math:`\sigma` are also shown.

.. _fig-detector-coordinates:

.. figure:: _static/image/reference_frames/detector_gnomonic_coordinates.jpg
    :align: center
    :width: 100%

    The EBSD pattern in :numref:`fig-detector-sample-geometry` (a) as viewed
    from behind the screen towards the sample (left), with the detector
    reference frame the same as in (c) with its origin (0, 0) in the upper left
    pixel. The detector pixels' gnomonic coordinates can be described with a
    calibrated projection center (PC) (right), with the gnomonic reference frame
    origin (0, 0) in (PC\ :sub:`x`\, PC\ :sub:`y`\). The circles indicate the
    angular distance from the PC in steps of :math:`10^{\circ}`.
