============
Feature maps
============

This section details methods for extracting quantitative information from
pattern intensities, called feature maps for lack of a better description.

.. _image-quality:

Image quality
=============

The image quality metric presented by Krieger Lassen [Lassen1994]_ can be
calculated for an :class:`~kikuchipy.signals.ebsd.EBSD` object with
:meth:`~kikuchipy.signals.ebsd.EBSD.get_image_quality`, or, for a single pattern
(:class:`numpy.ndarray`), with
:func:`~kikuchipy.util.pattern.get_image_quality`.

Whether to normalize patterns to a mean of zero and standard
deviation of 1 before calculating the image quality (default
is True).
