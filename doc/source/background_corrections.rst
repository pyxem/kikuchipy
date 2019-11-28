======================
Background corrections
======================

The raw EBSD signal can be empirically evaluated as a superposition of a Kikuchi
diffraction pattern and a smooth background intensity. For pattern indexing, the
latter intensity is undesirable, while for so-called :doc:`virtual forward
scatter detector (VFSD) imaging <virtual_forward_scatter_detector>`, this
intensity can reveal important topographical, compositional or diffraction
contrast. This section details methods to enhance the Kikuchi diffraction
pattern.

.. _static-background-correction:

Static background correction
============================

The slowly varying diffuse background in raw patterns can be removed by either
subtracting or dividing by a static background via
:py:meth:`~kikuchipy.signals.ebsd.EBSD.static_background_correction`:

.. code-block:: python

    >>> s.static_background_correction(operation='subtract', relative=True)

.. image:: _static/image/background_corrections/pattern_raw.jpg
    :scale: 50%
.. image:: _static/image/background_corrections/pattern_static.jpg
    :scale: 50%

Here the static background pattern is assumed to be stored as part of the signal
``metadata``, which can be loaded via
:py:meth:`~kikuchipy.signals.ebsd.EBSD.set_experimental_parameters`. The static
background pattern can also be passed to the ``static_bg`` parameter. Passing
``relative=True`` ensures that relative intensities between patterns are kept
when the patterns are scaled after correction to fill the intensity range
available for the data type, e.g. [0, 255] for ``uint8``.

.. _dynamic-background-correction:

Dynamic background correction
=============================

Uneven intensity in a static background subtracted pattern can be corrected by
subtracting or dividing by a dynamic background obtained by Gaussian blurring.
This so-called flat fielding is done with
:py:meth:`~kikuchipy.signals.ebsd.EBSD.dynamic_background_correction`, with
possibilities of setting the ``operation`` and standard deviation of the
Gaussian kernel, ``sigma``:

.. code-block:: python

    >>> s.dynamic_background_correction(operation='subtract', sigma=2)

.. image:: _static/image/background_corrections/pattern_static.jpg
    :scale: 50%
.. image:: _static/image/background_corrections/pattern_dynamic.jpg
    :scale: 50%

Patterns are rescaled to fill the available data type range.

.. _adaptive-histogram-equalization:

Adaptive histogram equalization
===============================

Adaptive histogram equalization has been found to significantly enhance pattern
contrast :ref:`[Marquardt2017] <[Marquardt2017]>`. With
:py:meth:`~kikuchipy.signals.ebsd.EBSD.adaptive_histogram_equalization`, the
intensities in the pattern histogram are spread to cover the available range,
e.g. [0, 255] for patterns of ``uint8`` data type:

.. code-block:: python

    >>> s.adaptive_histogram_equalization(kernel_size=(15, 15))

.. image:: _static/image/background_corrections/pattern_dynamic.jpg
    :scale: 50%
.. image:: _static/image/background_corrections/pattern_adapthist.jpg
    :scale: 50%

The ``kernel_size`` parameter determines the size of the contextual regions. See
e.g. Fig. 5 in :ref:`[Jackson2019] <[Jackson2019]>`, also available via
`EMsoft's GitHub repository wiki
<https://github.com/EMsoft-org/EMsoft/wiki/DItutorial#52-determination-of-pattern-pre-processing-parameters>`_,
for the effect of varying ``kernel_size``.

.. _rescale-intensities:

Rescale intensities
===================

Only changing the data type using
:py:meth:`~kikuchipy.signals.ebsd.EBSD.change_dtype` does not rescale pattern
intensities, leading to patterns not using the full available data type range,
e.g. [0, 65535] for ``uint16``:

.. code-block:: python

    >>> print(s.data.dtype, s.data.max())
    uint8 255
    >>> s.change_dtype(np.uint16)
    >>> print(s.data.dtype, s.data.max())
    uint16 255
    >>> s.plot(vmax=1000)

.. image:: _static/image/background_corrections/pattern_adapthist_uint16.jpg
    :align: center
    :scale: 50%

In these cases it is convenient to rescale intensities to a desired data type
range, either keeping relative intensities between patterns or not, by using
:py:meth:`~kikuchipy.signals.ebsd.EBSD.rescale_intensities`:

.. code-block:: python

    >>> s.rescale_intensities(relative=True)
    >>> print(s.data.dtype, s.data.max())
    uint16 65535
    >>> s.plot(vmax=65535)

.. image:: _static/image/background_corrections/pattern_adapthist_uint16_rescaled.jpg
    :align: center
    :scale: 50%
