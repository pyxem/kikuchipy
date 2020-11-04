====================
Visualizing patterns
====================

The :class:`~kikuchipy.signals.EBSD` and
:class:`~kikuchipy.signals.EBSDMasterPattern` object has a
powerful and versatile :meth:`~hyperspy.signal.BaseSignal.plot` method provided
by HyperSpy. Its uses are greatly detailed in HyperSpy's `visualisation user
guide
<http://hyperspy.org/hyperspy-doc/current/user_guide/visualisation.html>`_. This
section details example uses specific to EBSD and EBSDMasterPattern objects.

.. note::

    The Nickel data set used in this section can be downloaded from [Anes2019]_.

.. _navigate-in-custom-map:

Navigate in custom map
======================

Correlating results from e.g. crystal and phase structure determination, i.e.
indexing, with experimental patterns can inform their interpretation. When
calling :meth:`~hyperspy.signal.BaseSignal.plot` without any input
parameters, the navigator map is a grey scale image with pixel values
corresponding to the sum of all detector intensities within that pattern:

.. code-block::

    >>> s.plot()

.. _fig-standard-navigator:

.. figure:: _static/image/visualizing_patterns/s_plot.png
    :align: center
    :width: 100%

    Example of a standard navigator map (right), and the detector (left). This
    is the standard view when calling ``s.plot()`` for EBSD objects.

However, any :class:`~hyperspy.signal.BaseSignal` object with a
two-dimensional ``signal_shape`` corresponding to the scan ``navigation_shape``
can be passed in to the ``navgiator`` parameter in
:meth:`~hyperspy.signal.BaseSignal.plot`, including a virtual image showing
diffraction contrast, any quality metric map, or an orientation map or a phase
map.

.. _navigate-in-virtual-image:

Virtual image
-------------

A virtual backscatter electron (VBSE) image created from any detector region of
interest with the :meth:`~kikuchipy.signals.EBSD.get_virtual_bse_intensity`
method or :meth:`~kikuchipy.generators.VirtualBSEGenerator.get_rgb_image`
explained in the :doc:`virtual_backscatter_electron_imaging` section, can be
used as a navigator for a scan ``s``:

.. code-block:: python

    >>> vbse_gen = kp.generators.VirtualBSEGenerator(s)
    >>> vbse_rgb = vbse_gen.get_rgb_image(r=(4, 1), g=(4, 2), b=(4, 3))
    >>> vbse_rgb
    <VirtualBSEImage, title: , dimensions: (|200, 149)>
    >>> s.plot(navigator=vbse_rgb)

.. _fig-vbse-navigator:

.. figure:: _static/image/visualizing_patterns/vbse_navigation.jpg
    :align: center
    :width: 100%

    Navigating EBSD patterns (left) in an RGB virtual BSE image (right).

.. _image-map:

Any image
---------

Images loaded into a :class:`~hyperspy._signals.signal2d.Signal2D` object can be
used as navigators. E.g. a quality metric map, like the orientation similarity
obtained from dictionary indexing with
`EMsoft <https://github.com/EMsoft-org/EMsoft>`_ (see e.g.
:cite:`marquardt2017quantitative`.):

.. code-block::

    >>> import matplotlib.pyplot as plt
    >>> import hyperspy.api as hs
    >>> osm = plt.imread('path/to/orientation_similarity_map.png'))
    >>> s_osm = hs.signals.Signal2D(osm)
    >>> s_osm
    <Signal2D, title: , dimensions: (|2140, 1603)>
    >>> s_osm = s_osm.rebin(new_shape=s.axes_manager.navigation_shape)
    >>> s_osm
    <Signal2D, title: , dimensions: (|200, 149)>
    >>> s.plot(navigator=s_osm)

.. _fig-navigate-quality-metric:

.. figure:: _static/image/visualizing_patterns/osm_navigation.jpg
    :align: center
    :width: 100%

    Navigating EBSD patterns (left) in a quality metric map ``s_osm``, in this
    case an orientation similarity map from dictionary indexing with EMsoft.

Or, an `image quality map <feature_maps.ipynb#Image-quality>`_ calculated using
:meth:`~kikuchipy.signals.EBSD.get_image_quality`:

.. code-block::

    >>> iq = s.get_image_quality()
    >>> s_iq = hs.signals.Signal2D(iq)
    >>> s.plot(navigator=s_iq)

Using colour images (apart from creating RGB virtual BSE images, as shown
above), e.g. an orientation ``om`` or phase map, is a bit more involved:

.. code-block::

    >>> om = plt.imread('/path/to/orientation_map.jpg')
    >>> om_scaled = ske.rescale_intensity(om, out_range=np.uint8)
    >>> s_om = hs.signals.Signal2D(om_scaled)
    >>> s_om
    <Signal2D, title: , dimensions: (149|3, 200)>
    >>> s_om = s_om.transpose(signal_axes=1)
    >>> print(s_om, s_om.data.dtype)
    <Signal1D, title: , dimensions: (200, 149|3)> uint8
    >>> s_om.change_dtype('rgb8')
    >>> s_om
    <Signal2D, title: , dimensions: (|200, 149)> [('R', 'u1'), ('G', 'u1'), ('B', 'u1')]
    >>> s.plot(navigator=s_om)

.. _fig-orientation-map-navigator:

.. figure:: _static/image/visualizing_patterns/om_navigation.jpg
    :align: center
    :width: 100%

    Navigating EBSD patterns (left) in an orientation map ``s_om`` (right).

.. _plot-multiple-scans:

Plot multiple scans
===================

HyperSpy provides the function :func:`~hyperspy.drawing.utils.plot_signals` to
plot multiple signals with the same navigator, as explained in the `HyperSpy
user guide
<http://hyperspy.org/hyperspy-doc/current/user_guide/visualisation.html#plotting-several-signals>`_.
This enables e.g. plotting of experimental and simulated patterns side by side
as a visual inspection of the indexing results:

.. code-block:: python

    >>> import hyperspy.api as hs
    >>> nav_shape = s.axes_manager.navigation_shape[::-1]
    >>> s_sim = kp.load("simulated_patterns.h5", scan_size=nav_shape)
    >>> s_sim
    <EBSD, title: simulated_patterns, dimensions: (200, 149|60, 60)>
    >>> hs.plot.plot_signals([s, s_sim], navigator=s_om)

.. _fig-plot-multiple-scans:

.. figure:: _static/image/visualizing_patterns/plot_multiple_scans.gif
    :align: center
    :width: 100%

    Plotting of experimental and simulated patterns side by side for visual
    inspection, using an :ref:`orientation map as navigator
    <fig-orientation-map-navigator>`.

.. _plot-master-pattern:

Plot master patterns
====================

:class:`~kikuchipy.signals.EBSDMasterPattern` objects can be navigated along
their energy axis and/or the their northern/southern hemisphere:

.. code-block:: python

    >>> s
    <EBSDMasterPattern, title: , dimensions: (2, 11|1001, 1001)>
    >>> s.axes_manager
    <Axes manager, axes: (11, 2|1001, 1001)>
                Name |   size |  index |  offset |   scale |  units
    ================ | ====== | ====== | ======= | ======= | ======
              energy |     11 |      9 |      10 |       1 |    keV
                   y |      2 |      0 |       0 |       1 | hemisphere
    ---------------- | ------ | ------ | ------- | ------- | ------
               width |   1001 |        |  -5e+02 |       1 |     px
              height |   1001 |        |  -5e+02 |       1 |     px
    >>> s.plot()

.. _fig-master-pattern-plot:

.. figure:: _static/image/visualizing_patterns/master_pattern_plot.png
    :align: center
    :width: 450

    A spherical projection of the northern hemisphere at 19 keV for Ni (top)
    from an EMsoft simulation. A navigator (bottom) for EBSDMasterPattern
    objects, with the beam energy along the horizontal axis and the northern and
    southern hemispheres along the vertical axis, is also shown.
