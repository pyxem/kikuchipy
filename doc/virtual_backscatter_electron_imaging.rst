====================================
Virtual backscatter electron imaging
====================================

Angle resolved backscatter electron (BSE) imaging can be performed interactively
with the method
:meth:`~kikuchipy.signals.ebsd.EBSD.virtual_backscatter_electron_imaging`,
adopted from `pyXem <http://pyxem.org/>`_, by integrating the intensities within
a part, e.g. a (10 x 10) pixel rectangular region of interest (ROI), of the stack
of EBSD patterns:

.. code-block:: python

    >>> import hyperspy.api as hs
    >>> roi = hs.roi.RectangularROI(left=0, top=0, right=10, bottom=10)
    >>> roi
    RectangularROI(left=0, top=0, right=10, bottom=10)
    >>> s.virtual_backscatter_electron_imaging(roi)
    >>> roi
    RectangularROI(left=25, top=42, right=30, bottom=47)

.. _fig-virtual-backscatter-electron-imaging:

.. figure:: _static/image/virtual_backscatter_electron_imaging/virtual_backscatter_electron_imaging.gif
    :align: center
    :width: 100%

    Interactive virtual backscatter electron imaging with.

Note that the position of the ROI on the detector is updated during the
interactive plotting. See `HyperSpy's ROI user guide
<http://hyperspy.org/hyperspy-doc/current/user_guide/tools.html#region-of-interest-roi>`_
for more detailed use of these.

The virtual image, created from integrating the intensities within the ROI, can
then be written to an image file using
:py:meth:`~kikuchipy.signals.ebsd.EBSD.get_virtual_image`:

.. code-block:: python

    >>> import matplotlib.pyplot as plt
    >>> vbse = s.get_virtual_image(roi)
    >>> vbse
    <EBSD, title: Virtual Dark Field, dimensions: (|200, 149)>
    >>> plt.imsave(fname='/path/to/virtual_image.png', arr=vbse.data)
