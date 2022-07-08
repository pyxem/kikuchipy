=========
Tutorials
=========

This page contains more in-depth guides for using kikuchipy. It is broken up into
sections covering specific topics.

For shorter examples, see our :doc:`/examples/index`. For descriptions of
the functions, modules, and objects in kikuchipy, see the :doc:`/reference/index`.

The tutorials are live and available on MyBinder:

.. image:: https://static.mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/pyxem/kikuchipy/develop?filepath=doc/tutorials
   :alt: Launch on Binder

Load and save data
==================

These tutorials cover loading and saving of EBSD patterns, EBSD master patterns, and
virtual backscatter electron images in the file formats supported in kikuchipy.

.. nbgallery::

    load_save_data.ipynb

Reference frames
================

These tutorials cover the relevant reference frames in EBSD and how to determine the
detector-sample geometry, also known as the projection/pattern center (PC).

.. nbgallery::

    reference_frames.ipynb
    pc_calibration_moving_screen_technique.ipynb

Pattern processing
==================

These tutorials cover processing of EBSD pattern intensities for enhancing the Kikuchi
pattern.

.. nbgallery::
    pattern_processing.ipynb

Virtual backscatter electron imaging
====================================

These tutorials cover virtual imaging with the EBSD detector, so-called virtual
backscatter electron (VBSE) imaging.

.. nbgallery::
    virtual_backscatter_electron_imaging.ipynb

Feature maps
============

These tutorials cover extracting qualitative information from pattern intensities.

.. nbgallery::
    feature_maps.ipynb

Indexing
========

These tutorials cover crystal and/or phase determination from EBSD patterns, so-called
indexing.

.. nbgallery::
    pattern_matching.ipynb

Simulations
===========

These tutorials cover simulation of Kikuchi patterns.

.. nbgallery::
    geometrical_ebsd_simulations.ipynb
    kinematical_ebsd_simulations.ipynb

Multivariate analysis
=====================

These tutorials cover multivariate analysis of EBSD patterns.

.. nbgallery::
    multivariate_analysis.ipynb

Visualization
=============

These tutorials cover plotting and visualization of EBSD patterns and maps, as well as
plotting of EBSD master patterns and virtual backscatter electron images.

.. nbgallery::
    visualizing_patterns.ipynb

Tutorials given at workshops
============================

These hands-on tutorials were given at workshops.

.. nbgallery::

    mandm2021_sunday_short_course.ipynb
