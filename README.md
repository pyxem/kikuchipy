Introduction
------------
KikuchiPy is an open-source Python library for processing of electron
backscatter diffraction (EBSD) patterns. It builds upon the tools for
multi-dimensional data analysis provided by the HyperSpy library for treatment
of experimental EBSD data.

KikuchiPy is released under the GPL v3 license.

Install
------------
Coming soon.

Use
-----
Example usage:

```python
>>> import kikuchipy as kp
>>> s = kp.load('/path/to/Pattern.dat')
>>> s.find_deadpixels()
>>> s.remove_deadpixels()
>>> s.remove_background(bg='/path/to/background_pattern.bmp', static=True,
                        dynamic=True)  # Static and dynamic corrections
>>> s.save('/path/to/Pattern_bgsd.dat')  # Background subtracted patterns
```

Supported formats
-------
So far it is possible to import patterns stored in:
* NORDIF .dat binary files
* HDF5 files
* HyperSpy's .hspy files
