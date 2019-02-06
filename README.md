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
Typical workflow:

```python
>>> import kikuchipy as kp
>>> s = kp.load('/path/to/Pattern.dat')
>>> s.plot()  # Have a look at your data!
>>> s.find_deadpixels()
>>> s.remove_deadpixels()
>>> s.remove_background(bg='/path/to/background_pattern.bmp', static=True,
                        dynamic=True)  # Static and dynamic corrections
>>> s.decomposition()  # Assuming you can keep all computations in memory
>>> components = s.classify_decomposition_components()
# Inspect learning results, comparing them to suggested components to keep
>>> s.plot_decomposition_results()
>>> sc = s.get_decomposition_model(components)
>>> sc.save('/path/to/Pattern_denoised.dat')
```

Supported formats
-------
So far it is possible to import patterns stored in:
* NORDIF .dat binary files
* HDF5 files
* HyperSpy's .hspy files
