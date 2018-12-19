# ebsp-pro

A class for PROcessing of Electron Backscatter Patterns (EBSPs) using and
inheriting functionality from [HyperSpy](http://hyperspy.org/hyperspy-doc/current/user_guide/intro.html#what-is-hyperspy)
and [pyXem](http://pyxem.github.io/pyxem/). If you have a NORDIF .dat
file, the patterns can be converted to a format readable by HyperSpy using the
[nordif2hdf5](https://github.com/hwagit/nordif2hdf5) Python script, before
using these scripts.

All functions provide
[lazy](http://hyperspy.org/hyperspy-doc/current/user_guide/big_data.html)
functionality.

Below is a list of new functions.

### [remove_background](https://github.com/hwagit/ebsp-pro/blob/master/signals/electron_backscatter_diffraction.py)

Perform background correction, either static, dynamic or both. For the static
correction, a background image is subtracted from all patterns. For the dynamic
correction, each pattern is blurred using a Gaussian kernel with a standard
deviation set by you. The correction can either be done by subtraction or
division. Relative intensities between patterns are lost after dynamic
correction.

Example usage

```python
>>> import hyperspy.api as hs
>>> from signals.electron_backscatter_diffraction import \
ElectronBackscatterDiffraction
>>> s = hs.load('/path/to/data.hdf5')
>>> s = ElectronBackscatterDiffraction(s)
>>> s.remove_background(bgimg_path='/path/to/pattern.bmp', inplace=True,
                        parallel=True, static=True, dynamic=True)
```
