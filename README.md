# KikuchiPy

A class for processing of Kikuchi patterns in the form of Electron Backscatter Diffraction (EBSD) patterns using and
inheriting functionality from [HyperSpy](http://hyperspy.org/hyperspy-doc/current/user_guide/intro.html#what-is-hyperspy)
and [pyXem](http://pyxem.github.io/pyxem/). If you have a NORDIF .dat
file, the patterns can be converted to a format readable by HyperSpy using the
[nordif2hdf5](https://github.com/hwagit/nordif2hdf5) Python script, before
using these scripts.

All functions provide
[lazy](http://hyperspy.org/hyperspy-doc/current/user_guide/big_data.html)
functionality.

Example usage

```python
>>> import hyperspy.api as hs
>>> from signals.electron_backscatter_diffraction import \
ElectronBackscatterDiffraction
>>> s = hs.load('/path/to/data.hdf5')
>>> s = ElectronBackscatterDiffraction(s)
>>> deadpixels = s.find_deadpixels()
>>> s.remove_deadpixels(deadpixels)
>>> s.remove_background(bgimg_path='/path/to/pattern.bmp', inplace=True,
                        parallel=True, static=True, dynamic=True)
```

Below is a list of functions.

### [remove_background](https://github.com/hwagit/ebsp-pro/blob/master/signals/electron_backscatter_diffraction.py)

Perform background correction, either static, dynamic or both. For the static
correction, a background image is subtracted from all patterns. For the dynamic
correction, each pattern is blurred using a Gaussian kernel with a standard
deviation set by you. The correction can either be done by subtraction or
division. Relative intensities between patterns are lost after dynamic
correction.

If dead pixels are removed from the experimental patterns, the same pixels are
also removed from the background pattern before the static correction is
performed.

### [set_experimental_parameters](https://github.com/hwagit/ebsp-pro/blob/master/signals/electron_backscatter_diffraction.py)

Set useful experimental parameters in signal metadata. So far only data for
dead detector pixels can be set.

### [find_deadpixels](https://github.com/hwagit/ebsp-pro/blob/master/signals/electron_backscatter_diffraction.py)

Find dead pixels in experimentally acquired diffraction patterns by comparing
pixel values in a blurred version of a selected pattern to the original
pattern. If the intensity difference is above a threshold the pixel is
labeled as dead. The optimal threshold can be found by studying the returned
plot (which can be muted by passing `to_plot=False`. The output is used as
input for `remove_deadpixels()`.

### [remove_deadpixels](https://github.com/hwagit/ebsp-pro/blob/master/signals/electron_backscatter_diffraction.py)

Remove dead pixels from experimentally acquired diffraction patterns, either by
averaging or setting to a certain value. Uses pyXem's `remove_deadpixels()`
function.
