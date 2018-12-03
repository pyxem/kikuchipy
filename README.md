# ebsp-pro

A collection of Python command-line scripts for PROcessing of Electron BackScatter Patterns (EBSPs) using [HyperSpy](https://github.com/hyperspy/hyperspy/). If you have a NORDIF .dat file, the patterns can be converted to a format readable by HyperSpy using the [nordif2hdf5](https://github.com/hwagit/nordif2hdf5) Python script, before using these scripts.

All scripts allow a `--lazy` argument if operations are to be done [lazily](http://hyperspy.org/hyperspy-doc/current/user_guide/big_data.html).

### bakcground_static.py

Subtract a static background from electron backscatter diffraction patterns using a background image from NORDIF or your own background image of the same pixel size as the patterns. The background pattern can either be subtracted or divided by (by passing a `--divide` argument). Relative intensities are maintained.

Running it:

```bash
$ python background_static.py /path/to/pattern_file.hdf5 --bg_img /path/to/background_image_file.bmp
```

### background_dynamic.py

Subtract a dynamic background from electron backscatter patterns. Each pattern is blurred using a Gaussian kernel with a set sigma value. The blurred pattern can either be subtracted or divided by (by passing a `--divide` argument). Relative intensities are lost since a unique pattern is subtracted from each pattern.

Running it:

```bash
$ python background_dynamic.py /path/to/pattern_file.hdf5 --sigma 8
```

### decompose.py

Perform statistical decomposition (machine learning) of a stack of EBSPs, as explained in the [HyperSpy documentation](http://hyperspy.org/hyperspy-doc/current/user_guide/mva.html#saving-and-loading-results).

Running it:

```bash
$ python decompose.py /path/to/pattern_file.hdf5 --lazy --algorithm PCA --components 300
```

Pass a `--save-to-npz` argument if you want to write the learning results to an external .npz file. Default is to write the signal and results to one .hspy. If the original file is a .hspy file, this file is overwritten.

### create_model_signal.py

Create a stack of denoised EBSPs using a limited set of decomposition components to make a model of the initial data, omitting components that ideally contain noise. See the [HyperSpy documentation](http://hyperspy.org/hyperspy-doc/current/user_guide/mva.html#denoising) for more details.

This script has serious limitations on available memory, unless the learning results are stored in the original .hspy file and not read from an external .npz file. Then both signal and learning results can be read lazily. If the original data in 16-bit float precision (~3x the size of 8-bit integer precision?) can be kept in memory, passing a `--compute` argument will write the model signal to disk before rescaling intensities, reverting the data type to 8-bit integer precision and writing the model signal file.

Running it:

```bash
$ python create_model_signal.py /path/to/pattern_file.hspy 100 --lazy --compute
```
