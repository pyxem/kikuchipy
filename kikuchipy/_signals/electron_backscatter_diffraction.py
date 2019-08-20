# -*- coding: utf-8 -*-
# Copyright 2019 The KikuchiPy developers
#
# This file is part of KikuchiPy.
#
# KikuchiPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# KikuchiPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with KikuchiPy. If not, see <http://www.gnu.org/licenses/>.

import os
import gc
import numbers
import datetime
import tqdm
import logging
import numpy as np
import dask.array as da
import scipy.ndimage as scn
from h5py import File
from hyperspy.signals import Signal2D
from hyperspy._lazy_signals import LazySignal2D
from hyperspy.learn.mva import LearningResults
from pyxem.signals.electron_diffraction2d import ElectronDiffraction2D
from dask.diagnostics import ProgressBar
from hyperspy.misc.utils import (dummy_context_manager, DictionaryTreeBrowser)
from kikuchipy import io
from kikuchipy.utils.expt_utils import (rescale_pattern_intensity,
                                        equalize_adapthist_pattern)
from kikuchipy.utils.io_utils import (metadata_nodes, kikuchipy_metadata)
from kikuchipy.utils.phase_utils import update_phase_info

_logger = logging.getLogger(__name__)


class EBSD(Signal2D):
    _signal_type = 'EBSD'
    _alias_signal_types = ['electron_backscatter_diffraction']
    _lazy = False

    def __init__(self, *args, **kwargs):
        """Create an EBSD object from a hyperspy.signals.Signal2D or a
        numpy array."""

        if self._lazy and args:
            Signal2D.__init__(self, data=args[0], **kwargs)
        else:
            Signal2D.__init__(self, *args, **kwargs)

        # Update metadata if object is initialised from numpy array
        if not self.metadata.has_item(metadata_nodes(sem=False)):
            md = self.metadata.as_dictionary()
            md.update(kikuchipy_metadata().as_dictionary())
            self.metadata = DictionaryTreeBrowser(md)
        if not self.metadata.has_item('Sample.Phases'):
            self.set_phase_parameters()

    def set_experimental_parameters(self, detector=None,
                                    azimuth_angle=None,
                                    elevation_angle=None,
                                    sample_tilt=None,
                                    working_distance=None, binning=None,
                                    exposure_time=None, grid_type=None,
                                    gain=None, frame_number=None,
                                    frame_rate=None, scan_time=None,
                                    beam_energy=None, xpc=None,
                                    ypc=None, zpc=None,
                                    static_background=None,
                                    manufacturer=None, version=None,
                                    microscope=None, magnification=None):
        """Set experimental parameters in metadata.

        Parameters
        ----------
        azimuth_angle : float, optional
            Azimuth angle of the detector in degrees. If the azimuth is
            zero, the detector is perpendicular to the tilt axis.
        beam_energy : float, optional
            Energy of the electron beam in kV.
        binning : int, optional
            Camera binning.
        detector : str, optional
            Detector manufacturer and model.
        elevation_angle : float, optional
            Elevation angle of the detector in degrees. If the elevation
            is zero, the detector is perpendicular to the incident beam.
        exposure_time : float, optional
            Camera exposure time in µs.
        frame_number : float, optional
            Number of patterns integrated during acquisition.
        frame_rate : float, optional
            Frames per s.
        gain : float, optional
            Camera gain in dB.
        grid_type : str, optional
            Scan grid type, only square grid is supported.
        manufacturer : str, optional
            Manufacturer of software used to collect patterns.
        microscope : str, optional
            Microscope used to collect patterns.
        magnification : int, optional
            Microscope magnification at which patterns were collected.
        sample_tilt : float, optional
            Sample tilt angle from horizontal in degrees.
        scan_time : float, optional
            Scan time in s.
        static_background : np.ndarray, optional
            Static background pattern.
        version : str, optional
            Version of software used to collect patterns.
        working_distance : float, optional
            Working distance in mm.
        xpc, ypc : float, optional
            Pattern centre horizontal and vertical coordinate with
            respect to detector centre.
        zpc : float, optional
            Specimen to scintillator distance.
        """

        from kikuchipy.utils.general_utils import \
            write_parameters_to_dictionary

        md = self.metadata
        sem_node, ebsd_node = metadata_nodes()
        write_parameters_to_dictionary(
            {'beam_energy': beam_energy, 'magnification': magnification,
             'microscope': microscope, 'working_distance': working_distance},
            md, sem_node)
        write_parameters_to_dictionary(
            {'azimuth_angle': azimuth_angle, 'binning': binning,
             'detector': detector, 'elevation_angle': elevation_angle,
             'exposure_time': exposure_time, 'frame_number': frame_number,
             'frame_rate': frame_rate, 'gain': gain, 'grid_type': grid_type,
             'manufacturer': manufacturer, 'version': version,
             'sample_tilt': sample_tilt, 'scan_time': scan_time, 'xpc': xpc,
             'ypc': ypc, 'zpc': zpc, 'static_background': static_background},
            md, ebsd_node)

    def set_phase_parameters(self, number=1, atom_coordinates=None,
                             formula=None, info=None,
                             lattice_constants=None,
                             laue_group=None, material_name=None,
                             point_group=None, setting=None,
                             space_group=None, symmetry=None):
        """Set parameters for one phase in metadata, using the
        International Tables for Crystallography, Volume A. A phase node
        with default values is created if none is present in the
        metadata.

        Parameters
        ----------
        number : int, optional
            Phase number.
        atom_coordinates : dict, optional
            Dictionary of dictionaries one or more of the atoms in the
            unit cell, on the form {'1': {'atom': 'Ni', 'coordinates':
            [0, 0, 0], 'site_occupation': 1, 'debye_waller_factor': 0},
            '2': {'atom': 'O',... etc. `debye_waller_factor` in units of
            nm^2, `site_occupation` in range [0, 1].
        formula : str, optional
            Phase formula, e.g. Fe2 or Ni.
        info : str, optional
            Whatever phase info the user finds relevant.
        lattice_constants : array_like of floats, optional
            Six lattice constants a, b, c, alpha, beta, gamma.
        laue_group : str, optional
        material_name : str, optional
        point_group : str, optional
        setting : int, optional
            Space group's origin setting.
        space_group : int, optional
            Number between 1 and 230.
        symmetry : int, optional
        """

        # Ensure atom coordinates are numpy arrays
        if atom_coordinates is not None:
            for phase, val in atom_coordinates.items():
                atom_coordinates[phase]['coordinates'] = np.array(
                    atom_coordinates[phase]['coordinates']
                )

        inputs = {'atom_coordinates': atom_coordinates, 'formula': formula,
                  'info': info, 'lattice_constants': lattice_constants,
                  'laue_group': laue_group, 'material_name': material_name,
                  'point_group': point_group, 'setting': setting,
                  'space_group': space_group, 'symmetry': symmetry}

        # Remove None values
        phase = {k: v for k, v in inputs.items() if v is not None}
        update_phase_info(self.metadata, phase, number)

    def set_scan_calibration(self, step_x=1., step_y=1.):
        """Set the step size in µm.

        Parameters
        ----------
        step_x, step_y : float
            Scan step size in µm per pixel in horizontal, x and
            vertical, y direction.
        """

        x, y = self.axes_manager.navigation_axes
        x.name, y.name = ('x', 'y')
        x.scale, y.scale = (step_x, step_y)
        x.units, y.units = np.repeat(u'\u03BC'+'m', 2)

    def set_detector_calibration(self, delta):
        """Set detector pixel size in microns. The offset is set to the
        the detector centre.

        Parameters
        ----------
        delta : float
            Detector pixel size in microns.
        """

        centre = np.array(self.axes_manager.signal_shape) / 2 * delta
        dx, dy = self.axes_manager.signal_axes
        dx.units, dy.units = (u'\u03BC'+'m', u'\u03BC'+'m')
        dx.scale, dy.scale = (delta, delta)
        dx.offset, dy.offset = -centre

    def static_background_correction(self, operation='divide',
                                     relative=False, static_bg=None,
                                     out_range=np.uint8):
        """Correct static background using a static background pattern.

        Pattern intensities are rescaled according to specified
        intensity range.

        Parameters
        ----------
        operation : 'divide' or 'subtract', optional
            Divide or subtract by static background pattern.
        relative : bool, optional
            Keep relative intensities between patterns.
        static_bg : {np.ndarray, da.Array or None}, optional
            Static background pattern. If not passed we try to read it
            from the signal metadata.
        out_range : dtype or tuple, optional
            Output intensity range given by data type or (min, max). If
            a tuple is passed, the output data type is np.float32.
        """

        # Make sure static background pattern is valid
        ebsd_node = metadata_nodes(sem=False)
        md = self.metadata
        if not isinstance(static_bg, (np.ndarray, da.Array)):
            try:
                static_bg = md.get_item(ebsd_node + '.static_background')
            except TypeError:
                raise TypeError("Static background is not a numpy array or "
                                "could not be read from signal metadata.")
        pat_shape = self.axes_manager.signal_shape
        if static_bg.shape != pat_shape:
            raise IOError("Pattern {} and static background {} shapes must be "
                          "identical.".format(pat_shape, static_bg.shape))
        dtype = np.float16
        static_bg = static_bg.astype(dtype)

        if operation == 'divide':
            self.data = self.data / static_bg
        else:
            self.data = self.data - static_bg

        self.data = rescale_pattern_intensity(self, out_range=out_range,
                                              relative=relative)

    def dynamic_background_correction(self, operation='divide',
                                      sigma=None, out_range=np.uint8,
                                      **kwargs):
        """Correct dynamic background.

        Parameters
        ----------
        operation : 'divide' or 'subtract', optional
            Divide or subtract by dynamic background pattern.
        sigma : {int, float or None}, optional
            Standard deviation of the gaussian kernel. If None
            (default), a deviation of pattern width/30 is chosen.
        out_range : dtype or tuple, optional
            Output intensity range given by data type or (min, max). If
            a tuple is passed, the output data type is np.float32.
        **kwargs :
            Keyword arguments passed to map and map_blocks.
        """

        if sigma is None:
            sigma = self.axes_manager.signal_axes[0].size/30
        dtype = np.int16
        self.data = self.data.astype(dtype)
        if self._lazy:
            def func(block, sigma):
                return scn.gaussian_filter(block, sigma=sigma, truncate=2.0)
            blurred = self.data.map_blocks(func, sigma=sigma, **kwargs)
        else:
            blurred = self.map(scn.gaussian_filter, sigma=sigma, truncate=2.0,
                               inplace=False, show_progressbar=False, **kwargs)

        if operation == 'divide':
            self.data = self.data / blurred
        else:
            self.data = self.data - blurred

        self.data = rescale_pattern_intensity(self, out_range=out_range)

    def equalize_adapthist(self, kernel_size=None, clip_limit=0.01,
                           nbins=256, **kwargs):
        """Local contrast enhancement using contrast limited adaptive
        histogram equalisation (CLAHE).

        Input data is assumed to be a two-dimensional numpy array of
        patterns of dtype uint8.

        Parameters
        ----------
        kernel_size : integer or list-like, optional
            Defines the shape of contextual regions used in the
            algorithm. By default, ``kernel_size`` is 1/8 of ``pattern``
            height by 1/8 of its width, or a minimum of 20 in either
            direction.
        clip_limit : float, optional
            Clipping limit, normalised between 0 and 1 (higher values
            give more contrast).
        nbins : int, optional
            Number of gray bins for histogram ("data range").
        **kwargs
            Arguments to be passed to map().

        Notes
        -----
        Adapted from scikit-image, without rescaling the pattern before
        equalisation and returning it with correct data type. See
        ``skimage.exposure.equalize_adapthist`` documentation for more
        details.
        """

        if self._lazy:
            kwargs['ragged'] = False

        # Set kernel size, ensuring it is at least 20 in each direction
        sdim = 2
        if kernel_size is None:
            sx, sy = self.axes_manager.signal_shape
            kernel_size = (sx // 8, sy // 8)
        elif isinstance(kernel_size, numbers.Number):
            kernel_size = (kernel_size,) * sdim
        elif len(kernel_size) != sdim:
            ValueError(
                "Incorrect value of `kernel_size`: {}".format(kernel_size))
        kernel_size = [int(k) for k in kernel_size]
        kernel_size = [20 if i < 20 else i for i in kernel_size]

        self.map(equalize_adapthist_pattern, kernel_size=kernel_size,
                 clip_limit=clip_limit, nbins=nbins, **kwargs)

    def get_virtual_image(self, roi):
        """Method imported from
        pyxem.signals.ElectronDiffraction2D.get_virtual_image. Obtains
        a virtual image associated with a specified ROI.

        Parameters
        ----------
        roi: hyperspy.roi.BaseInteractiveROI
            Any interactive ROI detailed in HyperSpy.

        Returns
        -------
        dark_field_sum: hyperspy.signals.BaseSignal
            The virtual image signal associated with the specified roi.

        Examples
        --------
        .. code-block:: python

            import hyperspy.api as hs
            roi = hs.roi.RectangularROI(left=10, right=20, top=10,
                bottom=20)
            s.get_virtual_image(roi)
        """
        return ElectronDiffraction2D.get_virtual_image(self, roi)

    def plot_interactive_virtual_image(self, roi, **kwargs):
        """Method imported from
        pyXem.ElectronDiffraction.plot_interactive_virtual_image(self,
        roi). Plots an interactive virtual image formed with a
        specified and adjustable roi.

        Parameters
        ----------
        roi: hyperspy.roi.BaseInteractiveROI
            Any interactive ROI detailed in HyperSpy.
        **kwargs:
            Keyword arguments to be passed to `ElectronDiffraction.plot`

        Examples
        --------
        .. code-block:: python

            import hyperspy.api as hs
            roi = hs.roi.RectangularROI(left=10, right=20, top=10,
                bottom=20)
            s.plot_interactive_virtual_image(roi)
        """

        return ElectronDiffraction2D.plot_interactive_virtual_image(self, roi,
                                                                    **kwargs)

    def save(self, filename=None, overwrite=None, extension=None,
             **kwargs):
        """Save signal in the specified format.

        The function gets the format from the extension: `h5`, `hdf5` or
        `h5ebsd` for KikuchiPy's specification of the the h5ebsd format
        `dat` for the NORDIF binary format or `hspy` for HyperSpy's
        HDF5 specification. If no extension is provided the default
        file format as defined in the `preferences` is used. Please note
        that not all formats support saving datasets of arbitrary
        dimensions. Each format accepts a different set of parameters.

        For details see the specific format documentation in
        `kikuchipy.io_plugins.<format>.file_writer`.

        Parameters
        ----------
        filename : {str or None}, optional
            If None (default) and `tmp_parameters.filename` and
            `tmp_parameters.folder` are defined, the filename and path
            will be taken from there. A valid extension can be provided
            e.g. "data.h5", see `extension`.
        overwrite : {None, bool}, optional
            If None and the file exists, it will query the user. If
            True (False) it (does not) overwrite the file if it exists.
        extension : {None, 'h5', 'hdf5', 'h5ebsd', 'hspy', 'dat',
                     'png', 'tiff', etc.}, optional
            Extension of the file that defines the file format. 'h5',
            'hdf5' and 'h5ebsd' are equivalent. If None, the extension
            is determined from the following list in this order: i) the
            filename, ii)  `Signal.tmp_parameters.extension` or iii)
            `hspy` (HyperSpy's default extension)
        **kwargs :
            Keyword arguments passed to writer.
        """

        if filename is None:
            if (self.tmp_parameters.has_item('filename') and
                    self.tmp_parameters.has_item('folder')):
                filename = os.path.join(
                    self.tmp_parameters.folder,
                    self.tmp_parameters.filename)
                extension = (self.tmp_parameters.extension
                             if not extension
                             else extension)
            elif self.metadata.has_item('General.original_filename'):
                filename = self.metadata.General.original_filename
            else:
                raise ValueError("Filename not defined.")
        if extension is not None:
            basename, ext = os.path.splitext(filename)
            filename = basename + '.' + extension
        io.save(filename, self, overwrite=overwrite, **kwargs)

    def get_decomposition_model(self, components=None,
                                dtype_out=np.float16, *args, **kwargs):
        """Return the model signal generated with the selected number of
        principal components.

        This function calls HyperSpy's get_decomposition_model. The
        learning results are preconditioned before this call, doing the
        following: (1) set data type to desired dtype, (2) remove
        unwanted components, (3) rechunk, if dask arrays, to suitable
        chunk.

        Parameters
        ----------
        components : {None, int or list of ints}, optional
            If None (default), rebuilds the signal from all components.
            If int, rebuilds signal from components in range 0-given
            int. If list of ints, rebuilds signal from only components
            in given list.
        dtype_out : {np.float16, np.float32, np.float64}, optional
            Data type of learning results (default is float16).
            HyperSpy's ``decomposition`` returns them in float64, which
            here is assumed to be overkill.
        *args
            Passed to Hyperspy's `get_decomposition_model`.
        **kwargs
            Passed to Hyperspy's `get_decomposition_model`.

        Returns
        -------
        s_model : kikuchipy.signals.EBSD or kikuchipy.signals.LazyEBSD
        """

        # Change dtype
        target = self.learning_results
        factors_orig = target.factors.copy()  # Keep to revert target in the end
        loadings_orig = target.loadings.copy()
        factors = target.factors.astype(dtype_out)
        loadings = target.loadings.astype(dtype_out)

        # Extract relevant components
        if hasattr(components, '__iter__'):  # components is a list of ints
            # TODO: This should be implemented in HyperSpy
            factors = factors[:, components]
            loadings = loadings[:, components]
        else:  # components is an int
            factors = factors[:, :components]
            loadings = loadings[:, :components]

        # Update learning results
        self.learning_results.factors = factors
        self.learning_results.loadings = loadings

        # Rechunk
        if isinstance(factors, da.Array):
            chunks = self._rechunk_learning_results()
            self.learning_results.factors = factors.rechunk(chunks=chunks[0])
            self.learning_results.loadings = loadings.rechunk(chunks=chunks[1])

        # Call HyperSpy's function
        s_model = super().get_decomposition_model(*args, **kwargs)

        # Revert learning results to original results
        self.learning_results.factors = factors_orig
        self.learning_results.loadings = loadings_orig

        # Revert class
        assign_class = EBSD
        if self._lazy:
            assign_class = LazyEBSD
        self.__class__ = assign_class
        s_model.__class__ = assign_class

        # Remove learning results from model signal
        s_model.learning_results = LearningResults()

        return s_model

    def decomposition(self, normalize_poissonian_noise=False,
                      algorithm='svd', output_dimension=None,
                      centre=None, auto_transpose=True,
                      navigation_mask=None, signal_mask=None,
                      var_array=None, var_func=None,
                      polyfit=None, reproject=None, return_info=False, *args,
                      **kwargs):
        super().decomposition(normalize_poissonian_noise, algorithm,
                              output_dimension, centre, auto_transpose,
                              navigation_mask, signal_mask, var_array, var_func,
                              polyfit, reproject, return_info, *args, **kwargs)
        self.__class__ = EBSD

    def rebin(self, new_shape=None, scale=None, crop=True, out=None):
        out = super().rebin(new_shape=new_shape, scale=scale, crop=crop,
                            out=out)
        _logger.info("Rebinning changed data type to {}".format(out.data.dtype))

        # Update binning in metadata
        md = out.metadata
        ebsd_node = metadata_nodes(sem=False)
        if scale is None:
            sx = self.axes_manager.signal_shape[0]
            scale = [sx / new_shape[2]]
        old_binning = md.get_item(ebsd_node + '.binning')
        md.set_item(ebsd_node + '.binning', scale[2] * old_binning)

        return out


class LazyEBSD(EBSD, LazySignal2D):

    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self, progressbar=True, close_file=False):
        """Attempt to store the full signal in memory.

        Parameters
        ----------
        progressbar : bool, optional
        close_file: bool, optional
            If True, attempt to close the file associated with the dask
            array data if any. Note that closing the file will make all
            other associated lazy signals inoperative.
        """

        if progressbar:
            cm = ProgressBar
        else:
            cm = dummy_context_manager
        with cm():
            data = self.data
            data = data.compute()
            if close_file:
                self.close_file()
            self.data = data
        self._lazy = False
        self.__class__ = EBSD

    def get_decomposition_model_write(self, components=None,
                                      dtype_learn=np.float16,
                                      mbytes_chunk=100, out_dir=None,
                                      out_fname=None):
        """Write the model signal generated from the selected number of
        principal components directly to a .hspy file. The model signal
        intensities are rescaled to the original signals' data type
        range.

        Notes
        -----
        Multiplying the learning results' factors and loadings in memory
        to create the model signal cannot sometimes be done due to too
        large matrices. Here, instead, learning results are written to
        file, read into dask arrays and multiplied using dask's
        ``matmul``, out of core.

        Parameters
        ----------
        components : {None, int or list of ints}, optional
            If None (default), rebuilds the signal from all components.
            If int, rebuilds signal from components in range 0-given
            int. If list of ints, rebuilds signal from only components
            in given list.
        dtype_learn : {np.float16, np.float32 or np.float64}, optional
            Data type to set learning results to (default is float16).
        mbytes_chunk : int, optional
            Size of learning results chunks in MB, default is 100 MB as
            suggested in the Dask documentation.
        out_dir : str, optional
            Directory to place output signal in.
        out_fname : str, optional
            Name of output signal file.
        """

        # Change dtype
        target = self.learning_results
        factors = np.array(target.factors, dtype=dtype_learn)
        loadings = np.array(target.loadings, dtype=dtype_learn)

        # Extract relevant components
        if hasattr(components, '__iter__'):  # components is a list of ints
            # TODO: This should be implemented in HyperSpy
            factors = factors[:, components]
            loadings = loadings[:, components]
        else:  # components is an int
            factors = factors[:, :components]
            loadings = loadings[:, :components]

        # Write learning results to HDF5 file
        if out_dir is None:
            try:
                out_dir = self.original_metadata.General.original_filepath
            except AttributeError:
                raise AttributeError("Output directory has to be specified")

        t_str = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        file_learn = os.path.join(out_dir, 'learn_' + t_str + '.h5')
        with File(file_learn, 'w') as f:
            f.create_dataset(name='factors', data=factors)
            f.create_dataset(name='loadings', data=loadings)

        # Matrix multiplication
        with File(file_learn, 'r') as f:
            # Read learning results from HDF5 file
            chunks = self._rechunk_learning_results(mbytes_chunk=mbytes_chunk)
            factors = da.from_array(f['factors'], chunks=chunks[0])
            loadings = da.from_array(f['loadings'], chunks=chunks[1])

            # Perform the matrix multiplication
            loadings = loadings.T
            res = factors @ loadings
            res = res.T  # Transpose

            # Create new signal from multiplied matrix
            s_model = self.deepcopy()
            s_model.learning_results = LearningResults()
            s_model.data = res.reshape(s_model.data.shape)
            s_model.data = s_model.data.rechunk(chunks=(1, 1, -1, -1))

            # Rescale intensities and revert data type
            s_model.map(rescale_pattern_intensity, ragged=False)
            s_model.data = s_model.data.astype(self.data.dtype)

            # Write signal to file (rechunking saves a little time?)
            if out_fname is None:
                out_fname = 'model_' + t_str
            file_model = os.path.join(out_dir, out_fname)
            s_model.save(file_model)

        # Delete temporary files
        os.remove(file_learn)
        gc.collect()  # Don't sink

    def _rechunk_learning_results(self, mbytes_chunk=100):
        """Return suggested data chunks for learning results. It is
        assumed that the loadings are not transposed. The last axes of
        factors and loadings are not chunked. The aims in prioritised
        order:
            1. Split into at least as many chunks as available CPUs.
            2. Limit chunks to approx. input MB (`mbytes_chunk`).
            3. Keep first axis of factors (detector pixels).

        Parameters
        ----------
        mbytes_chunk : int, optional
            Size of chunks in MB, default is 100 MB as suggested in the
            Dask documentation.

        Returns
        -------
        List of two tuples
            The first/second tuple are suggested chunks to pass to
            ``dask.array.rechunk`` for factors/loadings, respectively.
        """

        target = self.learning_results
        if target.decomposition_algorithm is None:
            raise ValueError("No learning results were found.")

        # Get dask chunks
        tshape = target.factors.shape + target.loadings.shape

        # Make sure the last factors/loading axes have the same shapes
        # TODO: Should also handle the case where the first axes are the same
        if tshape[1] != tshape[3]:
            raise ValueError("The last dimensions in factors and loadings are "
                             "not the same.")

        # Determine maximum number of (strictly necessary) chunks
        suggested_size = mbytes_chunk * 2**20
        factors_size = target.factors.nbytes
        loadings_size = target.loadings.nbytes
        total_size = factors_size + loadings_size
        num_chunks = np.ceil(total_size / suggested_size)

        # Get chunk sizes
        cpus = os.cpu_count()
        if num_chunks <= cpus:  # Return approx. as many chunks as CPUs
            chunks = [(-1, -1), (int(tshape[2]/cpus), -1)]  # -1 = don't chunk
        elif factors_size <= suggested_size:  # Chunk first axis in loadings
            chunks = [(-1, -1), (int(tshape[2]/num_chunks), -1)]
        else:  # Chunk both first axes
            sizes = [factors_size, loadings_size]
            while (sizes[0] + sizes[1]) >= suggested_size:
                i = np.argmax(sizes)
                sizes[i] = np.floor(sizes[i] / 2)
            factors_chunks = int(np.ceil(factors_size/sizes[0]))
            loadings_chunks = int(np.ceil(loadings_size/sizes[1]))
            chunks = [(int(tshape[0]/factors_chunks), -1),
                      (int(tshape[2]/loadings_chunks), -1)]

        return chunks

    def decomposition(self, normalize_poissonian_noise=False,
                      algorithm=None, output_dimension=None,
                      mbytes_chunk=100, navigation_mask=None,
                      signal_mask=None, *args, **kwargs):
        """Decomposition with a choice of algorithms.

        For a full description of parameters see
        :func:`hyperspy._signals.lazy.decomposition()`.

        This is a wrapper for HyperSpy's ``decomposition()`` function,
        except for an alternative use of scikit-image's IncrementalPCA
        algorithm.

        Parameters
        ----------
        normalize_poissonian_noise : bool
            If True (default is False), scale the patterns to normalise
            Poissonian noise.
        algorithm : {'svd', 'IPCA', 'PCA', 'ORPCA' or 'ONMF'}, optional
            Default is 'svd', lazy SVD decomposition from dask. 'PCA'
            gives HyperSpy's use of scikit-learn's IncrementalPCA,
            while 'IPCA' gives our use of IncrementalPCA.
        output_dimension : int
            Number of significant components to keep. If None, keep all
            (only valid for SVD).
        mbytes_chunk : int, optional
            Size of chunks in MB, default is 100 MB as suggested in the
            Dask documentation.
        navigation_mask, signal_mask : boolean array_like
        *args :
            Arguments to be passed to ``decomposition()``.
        **kwargs :
            Keyword arguments to be passed to ``decomposition()``.

        Returns
        -------
        The results are stored in self.learning_results.
        """

        if self.data.dtype.char not in ['e', 'f', 'd']:  # If not float
            raise TypeError("To perform a decomposition the data must be of "
                            "float type, but the current type is '{}'. No "
                            "decomposition was "
                            "performed.".format(self.data.dtype))

        if algorithm == 'IPCA':
            if output_dimension is None:
                raise ValueError("With the IncrementalPCA algorithm, "
                                 "output_dimension must be specified")

            from sklearn.decomposition import IncrementalPCA

            # Normalise Poissonian noise
            original_data = self.data
            if normalize_poissonian_noise:
                rbH, raG = self._normalize_poissonian_noise(
                    navigation_mask=navigation_mask, signal_mask=signal_mask)

            # Prepare data matrix
            nx, ny, sx, sy = self.data.shape
            n, s = nx * ny, sx * sy
            X = self.data.reshape((n, s))

            # Determine number of chunks
            suggested_size = mbytes_chunk * 2 ** 20
            num_chunks = int(np.ceil(X.nbytes / suggested_size))
            cpus = os.cpu_count()
            if num_chunks <= cpus:
                num_chunks = cpus
            chunk_size = n // num_chunks

            # Get principal components (factors)
            ipca = IncrementalPCA(n_components=output_dimension)
            for i in tqdm.tqdm(iterable=range(0, num_chunks), total=num_chunks,
                               leave=True, desc='Learn'):
                start = i * chunk_size
                end = (i + 1) * chunk_size
                if i == (num_chunks - 1):  # Last iteration
                    end = None
                ipca.partial_fit(X[start:end])  # Fit
            factors = ipca.components_.T

            # Reproject data on the principal components (loadings)
            loadings = []
            for j in tqdm.tqdm(iterable=range(0, num_chunks), total=num_chunks,
                               leave=True, desc='Project'):
                start = j * chunk_size
                end = (j + 1) * chunk_size
                if j == (num_chunks - 1):  # Last iteration
                    end = None
                loadings.append(ipca.transform(X[start:end]))  # Reproject
            loadings = np.concatenate(loadings, axis=0)

            # Set signal's learning results
            target = self.learning_results
            target.decomposition_algorithm = algorithm
            target.output_dimension = output_dimension
            target.factors = factors
            target.loadings = loadings
            target.explained_variance = ipca.explained_variance_
            target.explained_variance_ratio = ipca.explained_variance_ratio_

            # Revert data
            self.data = original_data

            if normalize_poissonian_noise is True:
                target.factors = target.factors * rbH.ravel()[:, np.newaxis]
                target.loadings = target.loadings * raG.ravel()[:, np.newaxis]

        else:  # Call HyperSpy's implementation
            super().decomposition(normalize_poissonian_noise, algorithm,
                                  output_dimension, navigation_mask,
                                  signal_mask, *args, **kwargs)

        self.__class__ = LazyEBSD

    def _normalize_poissonian_noise(self, navigation_mask=None,
                                    signal_mask=None):
        """Scales the patterns following [1]_.

        Adapted from HyperSpy.

        Parameters
        ----------
        navigation_mask, signal_mask : boolean array_like

        Returns
        -------
        raG : array_like
            Matrix corresponding to square root of aG in referenced
            paper.
        rbH : array_like
            Matrix corresponding to square root of bH in referenced
            paper.

        References
        ----------
        .. [1] Keenan, Michael R, Kotula, Paul G: Accounting for Poisson
               noise in the multivariate analysis of ToF-SIMS spectrum
               images, Surface and Interface Analysis 36(3), Wiley
               Online Library, 203–212, 2004.
        """

        from hyperspy._signals.lazy import to_array
        data = self._data_aligned_with_axes
        ndim = self.axes_manager.navigation_dimension
        sdim = self.axes_manager.signal_dimension
        nav_chunks = data.chunks[:ndim]
        sig_chunks = data.chunks[ndim:]
        nm = da.logical_not(
            da.zeros(self.axes_manager.navigation_shape[::-1],
                     chunks=nav_chunks)
            if navigation_mask is None else to_array(
                navigation_mask, chunks=nav_chunks))
        sm = da.logical_not(
            da.zeros(
                self.axes_manager.signal_shape[::-1],
                chunks=sig_chunks)
            if signal_mask is None else to_array(
                signal_mask, chunks=sig_chunks))
        bH, aG = da.compute(
            data.sum(axis=tuple(range(ndim))),
            data.sum(axis=tuple(range(ndim, ndim + sdim))))
        bH = da.where(sm, bH, 1)
        aG = da.where(nm, aG, 1)

        raG = da.sqrt(aG)
        rbH = da.sqrt(bH)

        coeff = raG[(...,) + (None,) * rbH.ndim] * \
                rbH[(None,) * raG.ndim + (...,)]
        coeff.map_blocks(np.nan_to_num)
        coeff = da.where(coeff == 0, 1, coeff)
        data = data / coeff
        self.data = data

        return rbH, raG
