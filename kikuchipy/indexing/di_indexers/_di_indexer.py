# Copyright 2019-2023 The kikuchipy developers
#
# This file is part of kikuchipy.
#
# kikuchipy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# kikuchipy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with kikuchipy. If not, see <http://www.gnu.org/licenses/>.

import abc
from typing import List, Union, Tuple

import numpy as np


class DIIndexer(abc.ABC):
    """Abstract class implementing a function to match
    experimental and simulated EBSD patterns in a dictionary when
    the dictionary can be fit in memory.

    When writing a custom dictionary indexing class, the methods listed as
    `abstract` below must be implemented. Any number of custom
    parameters can be passed. Also listed are the attributes available
    to the methods if set properly during initialization or after.

    Parameters
    ----------
    dtype
        Which data type to cast the patterns to before matching to.
    """

    _allowed_dtypes: List[type] = []

    def __init__(
        self,
        dtype: Union[str, np.dtype, type]
    ):
        """Create a similarity metric matching experimental and
        simulated EBSD patterns in a dictionary.
        """
        self.dictionary_patterns = None
        self._keep_n = None
        self._dtype = np.dtype(dtype)

    def __repr__(self):
        string = f"{self.__class__.__name__}: {np.dtype(self._dtype).name}"
        return string

    def __call__(self, dictionary_patterns: np.ndarray, keep_n: int):
        """Ingest a dictionary of simulated patterns to match to.
        Must be 2D (1 navigation dimension, 1 signal dimension).

        Parameters
        ----------
        dictionary_patterns
            Dictionary of simulated patterns to match to. Must be 2D
            (1 navigation dimension, 1 signal dimension).
        keep_n
            Number of best matches to keep.
        """
        self._keep_n = keep_n
        assert len(dictionary_patterns.shape) == 2
        assert dictionary_patterns.shape[0] > 0 and dictionary_patterns.shape[1] > 0
        self.prepare_dictionary(dictionary_patterns)

    @property
    def allowed_dtypes(self) -> List[type]:
        """Return the list of allowed array data types used during
        matching.
        """
        return self._allowed_dtypes

    @property
    def dtype(self) -> np.dtype:
        """Return or set which data type to cast the patterns to before
        matching.

            Data type listed in :attr:`allowed_dtypes`.
        """
        return self._dtype

    @dtype.setter
    def dtype(self, value: Union[str, np.dtype, type]):
        """Set which data type to cast the patterns to before
        matching.
        """
        self._dtype = np.dtype(value)

    @property
    def keep_n(self) -> int:
        """Return or set which data type to cast the patterns to before
        matching.

            Data type listed in :attr:`allowed_dtypes`.
        """
        return self._keep_n

    @keep_n.setter
    def keep_n(self, value: int):
        """Set which data type to cast the patterns to before
        matching.
        """
        self._keep_n = value

    @property
    def dictionary_patterns(self) -> np.ndarray:
        """Return or set which data type to cast the patterns to before
        matching.

            Data type listed in :attr:`allowed_dtypes`.
        """
        return self._dictionary_patterns

    @dictionary_patterns.setter
    def dictionary_patterns(self, dictionary_patterns: np.ndarray):
        """Set which data type to cast the patterns to before
        matching.
        """
        self._dictionary_patterns = dictionary_patterns

    @abc.abstractmethod
    def prepare_dictionary(self, *args, **kwargs):
        """Prepare all dictionary patterns before matching to experimental
        patterns in :meth:`match`.
        """
        return NotImplemented  # pragma: no cover

    @abc.abstractmethod
    def prepare_experimental(self, *args, **kwargs):
        """Prepare experimental patterns block before matching to dictionary
        patterns in :meth:`match`.
        """
        return NotImplemented  # pragma: no cover

    @abc.abstractmethod
    def query(self, experimental_patterns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Match all experimental patterns block to dictionary patterns
        and return their similarities.
        """
        return NotImplemented  # pragma: no cover