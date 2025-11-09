"""
Author: Michele Romani
Email: michele.romani.zaltieri@gmail.com
Copyright 2025 Michele Romani
"""
import numpy as np
import logging
from typing import Optional, Literal
from mne.io import RawArray
from opencortex.neuroengine.flux.base.node import Node


class DownsampleNode(Node):
    """
    Downsample EEG signals in a RawArray.

    Supports multiple methods:
    - 'decimate': Simple decimation (take every Nth sample)
    - 'linear': Linear interpolation to target number of points
    - 'resample': MNE's resample method (anti-aliasing filter + resampling)
    """



    def __init__(
            self,
            target_sfreq: Optional[float] = None,
            num_points: Optional[int] = None,
            method: Literal['decimate', 'linear', 'resample'] = 'resample',
            npad: str = 'auto',
            name: str = None
    ):
        """
        Initialize the DownsampleNode.

        Args:
            target_sfreq: Target sampling frequency in Hz (e.g., 125 Hz)
            num_points: Target number of time points (alternative to target_sfreq)
            method: Downsampling method:
                - 'decimate': Simple decimation (fast, no anti-aliasing)
                - 'linear': Linear interpolation (fast, moderate quality)
                - 'resample': MNE resample with anti-aliasing (slower, best quality)
            npad: Padding for resampling ('auto' or int). Only for method='resample'.
            name: Optional name for this node.

        Note:
            Must specify either target_sfreq OR num_points, not both.
        """
        super().__init__(name or "Downsample")

        if target_sfreq is None and num_points is None:
            raise ValueError("Must specify either target_sfreq or num_points")

        if target_sfreq is not None and num_points is not None:
            raise ValueError("Cannot specify both target_sfreq and num_points")

        self.target_sfreq = target_sfreq
        self.num_points = num_points
        self.method = method
        self.npad = npad

        self.original_sfreq = None
        self.downsample_factor = None

    def __call__(self, data: RawArray) -> RawArray:
        """
        Downsample the RawArray.

        Args:
            data: MNE RawArray object

        Returns:
            Downsampled MNE RawArray object
        """
        self.original_sfreq = data.info['sfreq']

        # Calculate target based on input
        if self.target_sfreq is not None:
            if self.target_sfreq >= self.original_sfreq:
                logging.debug(
                    f"Target sfreq ({self.target_sfreq} Hz) >= original sfreq "
                    f"({self.original_sfreq} Hz). Returning original data."
                )
                return data

            actual_target_sfreq = self.target_sfreq
            self.downsample_factor = self.original_sfreq / self.target_sfreq

        else:  # num_points is specified
            current_n_times = data.n_times
            if self.num_points >= current_n_times:
                logging.debug(
                    f"Target num_points ({self.num_points}) >= original n_times "
                    f"({current_n_times}). Returning original data."
                )
                return data

            # Calculate equivalent sampling frequency
            actual_target_sfreq = (self.num_points / current_n_times) * self.original_sfreq
            self.downsample_factor = current_n_times / self.num_points

        # Apply downsampling based on method
        if self.method == 'resample':
            downsampled = self._resample_mne(data, actual_target_sfreq)
        elif self.method == 'decimate':
            downsampled = self._decimate(data, actual_target_sfreq)
        elif self.method == 'linear':
            downsampled = self._linear_interpolate(data, actual_target_sfreq)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        logging.debug(
            f"Downsampled from {self.original_sfreq} Hz ({data.n_times} samples) "
            f"to {downsampled.info['sfreq']:.2f} Hz ({downsampled.n_times} samples) "
            f"using method='{self.method}'"
        )

        return downsampled

    def _resample_mne(self, data: RawArray, target_sfreq: float) -> RawArray:
        """
        Downsample using MNE's resample method (best quality).
        Applies anti-aliasing filter before resampling.
        """
        downsampled = data.copy().resample(sfreq=target_sfreq, npad=self.npad)
        return downsampled

    def _decimate(self, data: RawArray, target_sfreq: float) -> RawArray:
        """
        Downsample using simple decimation (take every Nth sample).
        Fast but no anti-aliasing - may cause aliasing artifacts!
        """
        from mne import create_info

        # Calculate decimation factor
        decim_factor = int(np.round(self.original_sfreq / target_sfreq))

        # Get data and decimate
        data_array = data.get_data()  # (n_channels, n_times)
        decimated_data = data_array[:, ::decim_factor]

        # Calculate actual new sampling frequency
        new_sfreq = self.original_sfreq / decim_factor

        # Create new info
        new_info = data.info.copy()
        new_info['sfreq'] = new_sfreq

        # Create new RawArray
        downsampled = RawArray(decimated_data, new_info)

        return downsampled

    def _linear_interpolate(self, data: RawArray, target_sfreq: float) -> RawArray:
        """
        Downsample using linear interpolation.
        Better than decimation, faster than MNE resample.
        """
        from mne import create_info
        from scipy.interpolate import interp1d

        # Get data
        data_array = data.get_data()  # (n_channels, n_times)
        n_channels, n_times = data_array.shape

        # Calculate target number of points
        if self.num_points is not None:
            target_n_times = self.num_points
        else:
            target_n_times = int(n_times * target_sfreq / self.original_sfreq)

        # Create interpolation function
        old_times = np.arange(n_times)
        new_times = np.linspace(0, n_times - 1, target_n_times)

        # Interpolate each channel
        interpolated_data = np.zeros((n_channels, target_n_times))
        for ch_idx in range(n_channels):
            interp_func = interp1d(old_times, data_array[ch_idx, :], kind='linear')
            interpolated_data[ch_idx, :] = interp_func(new_times)

        # Calculate actual new sampling frequency
        new_sfreq = target_n_times * self.original_sfreq / n_times

        # Create new info
        new_info = data.info.copy()
        new_info['sfreq'] = new_sfreq

        # Create new RawArray
        downsampled = RawArray(interpolated_data, new_info)

        return downsampled

    def get_state(self) -> dict:
        """Get node state for serialization."""
        return {
            'target_sfreq': self.target_sfreq,
            'num_points': self.num_points,
            'method': self.method,
            'npad': self.npad
        }

    def get_config(self) -> dict:
        """Get node configuration for serialization."""
        return {
            'target_sfreq': self.target_sfreq,
            'num_points': self.num_points,
            'method': self.method,
            'npad': self.npad
        }

    def __str__(self):
        if self.target_sfreq:
            target_str = f"target_sfreq={self.target_sfreq}Hz"
        else:
            target_str = f"num_points={self.num_points}"

        return (f"{self.__class__.__name__}"
                f"({target_str}, method={self.method}, name={self.name})")

