"""
BandPass Filter signal processing
"""
from typing import Union, List, Tuple
from mne.io import RawArray
from opencortex.neuroengine.flux.base.node import MNENode
import logging

class BandPassFilterNode(MNENode):
    """
    A node that applies band-pass filtering to retain frequencies
    within a specific range.
    """

    def __init__(
            self,
            l_freq: float = 1.0,
            h_freq: float = 30.0,
            filter_length: str = 'auto',
            l_trans_bandwidth: float = 1.0,
            h_trans_bandwidth: float = 3.0,
            name: str = None
    ):
        """
        Initialize the BandPassFilterNode.

        Args:
            l_freq: Low cutoff frequency (Hz). Frequencies below this are attenuated.
            h_freq: High cutoff frequency (Hz). Frequencies above this are attenuated.
            filter_length: Length of the FIR filter. 'auto' for automatic selection.
            l_trans_bandwidth: Width of the transition band at the low cut frequency (Hz).
            h_trans_bandwidth: Width of the transition band at the high cut frequency (Hz).
            name: Optional name for this node.
        """
        super().__init__(name or "BandPassFilter")
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.filter_length = filter_length
        self.l_trans_bandwidth = l_trans_bandwidth
        self.h_trans_bandwidth = h_trans_bandwidth

        if self.l_freq >= self.h_freq:
            raise ValueError("l_freq must be less than h_freq")
        if self.l_freq < 0 or self.h_freq < 0:
            raise ValueError("Cutoff frequencies must be non-negative")
        if l_freq - l_trans_bandwidth < 0:
            logging.warning(f"l_trans_bandwidth must be less than h_freq, setting it to l_freq:{self.l_freq}Hz")
            self.l_trans_bandwidth = l_freq
            
    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "l_freq": self.l_freq,
            "h_freq": self.h_freq,
            "filter_length": self.filter_length,
            "l_trans_bandwidth": self.l_trans_bandwidth,
            "h_trans_bandwidth": self.h_trans_bandwidth
        })
        return config

    def __call__(self, data: RawArray) -> RawArray:
        """
        Apply band-pass filter to the data.

        Args:
            data: MNE RawArray object

        Returns:
            Filtered MNE RawArray object
        """
        if not isinstance(data, RawArray):
            raise TypeError("Input data must be an instance of mne.io.RawArray")

        filtered = data.copy().filter(
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            filter_length=self.filter_length,
            l_trans_bandwidth=self.l_trans_bandwidth,
            h_trans_bandwidth=self.h_trans_bandwidth
        )
        return filtered

    def __str__(self):
        return (f"{self.__class__.__name__}"
                f"({self.l_freq}-{self.h_freq}Hz, name={self.name})")