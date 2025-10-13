"""
Notch Filter signal processing
"""
import logging
from typing import Union, List, Tuple
from mne.io import RawArray
from opencortex.neuroengine.flux.base.node import MNENode


class NotchFilterNode(MNENode):
    """
    A node that applies notch filtering to remove powerline noise.
    Typically used to remove 50 Hz or 60 Hz interference.
    """


    def __init__(
            self,
            freqs: Union[float, List[float], Tuple[float, ...]] = (50, 60),
            filter_length: str = 'auto',
            trans_bandwidth: float = 7.0,
            name: str = None
    ):
        """
        Initialize the NotchFilterNode.

        Args:
            freqs: Frequency or frequencies to notch filter (Hz).
                   Can be a single float or tuple/list of frequencies.
                   Default is (50, 60) for common powerline frequencies.
            filter_length: Length of the FIR filter. 'auto' for automatic selection.
            trans_bandwidth: Width of the transition band (Hz).
            name: Optional name for this node.
        """
        super().__init__(name or "NotchFilter")

        # Convert single freq to list
        if isinstance(freqs, (int, float)):
            self.freqs = [float(freqs)]
        else:
            self.freqs = list(freqs)

        self.filter_length = filter_length
        self.trans_bandwidth = trans_bandwidth
        logging.info(f"{self.name}: Applying notch filter at {self.freqs} Hz")


    def __call__(self, data: RawArray) -> RawArray:
        """
        Apply notch filter to the data.

        Args:
            data: MNE RawArray object

        Returns:
            Filtered MNE RawArray object
        """

        if not isinstance(data, RawArray):
            logging.error(f"{self.name}: Data must be of type RawArray, got {type(data)}")
            raise TypeError("Input data must be an instance of mne.io.RawArray")

        filtered = data.copy().notch_filter(
            freqs=self.freqs,
            filter_length=self.filter_length,
            trans_bandwidth=self.trans_bandwidth
        )
        return filtered


    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "freqs": self.freqs,
            "filter_length": self.filter_length,
            "trans_bandwidth": self.trans_bandwidth
        })
        return config

    def __str__(self):
        freqs_str = ', '.join(f"{f}Hz" for f in self.freqs)
        return f"{self.__class__.__name__}(freqs=[{freqs_str}], name={self.name})"