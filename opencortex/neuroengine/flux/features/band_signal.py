import numpy as np
from mne.io import RawArray
from opencortex.neuroengine.flux.base.node import Node

from scipy.signal import butter, lfilter

def butter_bandpass(low_cut, high_cut, fs, order=5):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

class BandSignalExtractor(Node):
    """
        Extracts band signals from raw EEG data.
        Input shape: (channels, samples)
        Output shape: (bands, channels, samples)
    """

    def __init__(self, fs: int, ch_names: list, name: str = None, freq_bands: dict = None, order: int = 5, picks: list = "eeg"):
        super().__init__(name or "BandSignalExtractor")
        self.fs = fs
        self.ch_names = ch_names
        self.freq_bands = freq_bands or {
            "theta": (4, 8),
            "alpha": (8, 14),
            "beta": (14, 31),
            "gamma": (31, 49)
        }
        self.order = order
        self.picks = picks

    def __call__(self, data):
        try:
            if isinstance(data, RawArray):
                data = data.get_data(picks=self.picks)
            band_signals = []
            for band, (low_freq, high_freq) in self.freq_bands.items():
                # Bandpass filter the data for each frequency bands
                band_signal = self.bandpass_filter(data, low_freq, high_freq)
                band_signals.append(band_signal)
            return np.array(band_signals)
        except Exception as e:
            print(f"{self.name}: Error extracting band signals - {e}")
            raise e

    def bandpass_filter(self, data, low_freq, high_freq):
        b, a = butter_bandpass(low_freq, high_freq, fs=self.fs, order=self.order)
        return lfilter(b, a, data)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "_target_": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "fs": self.fs,
            "ch_names": self.ch_names,
            "freq_bands": self.freq_bands,
            "name": self.name
        })
        return config