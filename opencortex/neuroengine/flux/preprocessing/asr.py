"""
ASR (Artifact Subspace Reconstruction) node for EEG artifact removal
"""
import numpy as np
import logging
from typing import Optional, Tuple
from mne.io import RawArray
from meegkit.asr import ASR as MeegkitASR
from opencortex.neuroengine.flux.base.node import MNENode


class ASRNode(MNENode):
    """
    A node that applies Artifact Subspace Reconstruction (ASR) to remove
    high-amplitude artifacts from EEG data.

    ASR identifies and removes brief high-amplitude artifacts by reconstructing
    the corrupted signal subspace using a statistical model learned from clean data.

    Reference:
    [1] Kothe, C. A. E., & Jung, T. P. (2016). U.S. Patent Application No. 14/895,440. https://patents.google.com/patent/US20160113587A1/en
    [2] Blum, S., Jacobsen, N. S. J., Bleichner, M. G., & Debener, S. (2019). A Riemannian Modification of Artifact Subspace Reconstruction for EEG Artifact Handling. Frontiers in Human Neuroscience, 13. https://doi.org/10.3389/fnhum.2019.00141
    """

    def __init__(
            self,
            sfreq: Optional[float] = None,
            cutoff: float = 5.0,
            block_size: int = 10,
            win_len: float = 0.5,
            win_overlap: float = 0.66,
            max_dimension: float = 0.66,
            calibration_time: Optional[float] = None,
            calibrate: bool = True,
            name: str = None
    ):
        """
        Initialize the ASRNode.

        Args:
            sfreq: Sampling frequency in Hz. If None, will be extracted from data.
            cutoff: Standard deviation cutoff for rejection. Lower values = more aggressive.
                   Recommended range: 5-20. Default is 5.
            block_size: Block size for calculating covariance (seconds).
            win_len: Window length for artifact detection (seconds).
            win_overlap: Fraction of overlap between windows (0-1).
            max_dimension: Maximum dimensionality to reconstruct (fraction of channels or int).
            calibration_time: Time in seconds to use for calibration. If None, uses all data.
            calibrate: If True, calibrate on initial segment; if False, uses the already fitted model.
            name: Optional name for this node.
        """
        super().__init__(name or "ASR")
        self.sfreq = sfreq
        self.cutoff = cutoff
        self.block_size = block_size
        self.win_len = win_len
        self.win_overlap = win_overlap
        self.max_dimension = max_dimension
        self.calibration_time = calibration_time
        self.calibrate = calibrate

        self.asr = None
        self.is_fitted = False

    def __call__(self, data: RawArray) -> RawArray:
        """
        Apply ASR to the data.

        Args:
            data: MNE RawArray object

        Returns:
            Cleaned MNE RawArray object
        """

        if not isinstance(data, RawArray):
            raise TypeError("Input data must be an instance of mne.io.RawArray")

        # Extract sampling frequency if not provided
        sfreq = self.sfreq if self.sfreq is not None else data.info['sfreq']

        # Get EEG data (shape: n_channels x n_samples)
        eeg_data = data.get_data(picks='eeg')

        # Determine calibration data
        if self.calibration_time is not None:
            n_calib_samples = int(self.calibration_time * sfreq)
            calib_data = eeg_data[:, :n_calib_samples]
        else:
            calib_data = eeg_data

        logging.info(
            f"Calibrating ASR with {calib_data.shape[1] / sfreq:.1f}s "
            f"({calib_data.shape[1]} samples) of data"
        )

        # Initialize and fit ASR
        self.asr = MeegkitASR(
            sfreq=sfreq,
            cutoff=self.cutoff,
            blocksize=self.block_size,
            win_len=self.win_len,
            win_overlap=self.win_overlap,
            max_dimension=self.max_dimension
        )

        # Fit on calibration data (transpose to samples x channels)
        if not self.calibrate and self.is_fitted:
            logging.info("ASR model already fitted; skipping calibration")
        else:
            self.asr.fit(calib_data)
            self.is_fitted = True

        # Transform all data (transpose to samples x channels)
        cleaned_data = self.asr.transform(eeg_data)

        # Transpose back to channels x samples
        #cleaned_data = cleaned_data.T

        logging.info(f"ASR cleaning complete")

        # Create new RawArray with cleaned data
        cleaned_raw = data.copy()
        cleaned_raw._data[data.ch_names.index(data.ch_names[0]):
                          data.ch_names.index(data.ch_names[0]) + len(eeg_data)] = cleaned_data

        return cleaned_raw


    def get_config(self) -> dict:
        return {
            "_target_": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "sfreq": self.sfreq,
            "cutoff": self.cutoff,
            "block_size": self.block_size,
            "win_len": self.win_len,
            "win_overlap": self.win_overlap,
            "max_dimension": self.max_dimension,
            "calibration_time": self.calibration_time,
            "calibrate": self.calibrate,
            "name": self.name
        }

    def __str__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        return (f"{self.__class__.__name__}"
                f"(cutoff={self.cutoff}, calib_time={self.calibration_time}s, "
                f"status={status}, name={self.name})")