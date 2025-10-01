import logging
from multiprocessing import RawArray
from typing import Optional, Tuple, Dict, Union, List

import numpy as np
from mne import Epochs

from opencortex.neuroengine.flux.base.node import Node


class EpochingNode(Node):
    """
    A node that creates epochs from continuous data based on events.
    """

    def __init__(
            self,
            tmin: float = -0.2,
            tmax: float = 0.8,
            baseline: Optional[Tuple[Optional[float], float]] = (-0.1, 0.0),
            event_id: Optional[Dict[str, int]] = None,
            picks: Union[str, List[str]] = 'eeg',
            preload: bool = True,
            reject: Optional[Dict[str, float]] = None,
            name: str = None
    ):
        """
        Initialize the EpochingNode.

        Args:
            tmin: Start time before event (seconds, negative for before).
            tmax: End time after event (seconds).
            baseline: Baseline period (start, end) in seconds. None to skip baseline correction.
            event_id: Dictionary mapping event names to IDs (e.g., {'T': 1, 'NT': 3}).
            picks: Channels to include ('eeg', 'all', or list of channel names).
            preload: If True, load all epochs into memory.
            reject: Rejection criteria (e.g., {'eeg': 100e-6} for 100 µV threshold).
            name: Optional name for this node.
        """
        super().__init__(name or "Epoching")
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline
        self.event_id = event_id if event_id is not None else {'T': 1, 'NT': 3}
        self.picks = picks
        self.preload = preload
        self.reject = reject

        self.epochs = None
        self.epochs_data = None
        self.labels = None

    def __call__(
            self,
            data: Tuple[RawArray, np.ndarray, Dict[str, int], Dict[int, str]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create epochs from continuous data and events.

        Args:
            data: Tuple of (raw_data, events, event_ids, event_colors)

        Returns:
            Tuple of (epochs_data, labels)
            - epochs_data: ndarray of shape (n_epochs, n_channels, n_times)
            - labels: ndarray of shape (n_epochs,)
        """
        raw_data, events, event_ids, event_colors = data

        # Create MNE Epochs object
        self.epochs = Epochs(
            raw_data,
            events,
            event_id=self.event_id,
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=self.baseline,
            picks=self.picks,
            preload=self.preload,
            reject=self.reject
        )

        # Extract data and labels
        self.epochs_data = self.epochs.get_data(picks=self.picks)
        self.labels = self.epochs.events[:, -1]

        logging.info(
            f"Created {self.epochs_data.shape[0]} epochs with "
            f"{self.epochs_data.shape[1]} channels and "
            f"{self.epochs_data.shape[2]} time points"
        )

        return self.epochs_data, self.labels

    def __str__(self):
        n_epochs = len(self.labels) if self.labels is not None else 0
        return (f"{self.__class__.__name__}"
                f"(tmin={self.tmin}, tmax={self.tmax}, n_epochs={n_epochs}, name={self.name})")