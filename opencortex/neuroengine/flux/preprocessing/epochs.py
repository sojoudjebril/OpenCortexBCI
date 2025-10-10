import logging
import numpy as np
from mne.io import BaseRaw
from typing import Optional, Tuple, Dict, Union, List, Literal
from mne import Epochs, make_fixed_length_epochs
from opencortex.neuroengine.flux.base.node import Node


class EpochingNode(Node):
    """
    A node that creates epochs from continuous data.
    Supports event-based, fixed-length, and overlapping epochs.
    """

    def __init__(
            self,
            mode: Literal['events', 'fixed', 'fixed_overlap'] = 'events',
            # Event-based parameters
            tmin: float = -0.2,
            tmax: float = 0.8,
            baseline: Optional[Tuple[Optional[float], float]] = (-0.1, 0.0),
            event_id: Optional[Dict[str, int]] = None,
            # Fixed-length parameters
            duration: Optional[float] = None,
            overlap: float = 0.0,
            # Common parameters
            picks: Union[str, List[str]] = 'eeg',
            preload: bool = True,
            reject: Optional[Dict[str, float]] = None,
            name: str = None
    ):
        """
        Initialize the EpochingNode.

        Args:
            mode: Epoching mode:
                - 'events': Create epochs around events (requires events input)
                - 'fixed': Create fixed-length non-overlapping epochs
                - 'fixed_overlap': Create fixed-length overlapping epochs

            # Event-based parameters
            tmin: Start time before event (seconds, negative for before).
            tmax: End time after event (seconds).
            baseline: Baseline period (start, end) in seconds. None to skip baseline correction.
            event_id: Dictionary mapping event names to IDs (e.g., {'T': 1, 'NT': 3}).

            # Fixed-length parameters
            duration: Duration of each epoch in seconds (required for fixed modes).
            overlap: Overlap between epochs in seconds (only for 'fixed_overlap' mode).
                    For example, overlap=0.5 with duration=1.0 means 50% overlap.

            # Common parameters
            picks: Channels to include ('eeg', 'all', or list of channel names).
            preload: If True, load all epochs into memory.
            reject: Rejection criteria (e.g., {'eeg': 100e-6} for 100 µV threshold).
            name: Optional name for this node.
        """
        super().__init__(name or "Epoching")
        self.mode = mode

        # Event-based parameters
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline
        self.event_id = event_id if event_id is not None else {'T': 1, 'NT': 3}

        # Fixed-length parameters
        self.duration = duration
        self.overlap = overlap

        # Common parameters
        self.picks = picks
        self.preload = preload
        self.reject = reject

        # Validate parameters
        if self.mode in ['fixed', 'fixed_overlap'] and self.duration is None:
            raise ValueError(f"duration must be specified for mode='{self.mode}'")

        if self.mode == 'fixed_overlap' and self.overlap >= self.duration:
            raise ValueError(f"overlap ({self.overlap}s) must be less than duration ({self.duration}s)")

        self.epochs = None
        self.epochs_data = None
        self.labels = None

    def __call__(
            self,
            data: Union[
                Tuple[BaseRaw, np.ndarray, Dict[str, int], Dict[int, str]],  # Event-based
                BaseRaw  # Fixed-length
            ]
    ) -> Epochs:
        """
        Create epochs from continuous data.

        Args:
            data:
                - For 'events' mode: Tuple of (raw_data, events, event_ids, event_colors)
                - For 'fixed'/'fixed_overlap' modes: raw_data only

        Returns:
            Tuple of (epochs_data, labels)
            - epochs_data: ndarray of shape (n_epochs, n_channels, n_times)
            - labels: ndarray of shape (n_epochs,) - sequential indices for fixed-length epochs
        """

        if self.mode == 'events':
            return self._create_event_epochs(data)
        elif self.mode == 'fixed':
            return self._create_fixed_epochs(data, overlap=False)
        elif self.mode == 'fixed_overlap':
            return self._create_fixed_epochs(data, overlap=True)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _create_event_epochs(
            self,
            data: Tuple[BaseRaw, np.ndarray, Dict[str, int], Dict[int, str]]
    ) -> Epochs:
        """Create epochs based on events."""
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
            f"Created {self.epochs_data.shape[0]} event-based epochs with "
            f"{self.epochs_data.shape[1]} channels and "
            f"{self.epochs_data.shape[2]} time points"
        )


        return self.epochs

    def _create_fixed_epochs(
            self,
            data: BaseRaw,
            overlap: bool = False
    ) -> Epochs:
        """Create fixed-length epochs (with or without overlap)."""
        if not isinstance(data, BaseRaw):
            raise ValueError("For fixed-length epochs, data must be a BaseRaw instance, if you are relabeling events just remove the node")

        raw_data = data

        if overlap:
            # Calculate overlap for MNE (it expects the step size, not overlap)
            # overlap = duration - step_size
            # step_size = duration - overlap
            step_size = self.duration - self.overlap

            if step_size <= 0:
                raise ValueError(f"Invalid overlap: step_size={step_size}s must be positive")

            logging.info(
                f"Creating overlapping epochs: duration={self.duration}s, "
                f"overlap={self.overlap}s, step={step_size}s"
            )
        else:
            # No overlap: step_size = duration
            step_size = self.duration
            logging.info(f"Creating non-overlapping epochs: duration={self.duration}s")

        # Create fixed-length epochs using MNE
        self.epochs = make_fixed_length_epochs(
            raw_data,
            duration=self.duration,
            overlap=self.overlap if overlap else 0.0,
            preload=self.preload,
        )

        # Apply baseline correction if specified
        if self.baseline is not None:
            self.epochs.apply_baseline(self.baseline)

        # Extract data
        self.epochs_data = self.epochs.get_data(picks=self.picks)

        # Create sequential labels (no events for fixed-length epochs)
        self.labels = np.arange(len(self.epochs_data))

        # Calculate effective sampling rate
        n_epochs = self.epochs_data.shape[0]
        total_duration = raw_data.times[-1]
        coverage = n_epochs * self.duration
        if overlap:
            coverage = (n_epochs - 1) * step_size + self.duration

        logging.info(
            f"Created {n_epochs} fixed-length epochs with "
            f"{self.epochs_data.shape[1]} channels and "
            f"{self.epochs_data.shape[2]} time points "
            f"(coverage: {coverage:.1f}s / {total_duration:.1f}s)"
        )

        return self.epochs

    def get_state(self) -> Dict:
        """Get node state for serialization."""
        return {
            'mode': self.mode,
            'tmin': self.tmin,
            'tmax': self.tmax,
            'duration': self.duration,
            'overlap': self.overlap,
            'baseline': self.baseline,
            'event_id': self.event_id
        }

    def __str__(self):
        n_epochs = len(self.labels) if self.labels is not None else 0

        if self.mode == 'events':
            return (f"{self.__class__.__name__}"
                    f"(mode=events, tmin={self.tmin}, tmax={self.tmax}, "
                    f"n_epochs={n_epochs}, name={self.name})")
        elif self.mode == 'fixed':
            return (f"{self.__class__.__name__}"
                    f"(mode=fixed, duration={self.duration}s, "
                    f"n_epochs={n_epochs}, name={self.name})")
        else:  # fixed_overlap
            return (f"{self.__class__.__name__}"
                    f"(mode=fixed_overlap, duration={self.duration}s, "
                    f"overlap={self.overlap}s, n_epochs={n_epochs}, name={self.name})")