"""
Event extraction node for EEG/BCI signal processing
"""
import numpy as np
from typing import Dict, Optional, Tuple
from mne.io import RawArray
from mne import find_events
from opencortex.neuroengine.flux.base.node import Node


class ExtractEventsNode(Node):
    """
    A node that extracts events from MNE RawArray data.
    Events are typically stored in a stimulus channel and represent
    experimental markers (e.g., stimulus onset, responses).
    """

    def __init__(
            self,
            stim_channel: str = 'STI',
            ev_ids: Optional[Dict[str, int]] = None,
            event_color: Optional[Dict[int, str]] = None,
            auto_label: bool = True,
            name: str = None
    ):
        """
        Initialize the ExtractEventsNode.

        Args:
            stim_channel: Name of the stimulus channel containing event markers.
            ev_ids: Dictionary mapping event names to their IDs.
                    Default is {'NT': 1} (non-target).
            event_color: Dictionary mapping event IDs to colors for plotting.
                         Default is {1: 'r'} (red for non-target).
            auto_label: If True, automatically create labels for additional events
                       found in the data (T1, T2, T3, etc.).
            name: Optional name for this node.
        """
        super().__init__(name or "ExtractEvents")
        self.stim_channel = stim_channel
        self.ev_ids = ev_ids if ev_ids is not None else {'NT': 1}
        self.event_color = event_color if event_color is not None else {1: 'r'}
        self.auto_label = auto_label

        # Store extracted events
        self.events = None
        self.events_dict = None
        self.colors_dict = None

    def __call__(self, data: RawArray) -> RawArray:
        """
        Extract events from the data and store them in the node.
        The RawArray is returned unchanged (events are metadata).

        Args:
            data: MNE RawArray object

        Returns:
            Original MNE RawArray object (unchanged)
        """
        # Extract events from stimulus channel
        self.events = find_events(data, stim_channel=self.stim_channel)

        # Copy the dictionaries to avoid modifying the originals
        self.events_dict = self.ev_ids.copy()
        self.colors_dict = self.event_color.copy()

        # Auto-label additional events if requested
        if self.auto_label:
            labels = np.unique(self.events[:, 2])
            for i in range(1, len(labels)):
                label_id = i + 1
                if label_id not in self.events_dict.values():
                    self.events_dict[f'T{i}'] = label_id
                    self.colors_dict[label_id] = 'g'  # Green for targets

        return data

    def get_events(self) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
        """
        Get the extracted events, event IDs, and colors.

        Returns:
            Tuple of (events array, event_ids dict, event_colors dict)
        """
        if self.events is None:
            raise ValueError("No events extracted yet. Call the node with data first.")
        return self.events, self.events_dict, self.colors_dict

    def __str__(self):
        n_events = len(self.events) if self.events is not None else 0
        return (f"{self.__class__.__name__}"
                f"(stim={self.stim_channel}, n_events={n_events}, name={self.name})")