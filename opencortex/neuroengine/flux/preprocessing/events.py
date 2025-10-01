"""
Event extraction node for EEG/BCI signal processing
"""
import logging
import numpy as np
from typing import Dict, Optional, Tuple, List
from mne.io import RawArray
from mne import find_events
from opencortex.neuroengine.flux.base.node import Node


class ExtractEventsNode(Node):
    """
    A node that extracts events from MNE RawArray data.
    Events are stored in the node and can be passed downstream.
    """

    def __init__(
            self,
            stim_channel: str = 'STI',
            ev_ids: Optional[Dict[str, int]] = None,
            event_color: Optional[Dict[int, str]] = None,
            auto_label: bool = True,
            initial_event: bool = True,
            shortest_event: int = 1,
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
            initial_event: If True, include the initial event.
            shortest_event: Minimum duration of an event in samples.
            name: Optional name for this node.
        """
        super().__init__(name or "ExtractEvents")
        self.stim_channel = stim_channel
        self.ev_ids = ev_ids if ev_ids is not None else {'NT': 1}
        self.event_color = event_color if event_color is not None else {1: 'r'}
        self.auto_label = auto_label
        self.initial_event = initial_event
        self.shortest_event = shortest_event

        # Store extracted events and metadata
        self.events = None
        self.events_dict = None
        self.colors_dict = None
        self.raw_data = None

    def __call__(self, data: RawArray) -> Tuple[RawArray, np.ndarray, Dict[str, int], Dict[int, str]]:
        """
        Extract events from the data and return both data and events.

        Args:
            data: MNE RawArray object

        Returns:
            Tuple of (raw_data, events, event_ids, event_colors)
        """
        # Extract events from stimulus channel
        self.events = find_events(
            data,
            stim_channel=self.stim_channel,
            initial_event=self.initial_event,
            shortest_event=self.shortest_event
        )
        self.raw_data = data

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

        logging.debug(
            f"Extracted {len(self.events)} events with IDs: {np.unique(self.events[:, 2])}"
        )

        return data, self.events, self.events_dict, self.colors_dict

    def __str__(self):
        n_events = len(self.events) if self.events is not None else 0
        return f"{self.__class__.__name__}(stim={self.stim_channel}, n_events={n_events}, name={self.name})"


class FilterEventsNode(Node):
    """
    A node that filters events based on criteria.
    """

    def __init__(
            self,
            max_event_id: Optional[int] = None,
            min_event_id: Optional[int] = None,
            keep_event_ids: Optional[List[int]] = None,
            drop_event_ids: Optional[List[int]] = None,
            name: str = None
    ):
        """
        Initialize the FilterEventsNode.

        Args:
            max_event_id: Drop events with IDs greater than this value.
            min_event_id: Drop events with IDs less than this value.
            keep_event_ids: Only keep events with these IDs.
            drop_event_ids: Drop events with these IDs.
            name: Optional name for this node.
        """
        super().__init__(name or "FilterEvents")
        self.max_event_id = max_event_id
        self.min_event_id = min_event_id
        self.keep_event_ids = keep_event_ids
        self.drop_event_ids = drop_event_ids

    def __call__(
            self,
            data: Tuple[RawArray, np.ndarray, Dict[str, int], Dict[int, str]]
    ) -> Tuple[RawArray, np.ndarray, Dict[str, int], Dict[int, str]]:
        """
        Filter events based on configured criteria.

        Args:
            data: Tuple of (raw_data, events, event_ids, event_colors)

        Returns:
            Tuple of (raw_data, filtered_events, event_ids, event_colors)
        """
        raw_data, events, event_ids, event_colors = data

        filtered = events.copy()
        original_count = len(filtered)

        # Filter by max event ID
        if self.max_event_id is not None:
            filtered = filtered[filtered[:, 2] <= self.max_event_id]

        # Filter by min event ID
        if self.min_event_id is not None:
            filtered = filtered[filtered[:, 2] >= self.min_event_id]

        # Keep only specific event IDs
        if self.keep_event_ids is not None:
            mask = np.isin(filtered[:, 2], self.keep_event_ids)
            filtered = filtered[mask]

        # Drop specific event IDs
        if self.drop_event_ids is not None:
            mask = ~np.isin(filtered[:, 2], self.drop_event_ids)
            filtered = filtered[mask]

        logging.debug(
            f"Filtered events: {original_count} -> {len(filtered)} "
            f"(IDs: {np.unique(filtered[:, 2])})"
        )

        return raw_data, filtered, event_ids, event_colors

    def __str__(self):
        filters = []
        if self.max_event_id: filters.append(f"max={self.max_event_id}")
        if self.min_event_id: filters.append(f"min={self.min_event_id}")
        filter_str = ", ".join(filters) if filters else "no filters"
        return f"{self.__class__.__name__}({filter_str}, name={self.name})"


class RelabelEventsNode(Node):
    """
    A node that relabels events based on a mapping or conditions.
    """

    def __init__(
            self,
            event_mapping: Optional[Dict[int, int]] = None,
            target_class: Optional[int] = None,
            nontarget_label: int = 3,
            name: str = None
    ):
        """
        Initialize the RelabelEventsNode.

        Args:
            event_mapping: Dictionary mapping old event IDs to new ones.
                          Example: {1: 1, 2: 3, 3: 3} (keep 1, relabel 2 and 3 to 3)
            target_class: If specified, keep this class as-is and relabel all others to nontarget_label.
            nontarget_label: Label to use for non-target events (default: 3).
            name: Optional name for this node.
        """
        super().__init__(name or "RelabelEvents")
        self.event_mapping = event_mapping
        self.target_class = target_class
        self.nontarget_label = nontarget_label

    def __call__(
            self,
            data: Tuple[RawArray, np.ndarray, Dict[str, int], Dict[int, str]]
    ) -> Tuple[RawArray, np.ndarray, Dict[str, int], Dict[int, str]]:
        """
        Relabel events.

        Args:
            data: Tuple of (raw_data, events, event_ids, event_colors)

        Returns:
            Tuple of (raw_data, relabeled_events, event_ids, event_colors)
        """
        raw_data, events, event_ids, event_colors = data

        relabeled = events.copy()

        if self.event_mapping is not None:
            # Use explicit mapping
            for old_id, new_id in self.event_mapping.items():
                relabeled[:, 2][relabeled[:, 2] == old_id] = new_id

        elif self.target_class is not None:
            # Binary classification: target vs non-target
            relabeled[:, 2][relabeled[:, 2] != self.target_class] = self.nontarget_label

        logging.debug(f"Relabeled events to: {np.unique(relabeled[:, 2])}")

        return raw_data, relabeled, event_ids, event_colors

    def __str__(self):
        return f"{self.__class__.__name__}(target={self.target_class}, name={self.name})"