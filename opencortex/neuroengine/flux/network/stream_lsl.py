import numpy as np

from opencortex.neuroengine.flux.base.node import Node

from opencortex.neuroengine.network.lsl_stream import (
    start_lsl_eeg_stream, start_lsl_power_bands_stream,
    start_lsl_inference_stream, start_lsl_quality_stream,
    push_lsl_raw_eeg, push_lsl_band_powers, push_lsl_inference, push_lsl_quality
)


class StreamOutLSL(Node):
    """
    Node to stream data to LSL.
    Supports multiple data types: raw EEG, band powers, inference results, quality metrics.
    """

    def __init__(self, stream_type: str, name: str = None, **stream_params):
        """
        Args:
            stream_type: Type of data to stream ('eeg', 'band_powers', 'inference', 'quality')
            name: Name for this node
            stream_params: Additional parameters for the LSL stream (e.g., stream name, channel names)
        """
        super().__init__(name or f"StreamLSL_{stream_type}")
        self.stream_type = stream_type
        self.stream_params = stream_params

        # Initialize the appropriate LSL stream based on type
        if self.stream_type == 'eeg':
            self.outlet = start_lsl_eeg_stream(**self.stream_params)
            self.push_function = push_lsl_raw_eeg
        elif self.stream_type == 'band_powers':
            self.outlet = start_lsl_power_bands_stream(**self.stream_params)
            self.push_function = push_lsl_band_powers
        elif self.stream_type == 'inference':
            self.outlet = start_lsl_inference_stream(**self.stream_params)
            self.push_function = push_lsl_inference
        elif self.stream_type == 'quality':
            self.outlet = start_lsl_quality_stream(**self.stream_params)
            self.push_function = push_lsl_quality
        else:
            raise ValueError(f"Unsupported stream type: {self.stream_type}")

    def __call__(self, data):
        """
        Push data to the LSL outlet.

        Args:
            data: Data to be streamed (format depends on stream type)
        """
        # TODO make into a RawNode

        if self.stream_type == 'eeg':
            self.push_function(self.outlet, data, 0, len(data) - 1, 0)
        else:
            self.push_function(self.outlet, data)
        return data  # Pass through the data unchanged

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "_target_": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "stream_type": self.stream_type,
            "stream_params": self.stream_params,
            "name": self.name
        })
        return config
