import logging
import numpy as np
import logging
from typing import Union, List
from mne.io import RawArray
from typing import Union, List, Optional, Any
from mne.io import RawArray
from mne import create_info
from opencortex.neuroengine.flux.base.node import Node
from opencortex.neuroengine.flux.base.node import Node, MNENode
from opencortex.neuroengine.network.lsl_stream import (
    start_lsl_eeg_stream, start_lsl_power_bands_stream,
    start_lsl_inference_stream, start_lsl_quality_stream,
    push_lsl_raw_eeg, push_lsl_band_powers, push_lsl_inference, push_lsl_quality
)


class StreamOutLSL(MNENode, Node):
    """
    Node to stream data to LSL.
    Supports multiple data types: raw EEG, band powers, inference results, quality metrics.
    """

    def __init__(self, stream_type: str, name: str = None, picks:Union[str, List[str]] = None, **stream_params):
        """
        Args:
            stream_type: Type of data to stream ('eeg', 'band_powers', 'inference', 'quality')
            name: Name for this node
            picks: Channels to include ('eeg', 'all', or list of channel names).
            stream_params: Additional parameters for the LSL stream (e.g., stream name, channel names)
        """
        super().__init__(name or f"StreamLSL_{stream_type}")
        self.stream_type = stream_type
        self.picks = picks
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
        if not isinstance(data, (np.ndarray, RawArray, dict, list)):
            raise ValueError("Data must be: a numpy array, an MNE RawArray, a dict, or a list")

        if isinstance(data, RawArray):
            send_data = data.get_data(picks=self.picks)
            logging.log(5, f"Streaming RawArray data with shape {send_data.shape} to LSL as {self.stream_type}")
        else:
            send_data = data
            logging.log(5, f"Streaming data of type {type(send_data)} to LSL as {self.stream_type}")

        if self.stream_type == 'eeg':
            self.push_function(self.outlet, send_data, 0, len(send_data) - 1, 0)
        else:
            self.push_function(self.outlet, send_data)
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


class LSLInNode(Node):
    """
    Node to receive data from LSL streams.
    """


    def __init__(
            self,
            stream_name: Optional[str] = None,
            stream_type: Optional[str] = None,
            timeout: float = 5.0,
            max_samples: Optional[int] = None,
            buffer_size: int = 360,
            convert_to_raw: bool = True,
            sfreq: Optional[float] = None,
            ch_names: Optional[List[str]] = None,
            name: str = None
    ):
        """
        Args:
            stream_name: Name of LSL stream to connect to (None = any)
            stream_type: Type of LSL stream to connect to (None = any)
            timeout: Timeout for stream resolution in seconds
            max_samples: Maximum samples to pull per call (None = all available)
            buffer_size: Size of the inlet buffer in seconds
            convert_to_raw: If True, convert to MNE RawArray
            sfreq: Sampling frequency (required if convert_to_raw=True)
            ch_names: Channel names (required if convert_to_raw=True)
            name: Name for this node
        """
        super().__init__(name or "LSLIn")

        from pylsl import StreamInlet, resolve_bypred

        self.stream_name = stream_name
        self.stream_type = stream_type
        self.timeout = timeout
        self.max_samples = max_samples
        self.buffer_size = buffer_size
        self.convert_to_raw = convert_to_raw
        self.sfreq = sfreq
        self.ch_names = ch_names

        # Build predicate for stream resolution
        pred_parts = []
        if stream_name:
            pred_parts.append(f"name='{stream_name}'")
        if stream_type:
            pred_parts.append(f"type='{stream_type}'")

        predicate = " and ".join(pred_parts) if pred_parts else ""

        # Resolve and connect to stream
        logging.info(f"Resolving LSL stream: {predicate or 'any'}")
        streams = resolve_bypred(predicate, timeout=timeout)

        if not streams:
            raise RuntimeError(
                f"No LSL stream found matching: {predicate or 'any'} "
                f"(timeout={timeout}s)"
            )

        # Create inlet
        self.inlet = StreamInlet(streams[0], max_buflen=buffer_size)

        # Get stream info
        info = self.inlet.info()
        self.stream_sfreq = info.nominal_srate()
        self.stream_n_channels = info.channel_count()

        # Extract channel names from stream if not provided
        if not self.ch_names:
            self.ch_names = []
            ch = info.desc().child("channels").child("channel")
            for _ in range(self.stream_n_channels):
                self.ch_names.append(ch.child_value("label"))
                ch = ch.next_sibling()

            if not self.ch_names or len(self.ch_names) != self.stream_n_channels:
                self.ch_names = [f"Ch{i + 1}" for i in range(self.stream_n_channels)]

        # Use stream sfreq if not provided
        if not self.sfreq:
            self.sfreq = self.stream_sfreq

        logging.info(
            f"Connected to LSL stream: {info.name()} "
            f"({self.stream_n_channels} channels @ {self.stream_sfreq} Hz)"
        )

        if self.convert_to_raw and not self.sfreq:
            raise ValueError("sfreq must be provided when convert_to_raw=True")

    def __call__(self, data: Any = None) -> Union[np.ndarray, RawArray]:
        """
        Pull samples from LSL stream.

        Args:
            data: Ignored (for pipeline compatibility)

        Returns:
            - If convert_to_raw=True: MNE RawArray
            - If convert_to_raw=False: numpy array of shape (n_samples, n_channels)
        """
        # Pull samples
        if self.max_samples:
            samples, timestamps = self.inlet.pull_chunk(
                timeout=0.0,
                max_samples=self.max_samples
            )
        else:
            samples, timestamps = self.inlet.pull_chunk(timeout=0.0)

        if not samples:
            logging.warning("No samples available from LSL stream")
            return None

        # Convert to numpy array
        data_array = np.array(samples)  # Shape: (n_samples, n_channels)

        logging.debug(
            f"Pulled {len(samples)} samples from LSL "
            f"(shape: {data_array.shape})"
        )

        if self.convert_to_raw:
            # Transpose to (n_channels, n_samples) for MNE
            data_array = data_array.T

            # Create MNE info
            info = create_info(
                ch_names=self.ch_names,
                sfreq=self.sfreq,
                ch_types='eeg'
            )

            # Create RawArray
            raw = RawArray(data_array, info)
            return raw
        else:
            return data_array

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "_target_": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "stream_name": self.stream_name,
            "stream_type": self.stream_type,
            "timeout": self.timeout,
            "max_samples": self.max_samples,
            "buffer_size": self.buffer_size,
            "convert_to_raw": self.convert_to_raw,
            "sfreq": self.sfreq,
            "ch_names": self.ch_names,
            "name": self.name
        })
        return config

    def __str__(self):
        return (f"{self.__class__.__name__}"
                f"(stream={self.stream_name or 'any'}, "
                f"channels={self.stream_n_channels}, "
                f"sfreq={self.sfreq}Hz)")