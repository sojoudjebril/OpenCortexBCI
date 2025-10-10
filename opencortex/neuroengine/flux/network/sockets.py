"""
Streaming I/O nodes for LSL and WebSocket
"""
import logging
import asyncio
import json
import numpy as np
import websockets
from typing import Optional, Any, Callable
from mne.io import RawArray
from mne import create_info
from opencortex.neuroengine.flux.base.node import Node



class WebSocketOutNode(Node):
    """
    Node to send data via WebSocket.
    """

    def __init__(
            self,
            uri: str = "ws://localhost:8765",
            format: str = 'json',
            encoding: Optional[Callable] = None,
            name: str = None
    ):
        """
        Args:
            uri: WebSocket URI to connect to
            format: Data format ('json', 'msgpack', 'numpy', 'custom')
            encoding: Custom encoding function (data) -> bytes/str
            name: Name for this node
        """
        super().__init__(name or "WebSocketOut")


        self.uri = uri
        self.format = format
        self.encoding = encoding
        self.websocket = None
        self.loop = None

        logging.info(f"WebSocket output configured: {uri}")

    async def _connect(self):
        """Establish WebSocket connection."""
        if self.websocket is None:
            self.websocket = await websockets.connect(self.uri)
            logging.info(f"Connected to WebSocket: {self.uri}")

    async def _send(self, data: Any):
        """Send data via WebSocket."""
        await self._connect()

        # Encode data based on format
        if self.format == 'json':
            # Convert numpy arrays to lists for JSON serialization
            if isinstance(data, np.ndarray):
                data = {
                    'data': data.tolist(),
                    'shape': data.shape,
                    'dtype': str(data.dtype)
                }
            elif isinstance(data, RawArray):
                data = {
                    'data': data.get_data().tolist(),
                    'shape': data.get_data().shape,
                    'sfreq': data.info['sfreq'],
                    'ch_names': data.ch_names
                }

            message = json.dumps(data)
            await self.websocket.send(message)

        elif self.format == 'msgpack':
            import msgpack
            if isinstance(data, np.ndarray):
                message = msgpack.packb({
                    'data': data.tobytes(),
                    'shape': data.shape,
                    'dtype': str(data.dtype)
                })
            else:
                message = msgpack.packb(data)
            await self.websocket.send(message)

        elif self.format == 'numpy':
            # Send as raw bytes
            if isinstance(data, np.ndarray):
                message = data.tobytes()
            elif isinstance(data, RawArray):
                message = data.get_data().tobytes()
            else:
                raise ValueError(f"Cannot serialize {type(data)} as numpy format")
            await self.websocket.send(message)

        elif self.format == 'custom':
            if self.encoding is None:
                raise ValueError("encoding function must be provided for format='custom'")
            message = self.encoding(data)
            await self.websocket.send(message)

        else:
            raise ValueError(f"Unknown format: {self.format}")

        logging.debug(f"Sent {len(message) if isinstance(message, (str, bytes)) else 'data'} via WebSocket")

    def __call__(self, data: Any) -> Any:
        """
        Send data via WebSocket.

        Args:
            data: Data to send (numpy array, RawArray, dict, etc.)

        Returns:
            Original data (pass-through)
        """
        # Run async send in event loop
        if self.loop is None:
            try:
                self.loop = asyncio.get_event_loop()
            except RuntimeError:
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)

        self.loop.run_until_complete(self._send(data))

        return data  # Pass through

    def close(self):
        """Close WebSocket connection."""
        if self.websocket:
            self.loop.run_until_complete(self.websocket.close())
            logging.info("WebSocket connection closed")


    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "_target_": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "uri": self.uri,
            "format": self.format,
            "encoding": self.encoding,
            "name": self.name
        })
        return config

    def __str__(self):
        return f"{self.__class__.__name__}(uri={self.uri}, format={self.format})"


class WebSocketInNode(Node):
    """
    Node to receive data via WebSocket.
    """

    def __init__(
            self,
            uri: str = "ws://localhost:8765",
            format: str = 'json',
            decoding: Optional[Callable] = None,
            convert_to_raw: bool = False,
            timeout: Optional[float] = None,
            name: str = None
    ):
        """
        Args:
            uri: WebSocket URI to connect to
            format: Data format ('json', 'msgpack', 'numpy', 'custom')
            decoding: Custom decoding function (bytes/str) -> data
            convert_to_raw: If True, attempt to convert to MNE RawArray
            timeout: Timeout for receiving messages in seconds (None = wait forever)
            name: Name for this node
        """
        super().__init__(name or "WebSocketIn")


        self.uri = uri
        self.format = format
        self.decoding = decoding
        self.convert_to_raw = convert_to_raw
        self.timeout = timeout
        self.websocket = None
        self.loop = None

        logging.info(f"WebSocket input configured: {uri}")

    async def _connect(self):
        """Establish WebSocket connection."""
        if self.websocket is None:
            import websockets
            self.websocket = await websockets.connect(self.uri)
            logging.info(f"Connected to WebSocket: {self.uri}")

    async def _receive(self) -> Any:
        """Receive data from WebSocket."""
        await self._connect()

        # Receive message
        if self.timeout:
            message = await asyncio.wait_for(
                self.websocket.recv(),
                timeout=self.timeout
            )
        else:
            message = await self.websocket.recv()

        logging.debug(f"Received {len(message) if isinstance(message, (str, bytes)) else 'data'} via WebSocket")

        # Decode based on format
        if self.format == 'json':
            data = json.loads(message)

            # Reconstruct numpy array if needed
            if isinstance(data, dict) and 'data' in data and 'shape' in data:
                data = np.array(data['data']).reshape(data['shape'])

                if self.convert_to_raw and 'sfreq' in data and 'ch_names' in data:
                    info = create_info(
                        ch_names=data['ch_names'],
                        sfreq=data['sfreq'],
                        ch_types='eeg'
                    )
                    return RawArray(data, info)

            return data

        elif self.format == 'msgpack':
            import msgpack
            data = msgpack.unpackb(message)

            # Reconstruct numpy array
            if isinstance(data, dict) and 'data' in data:
                arr = np.frombuffer(
                    data['data'],
                    dtype=np.dtype(data['dtype'])
                ).reshape(data['shape'])
                return arr

            return data

        elif self.format == 'numpy':
            # Assume raw numpy bytes
            # Note: shape and dtype must be known in advance or transmitted separately
            return np.frombuffer(message)

        elif self.format == 'custom':
            if self.decoding is None:
                raise ValueError("decoding function must be provided for format='custom'")
            return self.decoding(message)

        else:
            raise ValueError(f"Unknown format: {self.format}")

    def __call__(self, data: Any = None) -> Any:
        """
        Receive data from WebSocket.

        Args:
            data: Ignored (for pipeline compatibility)

        Returns:
            Received data (numpy array, RawArray, dict, etc.)
        """
        # Run async receive in event loop
        if self.loop is None:
            try:
                self.loop = asyncio.get_event_loop()
            except RuntimeError:
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)

        received_data = self.loop.run_until_complete(self._receive())
        return received_data

    def close(self):
        """Close WebSocket connection."""
        if self.websocket:
            self.loop.run_until_complete(self.websocket.close())
            logging.info("WebSocket connection closed")


    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "_target_": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "uri": self.uri,
            "format": self.format,
            "decoding": self.decoding,
            "convert_to_raw": self.convert_to_raw,
            "timeout": self.timeout,
            "name": self.name
        })
        return config

    def __str__(self):
        return f"{self.__class__.__name__}(uri={self.uri}, format={self.format})"