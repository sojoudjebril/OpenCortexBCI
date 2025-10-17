import asyncio
import json
import logging
import threading
from typing import List, Set, Union
from mne.io import RawArray
import numpy as np
from websockets.legacy.server import WebSocketServerProtocol, serve

from opencortex.neuroengine.flux.base.node import Node


class WebSocketServer(Node):
    """
    Node that streams incoming data via WebSocket to connected clients.
    """

    def __init__(self, name: str = "WebSocketNode", host: str = "0.0.0.0", port: int = 8765,
                 channel_names: Union[List[str], None] = None, logger=None):
        super().__init__(name)
        self.host = host
        self.port = port
        self.channel_names = channel_names
        self.log = logger 
        self.clients: Set[WebSocketServerProtocol] = set()
        self.server = None
        self.logger = logger or logging.getLogger(__name__)

        # Store the main loop
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop; create one (for server)
            self.loop = asyncio.new_event_loop()
            threading.Thread(target=self._start_loop, daemon=True).start()
            
        # Start server in that loop
        asyncio.run_coroutine_threadsafe(self._start_server(), self.loop)

    def _start_loop(self):
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()

    async def _start_server(self):
        self.server = await serve(self._client_handler, self.host, self.port)
        self.logger.info(f"WebSocketNode: Server started at ws://{self.host}:{self.port}")

    async def _client_handler(self, websocket, path):
        self.clients.add(websocket)
        self.logger.info(f"WebSocketNode: Client connected ({websocket.remote_address})")
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)
            self.logger.info(f"WebSocketNode: Client disconnected ({websocket.remote_address})")

    async def _send_to_clients(self, message: str):
        if not self.clients:
            return


        self.logger.info(f"{self.name}: Sending data to clients: {message}")
        disconnected = set()
        for ws in self.clients:
            try:
                await ws.send(message)
            except Exception as e:
                self.logger.warning(f"WebSocketNode: Failed to send to client: {e}")
                disconnected.add(ws)

        self.clients -= disconnected

    def __call__(self, data):
        """
        Receives data and sends it over WebSocket.
        Returns data unchanged.
        """
        # Prepare data for sending
        if isinstance(data, RawArray):
            data = data.get_data().tolist()
        elif isinstance(data, np.ndarray):
            if data.ndim == 2:
                # shape: (n_channels, n_samples)
                data = data.tolist()
            elif data.ndim == 1:
                data = data.tolist()
        elif isinstance(data, dict):
            data = {k: float(v) if isinstance(v, (int, float)) else v for k, v in data.items()}
        elif isinstance(data, list):
            data = data
        else:
            raise ValueError("Unsupported data type for WebSocketNode")

        self.logger.info(f"{self.name}: Preparing to send data: {data}")

        # Wrap into a dictionary with optional channel names
        if isinstance(data, list) and self.channel_names and len(self.channel_names) == len(data):
            payload = dict(zip(self.channel_names, data))
            self.logger.info(f"{self.name}: Channel names provided, sending structured data: {payload}")
        else:
            payload = {"data": data}
        
        self.logger.info(f"{self.name}: Data prepared: {payload}")

        # Convert any numpy types to native Python types for JSON serialization. 
        # numpy int64 values are not JSON serializable by default and need to be converted to float or int.
        if isinstance(payload, dict):
            payload = {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v for k, v in payload.items()}

        payload = json.dumps(payload)


        self.logger.info(f"{self.name}: Payload prepared: {payload}")

        # Send safely to clients via main loop
        try:
            # self.logger.info(f"{self.name}: Sending data via WebSocket: {payload}")
            asyncio.run_coroutine_threadsafe(self._send_to_clients(payload), self.loop)
        except Exception as e:
            self.logger.error(f"{self.name}: Error sending data via WebSocket: {e}")

        return data  # Passthrough

    def get_config(self):
        return {
            "_target_": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "name": self.name,
            "host": self.host,
            "port": self.port,
            "channel_names": self.channel_names,
        }


