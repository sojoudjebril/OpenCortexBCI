"""
LSL to WebSocket Bridge for Quest 3 Standalone
Receives cognitive state data from LSL and broadcasts it via WebSocket
"""
import asyncio
import json
import time
import argparse
import pylsl
import websockets
from typing import Set, Optional


class LSLWebSocketBridge:
    def __init__(self, lsl_stream_name: str, websocket_port: int, host: str = "0.0.0.0"):
        self.lsl_stream_name = lsl_stream_name
        self.websocket_port = websocket_port
        self.host = host
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.inlet: Optional[pylsl.StreamInlet] = None
        self.channel_names = []
        self.running = False
        
    async def register_client(self, websocket):
        """Register a new WebSocket client"""
        self.clients.add(websocket)
        client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        print(f"[+] Client connected: {client_info} (Total clients: {len(self.clients)})")
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)
            print(f"[-] Client disconnected: {client_info} (Total clients: {len(self.clients)})")
    
    def init_lsl_stream(self):
        """Initialize connection to LSL stream"""
        print(f"\n🔍 Searching for LSL stream: '{self.lsl_stream_name}'...")
        streams = pylsl.resolve_byprop('name', self.lsl_stream_name, timeout=5.0)
        
        if not streams:
            raise RuntimeError(f"❌ No LSL stream found with name '{self.lsl_stream_name}'")
        
        print(f"✅ Found stream: {self.lsl_stream_name}")
        self.inlet = pylsl.StreamInlet(streams[0])
        
        # Get channel information
        info = self.inlet.info()
        channel_count = info.channel_count()
        print(f"   Channels: {channel_count}")
        print(f"   Sample Rate: {info.nominal_srate()} Hz")
        print(f"   Type: {info.type()}")
        
        # Extract channel names from XML description
        channels = info.desc().child("channels")
        if channels.empty():
            # Default channel names
            self.channel_names = [f"channel_{i}" for i in range(channel_count)]
        else:
            ch = channels.child("channel")
            self.channel_names = []
            while not ch.empty():
                label = ch.child_value("label")
                self.channel_names.append(label if label else f"channel_{len(self.channel_names)}")
                ch = ch.next_sibling("channel")
        
        print(f"   Channel Names: {self.channel_names}")
        
    async def broadcast_lsl_data(self):
        """Continuously read from LSL and broadcast to all WebSocket clients"""
        if not self.inlet:
            self.init_lsl_stream()
        
        print("\n📡 Starting data relay...")
        sample_count = 0
        
        while self.running:
            try:
                # Pull sample from LSL (non-blocking with timeout)
                sample, timestamp = self.inlet.pull_sample(timeout=0.0)
                
                if sample:
                    sample_count += 1
                    
                    # Create JSON message
                    message = {
                        "timestamp": timestamp,
                        "data": {name: float(value) for name, value in zip(self.channel_names, sample)},
                        "sample_count": sample_count
                    }
                    
                    # Broadcast to all connected clients
                    if self.clients:
                        json_message = json.dumps(message)
                        # Use websockets.broadcast for efficient broadcasting
                        websockets.broadcast(self.clients, json_message)
                        
                        if sample_count % 100 == 0:  # Log every 100 samples
                            print(f"📊 Samples sent: {sample_count} | Active clients: {len(self.clients)}")
                    
                else:
                    # No data available, yield control briefly
                    await asyncio.sleep(0.001)
                    
            except Exception as e:
                print(f"❌ Error reading/broadcasting data: {e}")
                await asyncio.sleep(0.1)
    
    async def start_server(self):
        """Start WebSocket server and LSL data relay"""
        self.running = True
        
        # Initialize LSL connection
        try:
            self.init_lsl_stream()
        except Exception as e:
            print(f"❌ Failed to initialize LSL stream: {e}")
            return
        
        # Start WebSocket server
        print(f"\n🚀 Starting WebSocket server on {self.host}:{self.websocket_port}")
        
        async with websockets.serve(self.register_client, self.host, self.websocket_port):
            print(f"✅ WebSocket server ready!")
            print(f"   Connect to: ws://{self.host}:{self.websocket_port}")
            print(f"\n⏳ Waiting for clients...")
            print("   Press Ctrl+C to stop.\n")
            
            # Start broadcasting LSL data
            await self.broadcast_lsl_data()
    
    def stop(self):
        """Stop the bridge"""
        self.running = False
        if self.inlet:
            self.inlet.close_stream()


async def main():
    parser = argparse.ArgumentParser(description="LSL to WebSocket Bridge for Quest 3")
    parser.add_argument(
        "--stream",
        type=str,
        default="CortexInference",
        help="Name of the LSL stream to bridge (default: CortexInference)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8788,
        help="WebSocket server port (default: 8788)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind to (default: 0.0.0.0 for all interfaces)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("LSL → WebSocket Bridge for Meta Quest 3")
    print("=" * 70)
    
    # Check dependencies
    try:
        import pylsl
        import websockets
        print(f"✅ pylsl version: {pylsl.__version__}")
        print(f"✅ websockets version: {websockets.__version__}")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nInstall with: pip install pylsl websockets")
        return
    
    bridge = LSLWebSocketBridge(
        lsl_stream_name=args.stream,
        websocket_port=args.port,
        host=args.host
    )
    
    try:
        await bridge.start_server()
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping bridge...")
        bridge.stop()
        print("✅ Bridge stopped successfully")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        bridge.stop()


if __name__ == "__main__":
    asyncio.run(main())