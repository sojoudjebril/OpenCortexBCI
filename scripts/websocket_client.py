import asyncio
import websockets
import json
import signal
import sys
import argparse
import time
from datetime import datetime

RUNNING = True
MESSAGE_COUNT = 0
LAST_MSG_TIME = None

def signal_handler(sig, frame):
    global RUNNING
    print("\nInterrupted. Exiting...")
    RUNNING = False
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

async def receive_data(uri="ws://localhost:8765", verbose=False):
    global MESSAGE_COUNT, LAST_MSG_TIME

    try:
        async with websockets.connect(uri) as websocket:
            while RUNNING:
                try:
                    message = await websocket.recv()
                    now = time.time()
                    current_time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    time_diff = (now - LAST_MSG_TIME) if LAST_MSG_TIME else 0.0
                    LAST_MSG_TIME = now
                    MESSAGE_COUNT += 1

                    try:
                        data = json.loads(message)
                        data_type = type(data).__name__
                        data_len = len(data) if hasattr(data, "__len__") else "N/A"
                        data_keys = list(data.keys()) if isinstance(data, dict) else []
                    except json.JSONDecodeError:
                        data = message
                        data_type = "raw"
                        data_len = len(data)

                    # Inline printing
                    line = f"\r[{current_time_str}] Δt={time_diff:.3f}s | Total={MESSAGE_COUNT}"
                    if verbose:
                        line += f" | Type={data_type}, Len={data_len}, Keys={data_keys}"
                    print(line, end="", flush=True)

                except websockets.exceptions.ConnectionClosed:
                    print("\nConnection to server closed.")
                    break
    except Exception as e:
        print(f"\nConnection error: {e}")
        await asyncio.sleep(2)

async def main_loop(uri="ws://localhost:8765", verbose=False):
    while RUNNING:
        await receive_data(uri, verbose=verbose)
        await asyncio.sleep(1)

def main():
    parser = argparse.ArgumentParser(description="WebSocket Data Receiver")
    parser.add_argument("--uri", type=str, default="ws://localhost:8765", help="WebSocket server URI")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show verbose message info")
    args = parser.parse_args()

    try:
        print(f"Connecting to WebSocket server at {args.uri}...")
        print("Press Ctrl+C to exit.")
        asyncio.run(main_loop(args.uri, verbose=args.verbose))
    except KeyboardInterrupt:
        print("\nShutdown requested by user.")

if __name__ == "__main__":
    main()
