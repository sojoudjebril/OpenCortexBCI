"""
CortexEngine - Main application controller and independent service

Authors: Michele Romani, Gregorio Andrea Giudici
"""

import json
import os
from pathlib import Path
import socket
import socket
import threading
import time
import uuid
import numpy as np
import logging
import queue
import onnxruntime as ort
import sys

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor
from brainflow.board_shim import BoardShim, BoardIds
import portalocker  # For safe file locking

from pythonosc import osc_server
from pythonosc.dispatcher import Dispatcher
from sklearn.preprocessing import LabelEncoder, StandardScaler

import opencortex.neuroengine.flux.base.operators  # Needed to enable >> and + operators
from opencortex.neuroengine.flux.base.aggregate import Aggregate
from opencortex.neuroengine.flux.base.parallel import Parallel
from opencortex.neuroengine.flux.base.sequential import Sequential
from opencortex.neuroengine.flux.estimation.lightning import LightningNode
from opencortex.neuroengine.flux.estimation.onnx import ONNXNode
from opencortex.neuroengine.flux.features.band_power import BandPowerExtractor
from opencortex.neuroengine.flux.features.quality_estimator import QualityEstimator
from opencortex.neuroengine.flux.network.sockets import WebSocketOutNode
from opencortex.neuroengine.flux.network.websockets import WebSocketServer

from opencortex.neuroengine.flux.pipeline_config import PipelineConfig
from opencortex.neuroengine.flux.pipeline_group import PipelineGroup
from opencortex.neuroengine.flux.preprocessing.bandpass import BandPassFilterNode
from opencortex.neuroengine.flux.preprocessing.dataset import DatasetNode
from opencortex.neuroengine.flux.preprocessing.epochs import EpochingNode
from opencortex.neuroengine.flux.preprocessing.extract import ExtractNode
from opencortex.neuroengine.flux.preprocessing.notch import NotchFilterNode
from opencortex.neuroengine.flux.base.simple_nodes import LogNode
from opencortex.neuroengine.flux.network.lsl import StreamOutLSL
from opencortex.neuroengine.flux.preprocessing.scaler import ScalerNode
from opencortex.neuroengine.network.lsl_stream import (
    start_lsl_eeg_stream, start_lsl_power_bands_stream,
    start_lsl_inference_stream, start_lsl_quality_stream,
    push_lsl_raw_eeg, push_lsl_inference, push_lsl_quality
)
from opencortex.utils.layouts import layouts
from opencortex.utils.loader import convert_to_mne


@dataclass
class StreamData:
    """Data packet sent to interfaces"""
    raw_eeg: np.ndarray
    filtered_eeg: np.ndarray
    band_powers: np.ndarray
    quality_scores: list
    timestamp: float
    trigger: int


@dataclass
class Command:
    """Command sent to StreamEngine"""
    action: str  # 'set_inference_mode', 'train', 'send_trigger', etc.
    params: Dict[str, Any]
    callback: Optional[Callable] = None


class CortexEngine:
    """
    Main application controller that runs independently.
    Can operate headless or with GUI attached.
    """

    DEFAULT_PORT = 8080
    INSTANCE_REGISTRY_FILE = "cortex_engine_registry.json"

    def __init__(self, board, config, window_size=2):
        self.board = board
        self.config = config
        self.window_size = window_size
        log = logging.getLogger()
        log.setLevel(logging.INFO)

        self.pid = os.getpid()
        self.hostname = socket.gethostname()
        self.uuid = f"{self.hostname}_{self.pid}_{str(uuid.uuid4())[:4]}"
        self.instance_id = self.uuid
        self.assigned_port = None
        self.board_id = self.board.get_board_id()
        logging.info(f"Starting CortexEngine with PID {self.uuid} and board {self.board_id}")

        # self.instances_file = Path(__file__).parent / self.INSTANCE_REGISTRY_FILE
        self.instances_file = Path.cwd() / self.INSTANCE_REGISTRY_FILE

        # Register instance for discovery
        self._register_instance()

        # Create a osc server on assigned port
        dispatcher = Dispatcher()
        dispatcher.map("/filter", print)
        # Create a server listening on the assigned port for OSC messages from everywhere
        self.osc_server = osc_server.ThreadingOSCUDPServer(
            ("0.0.0.0", self.assigned_port), dispatcher)
        logging.info(f"OSC server listening on port {self.assigned_port}")
        self.osc_thread = threading.Thread(
            target=self.osc_server.serve_forever,
            daemon=True  # Ensures thread won't block shutdown
        )
        self.osc_thread.start()

        # Cache frequently accessed board properties
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        try:
            self.eeg_names = BoardShim.get_eeg_names(self.board_id)
        except Exception as e:
            log.warning("Could not get EEG channels, using default 8 channels, caused by: {}".format(e))
            self.eeg_names = ["CPz", "P1", "Pz", "P2", "PO3", "POz", "PO4", "Oz"]
        self.eeg_channels_len = len(self.eeg_channels)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.num_points = self.window_size * self.sampling_rate
        self.ts_channel = self.board.get_timestamp_channel(self.board_id)
        self.update_interval_ms = config.get('update_buffer_speed_ms', 50)

        # Engine state
        self.running = False
        self.inference_mode = False
        self.first_prediction = True

        # Configuration
        band_power_pipeline = BandPowerExtractor(fs=self.sampling_rate, ch_names=self.eeg_names) \
                       >> Parallel(
            lsl=StreamOutLSL(stream_type='band_powers', name='BandPowerLSL', channels=self.eeg_names,
                             fs=self.sampling_rate, source_id=board.get_device_name(self.board_id)),
            socket=WebSocketServer(
                name="WebSocketServerBandPowers",
                host="0.0.0.0",
                port=8766,
                channel_names=self.eeg_names,
                logger=log
            ))
        
        # # TODO uniform IN and OUT of nodes, add error checking and/or handling
        # # TODO specialize Node in RawNode and EpochNode (NumPy node?)

        signal_quality_pipeline = Sequential(
            StreamOutLSL(stream_type="eeg", name="CortexEEG", channels=self.eeg_names + ["Trigger"],
                         fs=self.sampling_rate,
                         source_id=self.board.get_device_name(self.board_id)),
            NotchFilterNode((50, 60), name='NotchFilter'),
            BandPassFilterNode(0.1, 30.0, name='BandPassFilter'),
            QualityEstimator(quality_thresholds=config.get('quality_thresholds',
                                             [(-100, -50, 'yellow', 0.5), (-50, 50, 'green', 1.0),
                                              (50, 100, 'yellow', 0.5)]), name='QualityEstimator'),
            Parallel(
                lsl=StreamOutLSL(stream_type='quality', name='QualityLSL', channels=self.eeg_names,
                                 fs=self.sampling_rate, source_id=board.get_device_name(self.board_id)),
                socket=WebSocketServer(
                    name="WebSocketServerQuality",
                    host="0.0.0.0",
                    port=8765,
                    channel_names=self.eeg_names,
                    logger=log
                )),
            name='SignalQualityPipeline'
        )
        
        if hasattr(sys, '_MEIPASS'):
            base_path = os.path.join(sys._MEIPASS, "models")
        else:
            base_path = os.path.abspath(".")

        model_path = os.path.join(base_path, "model.onnx")
        self.onnx_session = ort.InferenceSession(model_path)

        classification_pipeline = Sequential(
            NotchFilterNode((50, 60), name="PowerlineNotch"),
            BandPassFilterNode(0.1, 30.0, name="ERPBand"),
            EpochingNode(
                mode='fixed_overlap',
                duration=1.0,  # 2 second windows
                overlap=.5,  # 1 second overlap (50%)
                baseline=None,  # No baseline
                name='OverlapEpochs'
            ),
            ExtractNode(label_encoder=LabelEncoder(), apply_label_encoding=True, label_mapping={1: 0, 3: 1},
                        picks=['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8'],
                        name='XyExtractor'),
            ScalerNode(scaler=StandardScaler(), per_channel=True, name='StdScaler'),
            DatasetNode(split_size=0.0, batch_size=1, shuffle=False, num_workers=0, name='TestDataset'),
            Parallel(
            model_1=ONNXNode(model_path=model_path, session=self.onnx_session, name='ONNXInference'),
            model_2=ONNXNode(model_path=model_path, session=self.onnx_session, name='ONNXInference2'),
            model_3=ONNXNode(model_path=model_path, session=self.onnx_session, name='ONNXInference3'),
            model_4=ONNXNode(model_path=model_path, session=self.onnx_session, name='ONNXInference4'),
            ),
            Aggregate(mode="list", name="AggregatePredictions"),
            Parallel(
                # lsl=StreamOutLSL(stream_type='inference', name='InferenceLSL',
                #                  channels=["Arousal", "Valence", "Metal Load", "Calmness"],
                #                  logger=log,
                #                  fs=self.sampling_rate,
                #                  source_id=board.get_device_name(self.board_id)),
                socket=WebSocketServer(
                    name="WebSocketServerInference",
                    host="0.0.0.0",
                    port=8767,
                    channel_names=["Arousal", "Valence", "Focus", "Calm"],
                    logger=log
                )
            ),
            name="Inference",

        )
        )

        configs = [
            PipelineConfig(
                pipeline=band_power_pipeline,
                name="LSLStream"
            ),
            PipelineConfig(
                pipeline=signal_quality_pipeline,
                name="SignalQuality"
            ),
            PipelineConfig(
                pipeline=classification_pipeline,
                name="Classifier"
            )

        ]

        self.pipeline = PipelineGroup(
            pipelines=configs,
            name="CortexEnginePipeline",
            max_workers=16,
            wait_for_all=True
        )

        # Data buffers
        self.filtered_eeg = np.zeros((len(self.eeg_channels) + 1, self.num_points))
        self.raw_data = None

        # Threading and communication
        self.command_queue = queue.Queue()
        self.main_thread = None

        # Interface callbacks (GUI, API, etc.)
        self.data_callbacks = []
        self.event_callbacks = []

        logging.info("StreamEngine initialized")


        # ===================== MAIN ENGINE CONTROL =====================

    def start(self):
        """Start the StreamEngine main loop."""
        if self.running:
            logging.warning("StreamEngine already running")
            return

        self.running = True

        # Start main processing loop
        self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
        self.main_thread.start()

        logging.info("StreamEngine started")

    def stop(self):
        """Stop the StreamEngine."""
        self.running = False

        # Remove all callbacks
        self.data_callbacks.clear()
        self.event_callbacks.clear()

        # Stop OSC server
        if self.osc_server:
            self.osc_server.shutdown()
            self.osc_server.server_close()
            logging.info("OSC server stopped")

        # Unregister instance
        self._unregister_instance()

        self._cleanup_stale_instances(self._load_instances())

        if self.main_thread:
            self.main_thread.join(timeout=5.0)
        logging.info("StreamEngine stopped")

        if hasattr(self, 'pipeline') and hasattr(self.pipeline, 'shutdown'):
            self.pipeline.shutdown()


    def _main_loop(self):
        """Main processing loop - runs independently."""
        last_update = time.perf_counter()
        update_interval = self.update_interval_ms / 1000.0  # Convert ms to seconds

        while self.running:
            current_time = time.perf_counter()

            # Process commands from interfaces
            self._process_commands()

            # Update data at configured interval
            if (current_time - last_update) >= update_interval:
                try:
                    self._update_data()
                    last_update = current_time
                except Exception as e:
                    logging.error(f"Error in main loop: {e}")

            # Adaptive sleep (sleep exactly the remaining time)
            elapsed = time.perf_counter() - current_time
            sleep_time = max(0, update_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _update_data(self):
        """Core data processing - heart of the engine."""
        try:
            # Cache attributes locally (faster access)
            eeg_channels = self.eeg_channels
            eeg_channels_len = self.eeg_channels_len
            num_points = self.num_points
            sampling_rate = self.sampling_rate
            
            # Get raw data from board
            data = self.board.get_current_board_data(num_samples=num_points)

            # Extract and filter EEG
            start_eeg = layouts[self.board_id]["eeg_start"]
            end_eeg = layouts[self.board_id]["eeg_end"]
            eeg = data[start_eeg:end_eeg]

            # Update filtered buffer
            self.filtered_eeg[:eeg_channels_len] = eeg

            # Extract trigger channel
            trigger = data[-1]
            self.filtered_eeg[-1] = trigger

            # Process through pipeline
            raw = self.filtered_eeg[0:len(self.eeg_channels)].T
            raw = convert_to_mne(eeg.T, trigger, fs=self.sampling_rate, chs=self.eeg_names, recompute=False)
            _ = self.pipeline(raw)


        except Exception as e:
            logging.error(f"Error updating data: {e}")
            raise

    # ===================== INSTANCE REGISTRATION =====================
    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available for both TCP and UDP."""
        # Check TCP
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
        except OSError:
            return False

        # Check UDP (OSC uses UDP)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.bind(('', port))
        except OSError:
            return False

        return True

    def _load_instances(self) -> Dict[str, Dict]:
        """Load existing instances from file."""
        if not self.instances_file.exists():
            return {}

        try:
            with open(self.instances_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def _save_instances(self, instances: Dict[str, Dict]) -> None:
        """Save instances to file."""
        with open(self.instances_file, 'w') as f:
            json.dump(instances, f, indent=2)

    def _cleanup_stale_instances(self, instances: Dict[str, Dict]) -> Dict[str, Dict]:
        """Remove instances that no longer have their ports in use."""
        cleaned = {}
        for inst_id, data in instances.items():
            port = data.get('port')
            if port and not self._is_port_available(port):
                # Port is in use, keep the instance
                cleaned[inst_id] = data
        return cleaned

    def _find_next_available_port(self, start_port: int, used_ports: list[int]) -> int:
        """Find the next available port starting from start_port."""
        port = start_port
        max_attempts = 100

        for _ in range(max_attempts):
            if port not in used_ports and self._is_port_available(port):
                return port
            port += 1

        raise RuntimeError(f"Could not find available port after {max_attempts} attempts")

    def _register_instance(self, preferred_port: Optional[int] = None) -> int:
        """
        Register this engine instance and get an assigned port.

        Args:
            preferred_port: Desired port number. If None, uses DEFAULT_PORT.
                          If port is taken, finds next available.

        Returns:
            The assigned port number.
        """
        # Acquire exclusive lock on the instances file
        lock_file = self.instances_file.with_suffix('.lock')

        with portalocker.Lock(lock_file, mode='a', timeout=10) as _:
            # Load and cleanup existing instances
            instances = self._load_instances()
            instances = self._cleanup_stale_instances(instances)

            # Determine which ports are in use
            used_ports = [data['port'] for data in instances.values()]

            # Determine the port to use
            start_port = preferred_port if preferred_port is not None else self.DEFAULT_PORT

            if start_port in used_ports or not self._is_port_available(start_port):
                # Find next available port
                self.assigned_port = self._find_next_available_port(start_port, used_ports)
            else:
                self.assigned_port = start_port

            # Register this instance
            instances[self.instance_id] = {
                'hostname': self.hostname,
                'port': self.assigned_port,
                'pid': os.getpid(),
                'board_id': self.board_id,
                'timestamp': time.time()
            }

            # Save updated instances
            self._save_instances(instances)

        # Log the instances that are currently registered
        logging.info(f"Registered CortexEngine instance {self.instance_id} on port {self.assigned_port}")
        logging.info(f"Current instances: {instances}")

        return self.assigned_port

    def _unregister_instance(self) -> None:
        """Unregister this instance when shutting down."""
        if not self.assigned_port:
            return

        lock_file = self.instances_file.with_suffix('.lock')

        try:
            with portalocker.Lock(lock_file, mode='a', timeout=10) as _:
                instances = self._load_instances()

                if self.instance_id in instances:
                    del instances[self.instance_id]
                    self._save_instances(instances)
        except Exception as e:
            print(f"Warning: Could not unregister instance: {e}")

    def get_all_instances(self) -> Dict[str, Dict]:
        """Get all registered instances (after cleanup)."""
        lock_file = self.instances_file.with_suffix('.lock')

        with portalocker.Lock(lock_file, mode='a', timeout=10) as _:
            instances = self._load_instances()
            return self._cleanup_stale_instances(instances)

    # ===================== COMMAND PROCESSING =====================

    def send_command(self, action: str, params: Dict[str, Any] = None, callback: Callable = None):
        """Send a command to the StreamEngine (thread-safe)."""
        command = Command(action=action, params=params or {}, callback=callback)
        self.command_queue.put(command)

    def _process_commands(self):
        """Process pending commands from the queue."""
        while not self.command_queue.empty():
            try:
                command = self.command_queue.get_nowait()
                self._execute_command(command)
            except queue.Empty:
                break
            except Exception as e:
                logging.error(f"Error processing command: {e}")

    def _execute_command(self, command: Command):
        """Execute a single command."""
        action = command.action
        params = command.params

        try:
            if action == 'set_inference_mode':
                self._set_inference_mode(params.get('mode'))
            elif action == 'send_trigger':
                self._send_trigger(params.get('trigger', 1), params.get('timestamp', 0))
            elif action == 'train_classifier':
                self._train_classifier(params.get('data'))
            elif action == 'plot_cm':
                self._plot_cm_async()
            elif action == 'predict':
                self._predict_class()
            elif action == 'configure_filters':
                self._configure_filters(params)
            else:
                logging.warning(f"Unknown command: {action}")

            # Call callback if provided
            if command.callback:
                command.callback(True, None)

        except Exception as e:
            logging.error(f"Error executing command {action}: {e}")
            if command.callback:
                command.callback(False, str(e))

            return

    # ===================== UTILITY METHODS =====================

    def get_status(self) -> Dict[str, Any]:
        """Get current engine status."""
        return {
            'running': self.running,
            'inference_mode': self.inference_mode,
            'board_id': self.board_id,
            'sampling_rate': self.sampling_rate,
            'eeg_channels': len(self.eeg_channels)
        }

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.copy()


class HeadlessCortexEngine(CortexEngine):
    """
    Headless version that can run without any GUI.
    Perfect for server deployments, background processing, etc.
    """

    def __init__(self, board, config, window_size=1, log_file=None):
        super().__init__(board, config, window_size)
        time.sleep(window_size)
        if log_file:
            logging.basicConfig(
                filename=log_file,
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )

    def run_forever(self):
        """Run the engine indefinitely (for server mode)."""
        self.start()
        try:
            while self.running:
                time.sleep(0.1)  # Reduced sleep time for more responsive shutdown
        except KeyboardInterrupt:
            logging.info("Received interrupt signal")
        finally:
            self.stop()
