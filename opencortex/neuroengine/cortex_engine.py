"""
CortexEngine - Main application controller and independent service

Authors: Michele Romani, Gregorio Andrea Giudici
"""

from asyncore import dispatcher
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
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor
from brainflow.board_shim import BoardShim, BoardIds
import portalocker  # For safe file locking

from pythonosc import osc_server
from pythonosc.dispatcher import Dispatcher

import opencortex.neuroengine.flux.base.operators  # Needed to enable >> and + operators
from opencortex.neuroengine.flux.base.parallel import Parallel
from opencortex.neuroengine.flux.base.sequential import Sequential
from opencortex.neuroengine.flux.features.band_power import BandPowerExtractor
from opencortex.neuroengine.flux.features.quality_estimator import QualityEstimator
from opencortex.neuroengine.flux.pipeline_config import PipelineConfig
from opencortex.neuroengine.flux.pipeline_group import PipelineGroup
from opencortex.neuroengine.flux.preprocessing.bandpass import BandPassFilterNode
from opencortex.neuroengine.flux.preprocessing.notch import NotchFilterNode
from opencortex.neuroengine.flux.base.simple_nodes import LogNode
from opencortex.neuroengine.flux.network.lsl import StreamOutLSL
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

    def __init__(self, board, config, window_size=1):
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

        self.instances_file = Path(__file__).parent / self.INSTANCE_REGISTRY_FILE

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


        # Core properties
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        try:
            self.eeg_names = BoardShim.get_eeg_names(self.board_id)
        except Exception as e:
            log.warning("Could not get EEG channels, using default 8 channels, caused by: {}".format(e))
            self.eeg_names = ["CPz", "P1", "Pz", "P2", "PO3", "POz", "PO4", "Oz"]
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.num_points = self.window_size * self.sampling_rate

        # Engine state
        self.running = False
        self.inference_mode = False
        self.first_prediction = True

        # Configuration
        self.model = config.get('model', 'LDA')
        self.proba = config.get('proba', False)
        self.group_predictions = config.get('group_predictions', False)
        self.nclasses = config.get('nclasses', 3)
        self.flash_time = config.get('flash_time', 250)
        self.epoch_length_ms = config.get('epoch_length_ms', 1000)
        self.baseline_ms = config.get('baseline_ms', 100)
        self.quality_thresholds = config.get('quality_thresholds',
                                             [(-100, -50, 'yellow', 0.5), (-50, 50, 'green', 1.0),
                                              (50, 100, 'yellow', 0.5)])
        self.over_sample = config.get('oversample', True)
        self.update_interval_ms = config.get('update_buffer_speed_ms', 50)

        lsl_pipeline = BandPowerExtractor(fs=self.sampling_rate, ch_names=self.eeg_names) \
                       >> StreamOutLSL(stream_type='band_powers', name='BandPowerLSL', logger=log, channels=self.eeg_names,
                                       fs=self.sampling_rate, source_id=board.get_device_name(self.board_id))
        # TODO uniform IN and OUT of nodes, add error checking and/or handling
        # TODO specialize Node in RawNode and EpochNode (NumPy node?)

        signal_quality_pipeline = Sequential(
            StreamOutLSL(stream_type="eeg", name="CortexEEG", logger=log, channels=self.eeg_names + ["Trigger"], fs=self.sampling_rate,
                         source_id=self.board.get_device_name(self.board_id)),
            NotchFilterNode((50, 60), name='NotchFilter'),
            BandPassFilterNode(0.1, 30.0, name='BandPassFilter'),
            QualityEstimator(quality_thresholds=self.quality_thresholds, name='QualityEstimator'),
            StreamOutLSL(stream_type='quality', name='QualityLSL', logger=log, channels=self.eeg_names, fs=self.sampling_rate,
                         source_id=board.get_device_name(self.board_id)),
            name='SignalQualityPipeline'
        )

        configs = [
            PipelineConfig(
                pipeline=lsl_pipeline,
                name="LSLStream"
            ),
            PipelineConfig(
                pipeline=signal_quality_pipeline,
                name="SignalQuality"
            )
        ]

        self.pipeline = PipelineGroup(
            pipelines=configs,
            name="CortexEnginePipeline",
            max_workers=2,
            wait_for_all=False
        )


        # Data buffers
        self.filtered_eeg = np.zeros((len(self.eeg_channels) + 1, self.num_points))
        self.raw_data = None

        # Threading and communication
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.command_queue = queue.Queue()
        self.main_thread = None
        self.classifier = None

        # Interface callbacks (GUI, API, etc.)
        self.data_callbacks = []
        self.event_callbacks = []

        # Timing calculations
        # self._calculate_timing_parameters()

        logging.info("StreamEngine initialized")

    # def _calculate_timing_parameters(self):
    #     """Calculate timing parameters for predictions and epochs."""
    #     self.off_time = (self.flash_time * (self.nclasses - 1))
    #     self.prediction_interval = int(2 * self.flash_time + self.off_time)
    #     self.epoch_data_points = int(self.epoch_length_ms * self.sampling_rate / 1000)
    #     self.inference_ms = self.baseline_ms + (self.flash_time * self.nclasses) + self.epoch_length_ms
    #     self.prediction_datapoints = int(self.inference_ms * self.sampling_rate / 1000)

    #     self.slicing_trigger = (self.epoch_length_ms + self.baseline_ms) // self.flash_time
    #     if self.slicing_trigger > self.nclasses:
    #         self.slicing_trigger = self.nclasses

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

        self.executor.shutdown(wait=True)
        logging.info("StreamEngine stopped")

    def _main_loop(self):
        """Main processing loop - runs independently."""
        last_update = time.time()

        while self.running:
            current_time = time.time()

            # Process commands from interfaces
            self._process_commands()

            # Update data at configured interval
            if (current_time - last_update) * 1000 >= self.update_interval_ms:
                try:
                    self._update_data()
                    last_update = current_time
                except Exception as e:
                    logging.error(f"Error in main loop: {e}")

            # Small sleep to prevent 100% CPU usage
            time.sleep(0.001)  # TODO: check if needed (NOTE: on Greg's PC this reduces CPU load from ~20% to ~8%)

    def _update_data(self):
        """Core data processing - heart of the engine."""
        try:
            # Get raw data from board
            data = self.board.get_current_board_data(num_samples=self.num_points)
            self.raw_data = data

            # Extract and filter EEG
            start_eeg = layouts[self.board_id]["eeg_start"]
            end_eeg = layouts[self.board_id]["eeg_end"]
            eeg = data[start_eeg:end_eeg]

            # Update filtered buffer
            for count, channel in enumerate(self.eeg_channels):
                self.filtered_eeg[count] = eeg[count]

            # Extract trigger and timestamp
            trigger = data[-1]
            ts_channel = self.board.get_timestamp_channel(self.board_id)
            ts = data[ts_channel]
            self.filtered_eeg[-1] = trigger

            raw = self.filtered_eeg[0:len(self.eeg_channels)].T
            # Process through pipeline
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
            'classifier_ready': self.classifier is not None,
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
