"""
CortexEngine - Main application controller and independent service
Runs as the primary loop, with optional GUI interface running on a separate thread.
Can also run headless for server deployments.

Author: Michele Romani
"""

import os
import threading
import time
import numpy as np
import logging
import queue
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor
from brainflow.board_shim import BoardShim, BoardIds

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
from opencortex.neuroengine.flux.network.stream_lsl import StreamOutLSL
from opencortex.neuroengine.network.lsl_stream import (
    start_lsl_eeg_stream, start_lsl_power_bands_stream,
    start_lsl_inference_stream, start_lsl_quality_stream,
    push_lsl_raw_eeg, push_lsl_inference, push_lsl_quality
)
from opencortex.utils.layouts import layouts


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

    def __init__(self, board, config, window_size=1):
        self.board = board
        self.config = config
        self.window_size = window_size

        self.pid = os.getpid()
        logging.info(f"Starting CortexEngine with PID {self.pid}")

        # Core properties
        self.board_id = self.board.get_board_id()
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        try:
            self.eeg_names = BoardShim.get_eeg_names(self.board_id)
        except Exception as e:
            logging.warning("Could not get EEG channels, using default 8 channels, caused by: {}".format(e))
            self.eeg_names = ["CPz", "P1", "Pz", "P2", "PO3", "POz", "PO4", "Oz"]
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.num_points = self.window_size * self.sampling_rate
        try:
            self.eeg_names = BoardShim.get_eeg_names(self.board_id)
        except Exception as e:
            logging.warning("Could not get EEG channels, using default 8 channels, caused by: {}".format(e))
            self.eeg_names = ["CPz", "P1", "Pz", "P2", "PO3", "POz", "PO4", "Oz"],

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

        lsl_pipeline = StreamOutLSL(stream_type="eeg", name="CortexEEG", channels=self.eeg_names, fs=self.sampling_rate,
                                    source_id=self.board.get_device_name(self.board_id)) \
                       >> LogNode(name="EEG") \
                       >> BandPowerExtractor(fs=self.sampling_rate, ch_names=self.eeg_names) \
                       >> StreamOutLSL(stream_type='band_powers', name='BandPowerLSL', channels=self.eeg_names,
                                       fs=self.sampling_rate, source_id=board.get_device_name(self.board_id))
        # TODO uniform IN and OUT of nodes, add error checking and/or handling
        # TODO specialize Node in RawNode and EpochNode (NumPy node?)

        signal_quality_pipeline = Sequential(
            NotchFilterNode((50, 60), name='NotchFilter'),
            BandPassFilterNode(0.1, 30.0, name='BandPassFilter'),
            StreamOutLSL(stream_type='eeg', name='FilteredEEGLSL', channels=self.eeg_names, fs=self.sampling_rate,
                         source_id=board.get_device_name(self.board_id)),
            QualityEstimator(quality_thresholds=self.quality_thresholds, name='QualityEstimator'),
            StreamOutLSL(stream_type='quality', name='QualityLSL', channels=self.eeg_names, fs=self.sampling_rate,
                         source_id=board.get_device_name(self.board_id)),
            name='SignalQualityPipeline'
        )

        configs = [
            PipelineConfig(
                pipeline=lsl_pipeline,
                name="StreamerPipeline"
            ),
            PipelineConfig(
                pipeline=signal_quality_pipeline,
                name="QualityPipeline"
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
                    self._notify_event('error', {'message': str(e)})

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

            # Process through pipeline
            _ = self.pipeline(self.filtered_eeg[0:len(self.eeg_channels)])


        except Exception as e:
            logging.error(f"Error updating data: {e}")
            raise


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
