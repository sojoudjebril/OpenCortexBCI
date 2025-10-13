import json
import logging
import time
import pylsl
import numpy as np
from datetime import datetime
from PyQt5.QtCore import QThread, pyqtSignal
from pylsl import resolve_byprop, StreamInlet, StreamOutlet
from opencortex.utils.processing.proc_helper import freq_bands


class LSLStreamThread(QThread):
    """Thread to read from an LSL stream and emit new sample data."""

    new_sample = pyqtSignal(object, float)
    set_train_start = pyqtSignal(object, float)
    start_train = pyqtSignal(object, float)
    start_predicting = pyqtSignal(object, float)
    stop_predicting = pyqtSignal(object, float)
    
    # Marker configuration mapping
    MARKER_CONFIG = {
        '98': ('set_train_start', 'Start of training'),
        '99': ('start_train', 'End of training'),
        '100': ('start_predicting', 'Start inference'),
        '101': ('stop_predicting', 'Stop inference')
    }
    
    def __init__(self):
        super().__init__()
        self.previous_ts = 0
        self.inlet = None
        self.running = True
        self._lock = None

    def run(self):
        """Run the LSL stream thread."""
        logging.info("Looking for LSL stream...")
        self.inlet = connect_lsl_marker_stream('CortexMarkers')

        while self.running:
            try:
                marker, timestamp = self.inlet.pull_sample(timeout=1.0)
                
                if marker is None:
                    continue
                
                delta_ts = np.round(timestamp - self.previous_ts, 2) if self.previous_ts != 0 else 0
                self.previous_ts = timestamp
                marker_value = marker[0]
                
                # Emit new sample for all markers
                self.new_sample.emit(marker_value, timestamp)
                
                # Handle special markers
                if marker_value in self.MARKER_CONFIG:
                    signal_name, log_msg = self.MARKER_CONFIG[marker_value]
                    getattr(self, signal_name).emit(marker_value, timestamp)
                    date_time = datetime.fromtimestamp(timestamp)
                    logging.info(f"{log_msg} trigger {marker_value} at {date_time}")
                else:
                    # Regular markers
                    delta_ts_ms = delta_ts * 1000
                    logging.debug(f"New sample: {marker_value} after {delta_ts_ms:.2f} ms")
                    
            except Exception as e:
                logging.error(f"Error while reading LSL stream: {e}")
                logging.info("Attempting to reconnect to LSL stream...")
                time.sleep(1)
                try:
                    self.inlet = connect_lsl_marker_stream('CortexMarkers')
                except Exception as reconnect_error:
                    logging.error(f"Failed to reconnect: {reconnect_error}")

    def stop(self):
        """Stop the thread gracefully."""
        self.running = False


def connect_lsl_marker_stream(stream_name='CortexMarkers', stream_type='Markers', timeout=30.0):
    """
    Connect to an LSL stream with timeout
    
    :param stream_name: str, name of the LSL stream
    :param stream_type: str, type of the LSL stream
    :param timeout: float, timeout in seconds for stream resolution
    :return: StreamInlet object
    """
    try:
        logging.info(f"Resolving LSL stream '{stream_name}'...")
        streams = resolve_byprop('name', stream_name, timeout=timeout)
        
        if not streams:
            raise RuntimeError(f"No stream found with name '{stream_name}'")
        
        inlet = StreamInlet(
            streams[0],
            max_buflen=360,
            processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter | pylsl.proc_threadsafe
        )
        
        inlet.open_stream(timeout=timeout)
        logging.info(f"LSL stream connected: {streams[0].name()}")
        return inlet
        
    except Exception as e:
        logging.error(f"Error connecting to LSL stream: {e}")
        raise


def start_lsl_stream(channels, fs, source_id, stream_name, stream_type, 
                     channel_type=None, channel_unit=None, add_trigger=False, desc=None):
    """
    Generic function to start any LSL stream
    
    :param channels: list of str or int, channel names or count
    :param fs: int, sampling rate
    :param source_id: str, source id
    :param stream_name: str, name of the LSL stream
    :param stream_type: str, type of the LSL stream
    :param channel_type: str, type for channel metadata (e.g., 'EEG', 'PSD')
    :param channel_unit: str, unit for channel metadata (e.g., 'microvolts')
    :param add_trigger: bool, whether to add a trigger channel
    :return: StreamOutlet object
    """
    try:
        # Handle channels as list of names or as count
        if isinstance(channels, int):
            ch_count = channels
            ch_names = [f"Ch{i+1}" for i in range(channels)]
        else:
            ch_count = len(channels) + (1 if add_trigger else 0)
            ch_names = channels
        
        if desc is not None:
            info = desc
        else:
            info = pylsl.StreamInfo(
                name=stream_name,
                type=stream_type,
                channel_count=ch_count,
                nominal_srate=fs,
                channel_format='float32',
                source_id=source_id
            )
        
        
        # Add channel metadata if provided
        if channel_type:
            chs = info.desc().append_child("channels")
            for ch in ch_names:
                channel = chs.append_child("channel")
                channel.append_child_value("name", ch)
                channel.append_child_value("type", channel_type)
                if channel_unit:
                    channel.append_child_value("unit", channel_unit)
            
            # Add trigger channel if requested
            if add_trigger:
                trigger_ch = chs.append_child("channel")
                trigger_ch.append_child_value("name", "Trigger")
                trigger_ch.append_child_value("type", "Trigger")
        
        outlet = StreamOutlet(info, chunk_size=32, max_buffered=360)
        logging.info(f"LSL {stream_type} stream started: {info.name()}")
        return outlet
        
    except Exception as e:
        logging.error(f"Error starting LSL {stream_type} stream: {e}")
        raise


def start_lsl_eeg_stream(channels, fs, source_id, stream_name='CortexEEG', type='EEG', desc=None, add_trigger=False):
    """Start an LSL stream for EEG data"""
    return start_lsl_stream(channels, fs, source_id, stream_name, type, 
                           channel_type='EEG', channel_unit='microvolts', add_trigger=add_trigger, desc=desc)


def start_lsl_power_bands_stream(channels, fs, source_id, stream_name='CortexPSD', type='PSD', desc=None, add_trigger=False):
    """Start an LSL stream for power bands data"""
    return start_lsl_stream(channels, fs, source_id, stream_name, type, channel_type='PSD', desc=desc)


def start_lsl_inference_stream(channels, fs, source_id, stream_name='CortexInference', type='Inference', desc=None):
    """Start an LSL stream for prediction data"""
    return start_lsl_stream(channels, fs, source_id, stream_name, type, desc=desc)


def start_lsl_quality_stream(channels, fs, source_id, stream_name='CortexQuality', type='Qualities', desc=None):
    """Start an LSL stream for quality data"""
    return start_lsl_stream(channels, fs, source_id, stream_name, type, desc=desc)


def _prepare_data_for_push(data, timestamps):
    """
    Helper to prepare data and timestamps for LSL push
    
    :param data: numpy array or list
    :param timestamps: timestamps array/list or None
    :return: tuple (data_list, timestamps_list or None)
    """
    # Convert data to numpy if needed
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    # Convert data to list
    data_list = data.tolist() if hasattr(data, 'tolist') else data
    
    # Convert timestamps if provided
    timestamps_list = None
    if timestamps is not None:
        timestamps_list = timestamps.tolist() if isinstance(timestamps, np.ndarray) else timestamps
    
    return data_list, timestamps_list


def push_lsl_raw_eeg(outlet: StreamOutlet, data):
    """
    Push a chunk of EEG data to the LSL stream

    :param outlet: StreamOutlet object
    :param data: numpy array of shape (n_channels, n_samples)
    :param start_eeg: int, start index of the EEG channels
    :param end_eeg: int, end index of the EEG channels
    :param counter: int, chunk counter
    :param timestamps: numpy array of timestamps for each sample, or None
    """
    try:
        # Transpose to (n_samples, n_channels) for LSL
        samples = data.T
        data_list, timestamps_list = _prepare_data_for_push(samples, None)

        # Push with or without timestamps
        if timestamps_list is None:
            outlet.push_chunk(data_list)
        else:
            outlet.push_chunk(data_list, timestamps_list)

        logging.debug(f"Pushed chunk ({len(data_list)} samples) to LSL stream {outlet.get_info().name()}")
    except Exception as e:
        logging.error(f"Error pushing EEG chunk to LSL: {e}")


def push_lsl_raw_eeg_old(outlet: StreamOutlet, data, start_eeg, end_eeg, counter, timestamps=None):
    """
    Push a chunk of EEG data to the LSL stream
    
    :param outlet: StreamOutlet object
    :param data: numpy array of shape (n_channels, n_samples)
    :param start_eeg: int, start index of the EEG channels
    :param end_eeg: int, end index of the EEG channels
    :param counter: int, chunk counter
    :param timestamps: numpy array of timestamps for each sample, or None
    """
    try:
        # Extract and combine EEG and Trigger channels
        eeg = data[start_eeg:end_eeg]
        trigger = data[-1:, :]
        combined_data = np.vstack((eeg, trigger))
        
        # Transpose to (n_samples, n_channels) for LSL
        samples = combined_data.T
        data_list, timestamps_list = _prepare_data_for_push(samples, timestamps)
        
        # Push with or without timestamps
        if timestamps_list is None:
            outlet.push_chunk(data_list)
        else:
            outlet.push_chunk(data_list, timestamps_list)
        
        logging.debug(f"Pushed chunk {counter} ({len(data_list)} samples) to LSL stream {outlet.get_info().name()}")
        
    except Exception as e:
        logging.error(f"Error pushing EEG chunk to LSL: {e}")


def push_lsl_band_powers(outlet: StreamOutlet, band_powers, timestamps=None):
    """
    Push the power bands to the LSL stream
    
    :param outlet: StreamOutlet object
    :param band_powers: numpy array or list of shape (n_bands, n_samples) or (n_samples, n_bands)
    :param timestamps: list or array of timestamps for each sample, or None
    """
    try:
        # Convert and reshape data
        if not isinstance(band_powers, np.ndarray):
            # Convert from dict to array if needed
            if isinstance(band_powers, dict):
                band_powers = np.array([band_powers[band] for band in freq_bands.keys()])
            else:
                band_powers = np.array(band_powers)
        
        # Ensure correct shape (n_samples, n_bands)
        if band_powers.ndim == 1:
            band_powers = band_powers.reshape(1, -1)
        elif band_powers.shape[0] > band_powers.shape[1]:
            band_powers = band_powers.T
        
        data_list, timestamps_list = _prepare_data_for_push(band_powers, timestamps)
        
        if timestamps_list is None:
            outlet.push_chunk(data_list)
        else:
            outlet.push_chunk(data_list, timestamps_list)
        
        logging.debug(f"Pushed {len(data_list)} band power samples to LSL stream {outlet.get_info().name()}")

    except Exception as e:
        logging.error(f"Error pushing band powers to LSL: {e}")


def push_lsl_sample(outlet: StreamOutlet, sample_data, timestamp=None):
    """
    Generic function to push a single sample to LSL stream
    
    :param outlet: StreamOutlet object
    :param sample_data: numeric value, dict with 'class' key, or list/array
    :param timestamp: float, timestamp value (None for automatic)
    """
    try:
        # Handle different input types
        if isinstance(sample_data, dict):
            sample = [float(sample_data.get('class', 0))]
        elif isinstance(sample_data, (list, tuple)):
            sample = [float(x) for x in sample_data]
        elif isinstance(sample_data, np.ndarray):
            sample = sample_data.tolist()
        else:
            sample = [float(sample_data)]
        
        # Push with or without timestamp
        if timestamp is None:
            outlet.push_sample(sample)
        else:
            outlet.push_sample(sample, timestamp)
        
        logging.debug(f"Pushed sample {sample} to LSL stream {outlet.get_info().name()}")
        
    except Exception as e:
        logging.error(f"Error pushing sample to LSL: {e}")


def push_lsl_inference(outlet: StreamOutlet, prediction, timestamp=None):
    """Push a prediction to the LSL stream"""
    push_lsl_sample(outlet, prediction, timestamp)


def push_lsl_quality(outlet: StreamOutlet, quality, timestamp=None):
    """Push quality indicators to the LSL stream"""
    push_lsl_sample(outlet, quality, timestamp)
