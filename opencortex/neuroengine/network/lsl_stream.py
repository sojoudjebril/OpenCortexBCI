# """
# This class handles the LSL stream thread to read from an LSL stream and emit new sample data.
# The LSL stream is used to send the EEG data, the power bands, the prediction, and the quality indicators.

# Author: Michele Romani
# Email: michele.romani.zaltieri@gmail.com
# Copyright 2024 Michele Romani
# """

# import json
# import logging
# import time
# import pylsl
# import numpy as np
# from datetime import datetime
# from PyQt5.QtCore import QThread, pyqtSignal
# from pylsl import resolve_byprop, StreamInlet, StreamOutlet
# from opencortex.utils.processing.proc_helper import freq_bands


# class LSLStreamThread(QThread):
#     """Thread to read from an LSL stream and emit new sample data."""

#     new_sample = pyqtSignal(object, float)  # Signal to emit new sample data
#     set_train_start = pyqtSignal(object, float)
#     start_train = pyqtSignal(object, float)
#     start_predicting = pyqtSignal(object, float)
#     stop_predicting = pyqtSignal(object, float)
#     previous_ts = 0

#     def run(self):
#         """ Run the LSL stream thread."""
#         logging.info("Looking for LSL stream...")
#         inlet = connect_lsl_marker_stream('CortexMarkers')

#         while True:
#             try:
#                 # Pull a new sample from the inlet
#                 marker, timestamp = inlet.pull_sample()
#                 timestamp = pylsl.local_clock()
#                 delta_ts = np.round(timestamp - self.previous_ts, 2) if self.previous_ts != 0 else 0
#                 self.previous_ts = timestamp

#                 if (marker[0] == '98'):
#                     self.new_sample.emit(marker[0], timestamp)
#                     self.set_train_start.emit(marker[0], timestamp)
#                     date_time = datetime.fromtimestamp(time.time())
#                     logging.info(f"Start of training trigger {marker[0]} written at {date_time}")
#                 if (marker[0] == '99'):
#                     self.new_sample.emit(marker[0], timestamp)
#                     self.start_train.emit(marker[0], timestamp)
#                     date_time = datetime.fromtimestamp(time.time())
#                     logging.info(f"End of training trigger {marker[0]} written at {date_time}")
#                 elif (marker[0] == '100'):
#                     self.new_sample.emit(marker[0], timestamp)
#                     self.start_predicting.emit(marker[0], timestamp)
#                     date_time = datetime.fromtimestamp(time.time())
#                     logging.info(f"Start inference trigger {marker[0]} written at {date_time}")
#                 elif (marker[0] == '101'):
#                     self.new_sample.emit(marker[0], timestamp)
#                     self.stop_predicting.emit(marker[0], timestamp)
#                     date_time = datetime.fromtimestamp(time.time())
#                     logging.info(f"Stop inference trigger {marker[0]} written at {date_time}")
#                 else:
#                     # Emit the new sample data
#                     delta_ts_ms = delta_ts * 1000
#                     logging.debug(f"New sample: {marker[0]} after {delta_ts_ms} ms")
#                     self.new_sample.emit(marker[0], timestamp)
#             except Exception as e:
#                 logging.error(f"Error while reading LSL stream: {e}")
#                 logging.info("Looking for LSL stream...")
#                 inlet = connect_lsl_marker_stream('CortexMarkers')



# def connect_lsl_marker_stream(stream_name='CortexMarkers', type='Markers'):
#     """
#     Connect to an LSL stream
#     :param source_id: str, source id
#     :param stream_name: str, name of the LSL stream
#     :param type: str, type of the LSL stream
#     :return: StreamInlet object
#     """
#     try:
#         streams = resolve_byprop('name', stream_name)
#         inlet = StreamInlet(streams[0],
#                             processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter | pylsl.proc_threadsafe)
#         logging.info(f"LSL stream connected {streams[0].name()}")
#         return inlet
#     except Exception as e:
#         logging.error(f"Error connecting to LSL stream: {e}")


# def start_lsl_eeg_stream(channels, fs, source_id, stream_name='CortexEEG', type='EEG'):
#     """
#     Start an LSL stream for the EEG data

#     :param channels: list of str, channel names
#     :param fs: int, sampling rate
#     :param source_id: str, source id
#     :param stream_name: str, name of the LSL stream
#     :param type: str, type of the LSL stream
#     :return: StreamOutlet object
#     """
#     try:
#         ch_count = len(channels) + 1
#         info = pylsl.StreamInfo(name=stream_name, type=type, channel_count=ch_count,
#                                 nominal_srate=fs, channel_format='float32',
#                                 source_id=source_id)
#         # Add channel names
#         chs = info.desc().append_child("channels")
#         for ch in channels:
#             channel = chs.append_child("channel")
#             channel.append_child_value("name", ch)
#             channel.append_child_value("type", "EEG")
#             channel.append_child_value("unit", "microvolts")
#         chs.append_child("channel").append_child_value("name", "Trigger")

#         eeg_outlet = StreamOutlet(info)
#         logging.debug(f"LSL EEG stream started {info.name()}")
#         return eeg_outlet
#     except Exception as e:
#         logging.error(f"Error starting LSL stream: {e}")


# def start_lsl_power_bands_stream(channels, fs, source_id, stream_name='CortexPSD', type='PSD'):
#     """
#     Start an LSL stream for the power bands data
#     :param channels: list of str, channel names
#     :param fs: int, sampling rate
#     :param source_id: str, source id
#     :param stream_name: str, name of the LSL stream
#     :param type: str, type of the LSL stream
#     :return: StreamOutlet object
#     """
#     try:
#         ch_count = len(channels)
#         info = pylsl.StreamInfo(name=stream_name, type=type, channel_count=ch_count,
#                                 nominal_srate=fs, channel_format='float32',
#                                 source_id=source_id)
#         # Add channel names
#         chs = info.desc().append_child("channels")
#         for ch in channels:
#             chs.append_child("channel").append_child_value("name", ch)
#         band_powers_outlet = StreamOutlet(info)
#         logging.debug(f"LSL power bands stream started {info.name()}")
#         return band_powers_outlet
#     except Exception as e:
#         logging.error(f"Error starting LSL stream: {e}")


# def start_lsl_inference_stream(channels, fs, source_id, stream_name='CortexInference', type='Inference'):
#     """
#     Start an LSL stream for the prediction data
#     :param fs: int, sampling rate
#     :param source_id: str, source id
#     :param stream_name: str, name of the LSL stream
#     :param type: str, type of the LSL stream
#     :return: StreamOutlet object
#     """
#     try:
#         info = pylsl.StreamInfo(name=stream_name, type=type, channel_count=channels,
#                                 nominal_srate=fs, channel_format='float32',
#                                 source_id=source_id)
#         prediction_outlet = pylsl.StreamOutlet(info)
#         logging.debug(f"LSL prediction stream started {info.name()}")
#         return prediction_outlet
#     except Exception as e:
#         logging.error(f"Error starting LSL stream: {e}")


# def start_lsl_quality_stream(channels, fs, source_id, stream_name='CortexQuality', type='Qualities'):
#     """ Start an LSL stream for the quality dat
#     :param channels: list of str, channel names
#     :param fs: int, sampling rate
#     :param source_id: str, source id
#     :param stream_name: str, name of the LSL stream
#     :param type: str, type of the LSL stream
#     :return StreamOutlet object
#     """
#     try:
#         ch_count = len(channels)
#         info = pylsl.StreamInfo(name=stream_name, type=type, channel_count=ch_count,
#                                 nominal_srate=fs, channel_format='float32',
#                                 source_id=source_id)
#         quality_outlet = StreamOutlet(info)
#         logging.debug(f"LSL quality stream started {info.name()}")
#         return quality_outlet
#     except Exception as e:
#         logging.error(f"Error starting LSL stream: {e}")


# def push_lsl_raw_eeg(outlet: StreamOutlet, data, start_eeg, end_eeg, counter, ts=0, chunk=False):
#     """
#     Push a chunk of data to the LSL stream
#     :param outlet: StreamOutlet object
#     :param data: numpy array of shape (n_channels, n_samples)
#     :param start_eeg: int, start index of the EEG channels
#     :param end_eeg: int, end index of the EEG channels
#     :param counter: int, chunk counter
#     :param ts: float, timestamp value
#     :param chunk: bool, whether to push a chunk of data or a single sample
#     """
#     try:
#         # Get EEG and Trigger from data and push it to LSL
#         eeg = data[start_eeg:end_eeg]
#         trigger = data[-1]

#         # Horizontal stack EEG and Trigger
#         eeg = np.concatenate((eeg, trigger.reshape(1, len(trigger))), axis=0)

#         ts_to_lsl_offset = time.time() - pylsl.local_clock()
#         # Get only the seconds part of the timestamp
#         ts = ts - ts_to_lsl_offset
#         if chunk:
#             outlet.push_chunk(eeg.T.tolist(), ts)
#             logging.debug(f"Pushed chunk {counter} to LSL stream {outlet.get_info().name()}")
#         else:
#             for i in range(eeg.shape[1]):
#                 sample = eeg[:, i]
#                 outlet.push_sample(sample.tolist(), ts[i])
#             logging.debug(f"Pushed {eeg.shape[1]} samples  of chunk {counter} to LSL stream {outlet.get_info().name()}")
#     except Exception as e:
#         logging.error(f"Error pushing chunk to LSL: {e}")


# def push_lsl_band_powers(outlet: StreamOutlet, band_powers, timestamp, freq_bands=freq_bands.keys()):
#     """
#     Push the power bands to the LSL stream
#     :param outlet: StreamOutlet object
#     :param band_powers: list of band power values
#     :param timestamp: float, timestamp value
#     """
#     try:
#         outlet.push_chunk(band_powers, timestamp)
#         logging.debug(
#             f"Pushed band powers {' '.join(list(freq_bands))} to LSL stream {outlet.get_info().name()}")
#     except Exception as e:
#         logging.error(f"Error pushing band powers to LSL: {e}")


# def push_lsl_inference(outlet: StreamOutlet, prediction):
#     """
#     Push a prediction to the LSL stream
#     :param outlet: StreamOutlet object
#     :param prediction: dict, prediction data
#     """
#     try:
#         # Serialize the dictionary to a JSON string
#         #prediction_json = json.dumps(prediction, default=convert_to_serializable)
#         #outlet.push_sample([prediction_json])

#         # convert the dictionary to a list
#         predicted_class = prediction['class']
#         outlet.push_sample([predicted_class])

#         logging.debug(f"Pushed prediction {prediction} to LSL stream {outlet.get_info().name()} ")
#     except Exception as e:
#         logging.error(f"Error pushing prediction to LSL: {e}")


# def push_lsl_quality(outlet: StreamOutlet, quality):
#     """
#     Push a quality indicator to the LSL stream
#     :param outlet: StreamOutlet object
#     :param quality: list of quality indicators
#     """
#     try:
#         outlet.push_sample(quality)
#         logging.debug(f"Pushed quality {quality} to LSL stream {outlet.get_info().name()}")
#     except Exception as e:
#         logging.error(f"Error pushing quality to LSL: {e}")


"""
This class handles the LSL stream thread to read from an LSL stream and emit new sample data.
The LSL stream is used to send the EEG data, the power bands, the prediction, and the quality indicators.

Author: Michele Romani
Email: michele.romani.zaltieri@gmail.com
Copyright 2024 Michele Romani

Optimized version with improved time synchronization and performance
"""

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

    new_sample = pyqtSignal(object, float)  # Signal to emit new sample data
    set_train_start = pyqtSignal(object, float)
    start_train = pyqtSignal(object, float)
    start_predicting = pyqtSignal(object, float)
    stop_predicting = pyqtSignal(object, float)
    
    def __init__(self):
        super().__init__()
        self.previous_ts = 0
        self.inlet = None
        self.running = True
        self._lock = None  # Thread lock for LSL calls (Linux compatibility)

    def run(self):
        """Run the LSL stream thread."""
        logging.info("Looking for LSL stream...")
        self.inlet = connect_lsl_marker_stream('CortexMarkers')

        while self.running:
            try:
                # Pull a new sample from the inlet
                # The timestamp returned is already synchronized across the network via LSL
                marker, timestamp = self.inlet.pull_sample(timeout=1.0)
                
                if marker is None:
                    continue
                
                # Calculate delta using LSL synchronized timestamps
                delta_ts = np.round(timestamp - self.previous_ts, 2) if self.previous_ts != 0 else 0
                self.previous_ts = timestamp

                marker_value = marker[0]
                
                # Handle training start trigger
                if marker_value == '98':
                    self.new_sample.emit(marker_value, timestamp)
                    self.set_train_start.emit(marker_value, timestamp)
                    date_time = datetime.fromtimestamp(timestamp)
                    logging.info(f"Start of training trigger {marker_value} at {date_time}")
                
                # Handle training end trigger
                elif marker_value == '99':
                    self.new_sample.emit(marker_value, timestamp)
                    self.start_train.emit(marker_value, timestamp)
                    date_time = datetime.fromtimestamp(timestamp)
                    logging.info(f"End of training trigger {marker_value} at {date_time}")
                
                # Handle inference start trigger
                elif marker_value == '100':
                    self.new_sample.emit(marker_value, timestamp)
                    self.start_predicting.emit(marker_value, timestamp)
                    date_time = datetime.fromtimestamp(timestamp)
                    logging.info(f"Start inference trigger {marker_value} at {date_time}")
                
                # Handle inference stop trigger
                elif marker_value == '101':
                    self.new_sample.emit(marker_value, timestamp)
                    self.stop_predicting.emit(marker_value, timestamp)
                    date_time = datetime.fromtimestamp(timestamp)
                    logging.info(f"Stop inference trigger {marker_value} at {date_time}")
                
                # Handle regular markers
                else:
                    delta_ts_ms = delta_ts * 1000
                    logging.debug(f"New sample: {marker_value} after {delta_ts_ms:.2f} ms")
                    self.new_sample.emit(marker_value, timestamp)
                    
            except Exception as e:
                logging.error(f"Error while reading LSL stream: {e}")
                logging.info("Attempting to reconnect to LSL stream...")
                time.sleep(1)  # Wait before reconnecting
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
            max_buflen=360,  # Buffer up to 360 seconds
            processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter | pylsl.proc_threadsafe
        )
        
        # Open the stream and wait for first sample to ensure connection
        inlet.open_stream(timeout=timeout)
        
        logging.info(f"LSL stream connected: {streams[0].name()}")
        return inlet
        
    except Exception as e:
        logging.error(f"Error connecting to LSL stream: {e}")
        raise


def start_lsl_eeg_stream(channels, fs, source_id, stream_name='CortexEEG', type='EEG'):
    """
    Start an LSL stream for the EEG data

    :param channels: list of str, channel names
    :param fs: int, sampling rate
    :param source_id: str, source id
    :param stream_name: str, name of the LSL stream
    :param type: str, type of the LSL stream
    :return: StreamOutlet object
    """
    try:
        ch_count = len(channels) + 1  # +1 for trigger channel
        info = pylsl.StreamInfo(
            name=stream_name,
            type=type,
            channel_count=ch_count,
            nominal_srate=fs,
            channel_format='float32',
            source_id=source_id
        )
        
        # Add channel metadata
        chs = info.desc().append_child("channels")
        for ch in channels:
            channel = chs.append_child("channel")
            channel.append_child_value("name", ch)
            channel.append_child_value("type", "EEG")
            channel.append_child_value("unit", "microvolts")
        
        # Add trigger channel
        trigger_ch = chs.append_child("channel")
        trigger_ch.append_child_value("name", "Trigger")
        trigger_ch.append_child_value("type", "Trigger")

        eeg_outlet = StreamOutlet(info, chunk_size=32, max_buffered=360)
        logging.info(f"LSL EEG stream started: {info.name()}")
        return eeg_outlet
        
    except Exception as e:
        logging.error(f"Error starting LSL EEG stream: {e}")
        raise


def start_lsl_power_bands_stream(channels, fs, source_id, stream_name='CortexPSD', type='PSD'):
    """
    Start an LSL stream for the power bands data
    
    :param channels: list of str, channel names (e.g., ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'])
    :param fs: int, sampling rate
    :param source_id: str, source id
    :param stream_name: str, name of the LSL stream
    :param type: str, type of the LSL stream
    :return: StreamOutlet object
    """
    try:
        ch_count = len(channels)
        info = pylsl.StreamInfo(
            name=stream_name,
            type=type,
            channel_count=ch_count,
            nominal_srate=fs,
            channel_format='float32',
            source_id=source_id
        )
        
        # Add channel names
        chs = info.desc().append_child("channels")
        for ch in channels:
            channel = chs.append_child("channel")
            channel.append_child_value("name", ch)
            channel.append_child_value("type", "PSD")
        
        band_powers_outlet = StreamOutlet(info, chunk_size=32, max_buffered=360)
        logging.info(f"LSL power bands stream started: {info.name()}")
        return band_powers_outlet
        
    except Exception as e:
        logging.error(f"Error starting LSL power bands stream: {e}")
        raise


def start_lsl_inference_stream(channels, fs, source_id, stream_name='CortexInference', type='Inference'):
    """
    Start an LSL stream for the prediction data
    
    :param channels: int, number of channels (e.g., 1 for single class prediction)
    :param fs: int, sampling rate
    :param source_id: str, source id
    :param stream_name: str, name of the LSL stream
    :param type: str, type of the LSL stream
    :return: StreamOutlet object
    """
    try:
        info = pylsl.StreamInfo(
            name=stream_name,
            type=type,
            channel_count=channels,
            nominal_srate=fs,
            channel_format='float32',
            source_id=source_id
        )
        
        prediction_outlet = StreamOutlet(info, chunk_size=32, max_buffered=360)
        logging.info(f"LSL inference stream started: {info.name()}")
        return prediction_outlet
        
    except Exception as e:
        logging.error(f"Error starting LSL inference stream: {e}")
        raise


def start_lsl_quality_stream(channels, fs, source_id, stream_name='CortexQuality', type='Qualities'):
    """
    Start an LSL stream for the quality data
    
    :param channels: list of str, channel names
    :param fs: int, sampling rate
    :param source_id: str, source id
    :param stream_name: str, name of the LSL stream
    :param type: str, type of the LSL stream
    :return: StreamOutlet object
    """
    try:
        ch_count = len(channels)
        info = pylsl.StreamInfo(
            name=stream_name,
            type=type,
            channel_count=ch_count,
            nominal_srate=fs,
            channel_format='float32',
            source_id=source_id
        )
        
        quality_outlet = StreamOutlet(info, chunk_size=32, max_buffered=360)
        logging.info(f"LSL quality stream started: {info.name()}")
        return quality_outlet
        
    except Exception as e:
        logging.error(f"Error starting LSL quality stream: {e}")
        raise


def push_lsl_raw_eeg(outlet: StreamOutlet, data, start_eeg, end_eeg, counter, timestamps=None):
    """
    Push a chunk of EEG data to the LSL stream
    
    :param outlet: StreamOutlet object
    :param data: numpy array of shape (n_channels, n_samples)
    :param start_eeg: int, start index of the EEG channels
    :param end_eeg: int, end index of the EEG channels
    :param counter: int, chunk counter
    :param timestamps: numpy array of timestamps for each sample, or None to use current time
    """
    try:
        # Extract EEG and Trigger channels
        eeg = data[start_eeg:end_eeg]
        trigger = data[-1:, :]  # Keep as 2D array
        
        # Vertically stack EEG and Trigger (both are 2D)
        combined_data = np.vstack((eeg, trigger))
        
        # Transpose to get (n_samples, n_channels) for LSL
        samples = combined_data.T
        
        # Use provided timestamps or generate them
        if timestamps is None:
            # Push with automatic timestamping
            outlet.push_chunk(samples.tolist())
        else:
            # Ensure timestamps is a list
            if isinstance(timestamps, np.ndarray):
                timestamps = timestamps.tolist()
            outlet.push_chunk(samples.tolist(), timestamps)
        
        logging.debug(f"Pushed chunk {counter} ({samples.shape[0]} samples) to LSL stream {outlet.get_info().name()}")
        
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
        # Convert to numpy array if needed
        if not isinstance(band_powers, np.ndarray):
            band_powers = np.array(band_powers)
        
        # Ensure correct shape (n_samples, n_bands)
        if band_powers.ndim == 1:
            band_powers = band_powers.reshape(1, -1)
        elif band_powers.shape[0] > band_powers.shape[1]:
            # Likely in (n_bands, n_samples) format, transpose
            band_powers = band_powers.T
        
        # Convert to list for LSL
        samples = band_powers.tolist()
        
        if timestamps is None:
            outlet.push_chunk(samples)
        else:
            if isinstance(timestamps, np.ndarray):
                timestamps = timestamps.tolist()
            outlet.push_chunk(samples, timestamps)
        
        logging.debug(f"Pushed {len(samples)} band power samples to LSL stream {outlet.get_info().name()}")
        
    except Exception as e:
        logging.error(f"Error pushing band powers to LSL: {e}")


def push_lsl_inference(outlet: StreamOutlet, prediction, timestamp=None):
    """
    Push a prediction to the LSL stream
    
    :param outlet: StreamOutlet object
    :param prediction: dict with 'class' key, or numeric value
    :param timestamp: float, timestamp value (None for automatic)
    """
    try:
        # Extract predicted class
        if isinstance(prediction, dict):
            predicted_class = prediction.get('class', 0)
        else:
            predicted_class = prediction
        
        # Push sample
        if timestamp is None:
            outlet.push_sample([float(predicted_class)])
        else:
            outlet.push_sample([float(predicted_class)], timestamp)
        
        logging.debug(f"Pushed prediction {predicted_class} to LSL stream {outlet.get_info().name()}")
        
    except Exception as e:
        logging.error(f"Error pushing prediction to LSL: {e}")


def push_lsl_quality(outlet: StreamOutlet, quality, timestamp=None):
    """
    Push quality indicators to the LSL stream
    
    :param outlet: StreamOutlet object
    :param quality: list or array of quality indicators
    :param timestamp: float, timestamp value (None for automatic)
    """
    try:
        # Ensure quality is a list
        if isinstance(quality, np.ndarray):
            quality = quality.tolist()
        
        if timestamp is None:
            outlet.push_sample(quality)
        else:
            outlet.push_sample(quality, timestamp)
        
        logging.debug(f"Pushed quality indicators to LSL stream {outlet.get_info().name()}")
        
    except Exception as e:
        logging.error(f"Error pushing quality to LSL: {e}")
