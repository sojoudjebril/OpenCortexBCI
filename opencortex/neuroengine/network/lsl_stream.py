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
# from pylsl import resolve_stream, StreamInlet, StreamOutlet
# from opencortex.processing.proc_helper import freq_bands


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
#         streams = resolve_stream('name', stream_name)
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
"""

import json
import logging
import time
import pylsl
import numpy as np
from datetime import datetime
from PyQt5.QtCore import QThread, pyqtSignal
from pylsl import resolve_stream, StreamInlet, StreamOutlet
from opencortex.processing.proc_helper import freq_bands


class LSLStreamThread(QThread):
    """Thread to read from an LSL stream and emit new sample data."""

    new_sample = pyqtSignal(object, float)  # Signal to emit new sample data
    set_train_start = pyqtSignal(object, float)
    start_train = pyqtSignal(object, float)
    start_predicting = pyqtSignal(object, float)
    stop_predicting = pyqtSignal(object, float)
    previous_ts = 0
    
    # Time synchronization parameters
    WARMUP_DURATION = 3.0  # seconds for clock sync to stabilize
    CORRECTION_UPDATE_INTERVAL = 10.0  # seconds between correction updates

    def run(self):
        """Run the LSL stream thread with proper time synchronization."""
        logging.info("Looking for LSL stream...")
        inlet = connect_lsl_marker_stream('CortexMarkers')
        
        # Warm-up period: Allow time synchronization to stabilize
        logging.info(f"Allowing {self.WARMUP_DURATION}s warm-up for time synchronization...")
        warmup_start = pylsl.local_clock()
        
        # Get initial time correction and log uncertainty
        try:
            time_correction = inlet.time_correction(timeout=5.0)
            logging.info(f"Initial time correction: {time_correction*1000:.3f} ms")
        except Exception as e:
            logging.warning(f"Could not get initial time correction: {e}")
            time_correction = 0.0
        
        # Wait for warm-up period
        while pylsl.local_clock() - warmup_start < self.WARMUP_DURATION:
            time.sleep(0.1)
        
        logging.info("Time synchronization warm-up complete. Starting data acquisition...")
        
        last_correction_update = pylsl.local_clock()

        while True:
            try:
                # Pull a new sample from the inlet (uses remote timestamp)
                marker, remote_timestamp = inlet.pull_sample()
                
                # Periodically update time correction estimate
                current_time = pylsl.local_clock()
                if current_time - last_correction_update > self.CORRECTION_UPDATE_INTERVAL:
                    try:
                        time_correction = inlet.time_correction(timeout=1.0)
                        last_correction_update = current_time
                        logging.debug(f"Updated time correction: {time_correction*1000:.3f} ms")
                    except Exception as e:
                        logging.warning(f"Failed to update time correction: {e}")
                
                # Convert remote timestamp to local time domain
                local_timestamp = remote_timestamp + time_correction
                
                # Calculate delta timestamp
                delta_ts = np.round(local_timestamp - self.previous_ts, 2) if self.previous_ts != 0 else 0
                self.previous_ts = local_timestamp

                if (marker[0] == '98'):
                    self.new_sample.emit(marker[0], local_timestamp)
                    self.set_train_start.emit(marker[0], local_timestamp)
                    date_time = datetime.fromtimestamp(local_timestamp)
                    logging.info(f"Start of training trigger {marker[0]} at LSL time {local_timestamp:.6f} ({date_time})")
                elif (marker[0] == '99'):
                    self.new_sample.emit(marker[0], local_timestamp)
                    self.start_train.emit(marker[0], local_timestamp)
                    date_time = datetime.fromtimestamp(local_timestamp)
                    logging.info(f"End of training trigger {marker[0]} at LSL time {local_timestamp:.6f} ({date_time})")
                elif (marker[0] == '100'):
                    self.new_sample.emit(marker[0], local_timestamp)
                    self.start_predicting.emit(marker[0], local_timestamp)
                    date_time = datetime.fromtimestamp(local_timestamp)
                    logging.info(f"Start inference trigger {marker[0]} at LSL time {local_timestamp:.6f} ({date_time})")
                elif (marker[0] == '101'):
                    self.new_sample.emit(marker[0], local_timestamp)
                    self.stop_predicting.emit(marker[0], local_timestamp)
                    date_time = datetime.fromtimestamp(local_timestamp)
                    logging.info(f"Stop inference trigger {marker[0]} at LSL time {local_timestamp:.6f} ({date_time})")
                else:
                    # Emit the new sample data
                    delta_ts_ms = delta_ts * 1000
                    logging.debug(f"New sample: {marker[0]} after {delta_ts_ms:.2f} ms")
                    self.new_sample.emit(marker[0], local_timestamp)
                    
            except Exception as e:
                logging.error(f"Error while reading LSL stream: {e}")
                logging.info("Looking for LSL stream...")
                inlet = connect_lsl_marker_stream('CortexMarkers')
                # Reset warm-up after reconnection
                warmup_start = pylsl.local_clock()
                while pylsl.local_clock() - warmup_start < self.WARMUP_DURATION:
                    time.sleep(0.1)
                try:
                    time_correction = inlet.time_correction(timeout=5.0)
                except:
                    time_correction = 0.0
                last_correction_update = pylsl.local_clock()


def connect_lsl_marker_stream(stream_name='CortexMarkers', type='Markers'):
    """
    Connect to an LSL stream with optimal synchronization settings
    :param stream_name: str, name of the LSL stream
    :param type: str, type of the LSL stream
    :return: StreamInlet object
    """
    try:
        streams = resolve_stream('name', stream_name)
        inlet = StreamInlet(streams[0],
                            max_buflen=360,  # Buffer up to 6 minutes
                            max_chunklen=0,  # Pull samples one at a time for markers
                            processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter | pylsl.proc_threadsafe)
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
        ch_count = len(channels) + 1
        info = pylsl.StreamInfo(name=stream_name, type=type, channel_count=ch_count,
                                nominal_srate=fs, channel_format='float32',
                                source_id=source_id)
        # Add channel names
        chs = info.desc().append_child("channels")
        for ch in channels:
            channel = chs.append_child("channel")
            channel.append_child_value("name", ch)
            channel.append_child_value("type", "EEG")
            channel.append_child_value("unit", "microvolts")
        chs.append_child("channel").append_child_value("name", "Trigger")

        eeg_outlet = StreamOutlet(info, chunk_size=0, max_buffered=360)
        logging.debug(f"LSL EEG stream started: {info.name()}")
        return eeg_outlet
    except Exception as e:
        logging.error(f"Error starting LSL stream: {e}")
        raise


def start_lsl_power_bands_stream(channels, fs, source_id, stream_name='CortexPSD', type='PSD'):
    """
    Start an LSL stream for the power bands data
    :param channels: list of str, channel names
    :param fs: int, sampling rate
    :param source_id: str, source id
    :param stream_name: str, name of the LSL stream
    :param type: str, type of the LSL stream
    :return: StreamOutlet object
    """
    try:
        ch_count = len(channels)
        info = pylsl.StreamInfo(name=stream_name, type=type, channel_count=ch_count,
                                nominal_srate=fs, channel_format='float32',
                                source_id=source_id)
        # Add channel names
        chs = info.desc().append_child("channels")
        for ch in channels:
            chs.append_child("channel").append_child_value("name", ch)
        band_powers_outlet = StreamOutlet(info, chunk_size=0, max_buffered=360)
        logging.debug(f"LSL power bands stream started: {info.name()}")
        return band_powers_outlet
    except Exception as e:
        logging.error(f"Error starting LSL stream: {e}")
        raise


def start_lsl_inference_stream(channels, fs, source_id, stream_name='CortexInference', type='Inference'):
    """
    Start an LSL stream for the prediction data
    :param channels: int, number of channels
    :param fs: int, sampling rate
    :param source_id: str, source id
    :param stream_name: str, name of the LSL stream
    :param type: str, type of the LSL stream
    :return: StreamOutlet object
    """
    try:
        info = pylsl.StreamInfo(name=stream_name, type=type, channel_count=channels,
                                nominal_srate=fs, channel_format='float32',
                                source_id=source_id)
        prediction_outlet = pylsl.StreamOutlet(info, chunk_size=0, max_buffered=360)
        logging.debug(f"LSL prediction stream started: {info.name()}")
        return prediction_outlet
    except Exception as e:
        logging.error(f"Error starting LSL stream: {e}")
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
        info = pylsl.StreamInfo(name=stream_name, type=type, channel_count=ch_count,
                                nominal_srate=fs, channel_format='float32',
                                source_id=source_id)
        quality_outlet = StreamOutlet(info, chunk_size=0, max_buffered=360)
        logging.debug(f"LSL quality stream started: {info.name()}")
        return quality_outlet
    except Exception as e:
        logging.error(f"Error starting LSL stream: {e}")
        raise


def push_lsl_raw_eeg(outlet: StreamOutlet, data, start_eeg, end_eeg, counter, ts=0, chunk=False):
    """
    Push a chunk of data to the LSL stream with proper timestamps
    :param outlet: StreamOutlet object
    :param data: numpy array of shape (n_channels, n_samples)
    :param start_eeg: int, start index of the EEG channels
    :param end_eeg: int, end index of the EEG channels
    :param counter: int, chunk counter
    :param ts: float or array, timestamp value(s) in LSL time domain
    :param chunk: bool, whether to push a chunk of data or a single sample
    """
    try:
        # Get EEG and Trigger from data and push it to LSL
        eeg = data[start_eeg:end_eeg]
        trigger = data[-1]

        # Horizontal stack EEG and Trigger
        eeg = np.concatenate((eeg, trigger.reshape(1, len(trigger))), axis=0)

        # Use pylsl.local_clock() for timestamps - DO NOT convert from time.time()
        # Timestamps should already be in LSL time domain
        if chunk:
            outlet.push_chunk(eeg.T.tolist(), ts)
            logging.debug(f"Pushed chunk {counter} to LSL stream {outlet.get_info().name()}")
        else:
            for i in range(eeg.shape[1]):
                sample = eeg[:, i]
                # If ts is scalar, use pylsl.local_clock() for each sample
                timestamp = ts[i] if isinstance(ts, (list, np.ndarray)) else pylsl.local_clock()
                outlet.push_sample(sample.tolist(), timestamp)
            logging.debug(f"Pushed {eeg.shape[1]} samples of chunk {counter} to LSL stream {outlet.get_info().name()}")
    except Exception as e:
        logging.error(f"Error pushing chunk to LSL: {e}")


def push_lsl_band_powers(outlet: StreamOutlet, band_powers, timestamp, freq_bands=freq_bands.keys()):
    """
    Push the power bands to the LSL stream
    :param outlet: StreamOutlet object
    :param band_powers: list of band power values
    :param timestamp: float or array, timestamp value(s) in LSL time domain
    """
    try:
        outlet.push_chunk(band_powers, timestamp)
        logging.debug(
            f"Pushed band powers {' '.join(list(freq_bands))} to LSL stream {outlet.get_info().name()}")
    except Exception as e:
        logging.error(f"Error pushing band powers to LSL: {e}")


def push_lsl_inference(outlet: StreamOutlet, prediction):
    """
    Push a prediction to the LSL stream with automatic timestamping
    :param outlet: StreamOutlet object
    :param prediction: dict, prediction data
    """
    try:
        # Convert the dictionary to a list
        predicted_class = prediction['class']
        # Let LSL automatically timestamp with pylsl.local_clock()
        outlet.push_sample([predicted_class])
        
        logging.debug(f"Pushed prediction {prediction} to LSL stream {outlet.get_info().name()}")
    except Exception as e:
        logging.error(f"Error pushing prediction to LSL: {e}")


def push_lsl_quality(outlet: StreamOutlet, quality):
    """
    Push a quality indicator to the LSL stream with automatic timestamping
    :param outlet: StreamOutlet object
    :param quality: list of quality indicators
    """
    try:
        # Let LSL automatically timestamp with pylsl.local_clock()
        outlet.push_sample(quality)
        logging.debug(f"Pushed quality {quality} to LSL stream {outlet.get_info().name()}")
    except Exception as e:
        logging.error(f"Error pushing quality to LSL: {e}")
