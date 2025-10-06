"""
This class creates a GUI to plot and handle LSL events.
Filters can be applied to the data.
"""
import sys
import threading
import time
import numpy as np
import logging
import importlib.resources as pkg_resources
import pyqtgraph as pg
import os
import yaml
from PyQt5 import QtWidgets, QtCore
from opencortex.neuroengine.models.classifier import Classifier
from opencortex.neuroengine.core.cortex_engine import CortexEngine
from opencortex.neuroengine.flux.base.parallel import Parallel
from opencortex.neuroengine.flux.features.band_power import BandPowerExtractor
from opencortex.neuroengine.flux.features.quality_estimator import QualityEstimator
from opencortex.gui.widgets.frequency_band_widget import FrequencyBandPanel
from opencortex.gui.widgets.stream_select_widget import LSLSelector
from opencortex.gui.gui_adapter import GUIAdapter
from opencortex.neuroengine.network.lsl_stream import LSLStreamThread, start_lsl_eeg_stream, start_lsl_power_bands_stream, \
    start_lsl_inference_stream, start_lsl_quality_stream, push_lsl_raw_eeg, push_lsl_inference, \
    push_lsl_quality
from pyqtgraph import ScatterPlotItem, mkBrush
from brainflow.board_shim import BoardShim
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from concurrent.futures import ThreadPoolExecutor
from opencortex.neuroengine.network.osc_stream import OscStreamThread
from opencortex.neuroengine.network.lsl_stream import LSLStreamThread
from opencortex.utils.layouts import layouts

colors = ["blue", "green", "yellow", "purple", "orange", "pink", "brown", "gray",
          "cyan", "magenta", "lime", "teal", "lavender", "turquoise", "maroon", "olive",
          "blue", "green", "yellow", "purple", "orange", "pink", "brown", "gray",
          "cyan", "magenta", "lime", "teal", "lavender", "turquoise", "maroon", "olive"]


def write_header(file, board_id):
    for column in layouts[board_id]["header"]:
        file.write(str(column) + '\t')
    file.write('\n')


class StreamerGUI:

    def __init__(self, config_file='default_config.yaml'):
        # Load configuration from file
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        # TO BE REMOVED
        self.windows_size = 1
        self.sampling_rate = 250
        self.update_plot_speed_ms = 100
        self.update_data_buffer_speed_ms = 100 # int(1000 / (self.sampling_rate / 4))
        self.num_points = int(self.windows_size * self.sampling_rate)
        self.eeg_channels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.eeg_names = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8', 'F5', 'F7', 'F3', 'F1', 'F2', 'F4', 'F6', 'F8']
        
        self.is_streaming = False
        self.plot = True
        self.lsl_state = True
        self.osc_state = False
        self.osc_thread = None
        self.initial_ts = time.time()
        
        # LSL stream variables
        self.lsl_inlet = None
        self.lsl_stream_info = None
        self.lsl_channel_count = 0
        
        logging.info("Initializing GUI...")

        self.app = QtWidgets.QApplication([])

        self.win = pg.GraphicsLayoutWidget(title='OpenCortex Streamer', size=(1920, 1080))
        self.win.setWindowTitle('OpenCortex Streamer')
        self.win.setWindowIcon(QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.SP_ComputerIcon))
        self.win.show()
        
        # Create LSL selector widget
        self.lsl_streamer = LSLSelector()
        self.lsl_streamer.setStyleSheet("background-color: #2b2b2b; color: white;")
        # self.lsl_streamer.setStyleSheet("""
        #             QLabel {
        #                 color: #4CAF50;
        #                 font-size: 12px;
        #                 font-weight: bold;
        #                 padding: 5px;
        #                 background-color: #2a2a2a;
        #                 border-radius: 4px;
        #             }
        # }""")
        
        # Connect the LSL selector to handle stream selection
        self.lsl_streamer.confirm_button.clicked.connect(self.on_lsl_stream_selected)
        
        parameter_panel = self.create_parameters_panel()
        
        plot = self.init_plot()

        self.freq_band_panel = FrequencyBandPanel()
        
        side_panel_widget = QtWidgets.QWidget()
        side_panel_layout = QtWidgets.QVBoxLayout()
        side_panel_layout.addWidget(self.lsl_streamer)
        side_panel_layout.addWidget(parameter_panel)
        side_panel_widget.setLayout(side_panel_layout)

        # Create a layout for the main window
        self.main_layout = QtWidgets.QGridLayout()
        self.main_layout.addWidget(plot, 0, 0)
        self.main_layout.addWidget(side_panel_widget, 0, 1, alignment=QtCore.Qt.AlignCenter)

        # Set the main layout for the window
        self.win.setLayout(self.main_layout)


        self.app.exec_()

    def on_lsl_stream_selected(self):
        """Called when user confirms LSL stream selection"""
        if self.lsl_streamer.inlet:
            self.lsl_inlet = self.lsl_streamer.inlet
            self.lsl_stream_info = self.lsl_inlet.info()
            self.lsl_channel_count = self.lsl_stream_info.channel_count()
            self.sampling_rate = self.lsl_stream_info.nominal_srate()
            
            # Update timing parameters based on actual sampling rate
            self.num_points = int(self.windows_size * self.sampling_rate)
            
            # Get channel names if available
            info_desc = self.lsl_stream_info.desc()
            channel_names = []
            if info_desc.child("channels"):
                ch = info_desc.child("channels").child("channel")
                while not ch.empty():
                    channel_names.append(ch.child_value("name"))
                    ch = ch.next_sibling("channel")
            
            if channel_names:
                self.eeg_names = channel_names[:len(self.eeg_channels)]
            
            # Update channel list based on actual stream
            self.eeg_channels = list(range(min(self.lsl_channel_count, len(self.eeg_channels))))
            
            logging.info(f"Connected to LSL stream: {self.lsl_stream_info.name()}")
            logging.info(f"Channels: {self.lsl_channel_count}, Sample rate: {self.sampling_rate} Hz")
            
            # Enable streaming
            self.is_streaming = True
            
            # Update GUI to show connection
            if hasattr(self, 'lsl_status'):
                self.lsl_status.setText(f"●  Connected: {self.lsl_stream_info.name()}")
                self.lsl_status.setStyleSheet("""
                    QLabel {
                        color: #4CAF50;
                        font-size: 12px;
                        font-weight: bold;
                        padding: 5px;
                        background-color: #2a2a2a;
                        border-radius: 4px;
                    }
                """)
                
            # Draw the plot again with updated channels
            self.win.clear()
            self.plot = self.init_plot()
            self.main_layout.addWidget(self.plot, 0, 0)
            self.win.setLayout(self.main_layout)
            
            # Start timers for data update and plot update
            self.data_timer = QtCore.QTimer()
            self.data_timer.timeout.connect(self.update_data_buffer)
            self.data_timer.start(self.update_data_buffer_speed_ms)
            self.plot_timer = QtCore.QTimer()
            self.plot_timer.timeout.connect(self.update_plot)
            self.plot_timer.start(self.update_plot_speed_ms)
            

    def create_parameters_panel(self):
        """Create buttons to interact with the neuroengine - Modern styled version"""

        # Main container
        parameters = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # ==================== CONNECTION STATUS ====================
        status_group = QtWidgets.QGroupBox()
        status_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #555;
                border-radius: 8px;
                margin: 5px;
                padding: 10px;
                background-color: #3a3a3a;
            }
        """)
        status_layout = QtWidgets.QVBoxLayout()

        status_label = QtWidgets.QLabel("Connection Status")
        status_label.setStyleSheet("color: white; font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        status_label.setAlignment(QtCore.Qt.AlignCenter)
        status_layout.addWidget(status_label)

        self.lsl_status = QtWidgets.QLabel("●  Not Connected")
        self.lsl_status.setStyleSheet("""
            QLabel {
                color: #f44336;
                font-size: 12px;
                font-weight: bold;
                padding: 5px;
                background-color: #2a2a2a;
                border-radius: 4px;
            }
        """)
        self.lsl_status.setAlignment(QtCore.Qt.AlignCenter)
        status_layout.addWidget(self.lsl_status)

        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)

        # ==================== EEG OPTIONS SECTION ====================
        eeg_group = QtWidgets.QGroupBox()
        eeg_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #555;
                border-radius: 8px;
                margin: 5px;
                padding: 10px;
                background-color: #3a3a3a;
            }
        """)
        eeg_layout = QtWidgets.QVBoxLayout()

        eeg_options_label = QtWidgets.QLabel("EEG Options")
        eeg_options_label.setStyleSheet("color: white; font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        eeg_options_label.setAlignment(QtCore.Qt.AlignCenter)
        eeg_layout.addWidget(eeg_options_label)

        self.save_data_checkbox = QtWidgets.QCheckBox('Save to CSV')
        self.save_data_checkbox.setStyleSheet("""
            QCheckBox {
                color: white;
                font-weight: bold;
                font-size: 12px;
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:checked {
                background-color: #4CAF50;
                border: 2px solid #45a049;
                border-radius: 3px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #555;
                border: 2px solid #777;
                border-radius: 3px;
            }
        """)
        eeg_layout.addWidget(self.save_data_checkbox)

        self.start_button = QtWidgets.QPushButton('Stop Plot')
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:pressed {
                background-color: #b71c1c;
            }
        """)
        self.start_button.clicked.connect(lambda: self.toggle_plot())
        eeg_layout.addWidget(self.start_button)

        eeg_group.setLayout(eeg_layout)
        main_layout.addWidget(eeg_group)

        # ==================== FILTERS SECTION ====================
        filters_group = QtWidgets.QGroupBox()
        filters_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #555;
                border-radius: 8px;
                margin: 5px;
                padding: 10px;
                background-color: #3a3a3a;
            }
        """)
        filters_layout = QtWidgets.QVBoxLayout()

        filters_label = QtWidgets.QLabel("Signal Filters")
        filters_label.setStyleSheet("color: white; font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        filters_label.setAlignment(QtCore.Qt.AlignCenter)
        filters_layout.addWidget(filters_label)

        bandpass_container = QtWidgets.QWidget()
        bandpass_layout = QtWidgets.QVBoxLayout()
        bandpass_layout.setSpacing(8)

        self.bandpass_checkbox = QtWidgets.QCheckBox('Bandpass Filter (Hz)')
        self.bandpass_checkbox.setStyleSheet("""
            QCheckBox {
                color: white;
                font-weight: bold;
                font-size: 12px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:checked {
                background-color: #2196F3;
                border: 2px solid #1976D2;
                border-radius: 3px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #555;
                border: 2px solid #777;
                border-radius: 3px;
            }
        """)
        bandpass_layout.addWidget(self.bandpass_checkbox)

        bandpass_inputs_layout = QtWidgets.QHBoxLayout()
        bandpass_inputs_layout.setSpacing(10)

        low_freq_layout = QtWidgets.QVBoxLayout()
        low_freq_layout.setSpacing(2)
        low_label = QtWidgets.QLabel("Low")
        low_label.setStyleSheet("color: #bbb; font-size: 10px;")
        low_label.setAlignment(QtCore.Qt.AlignCenter)

        self.bandpass_box_low = QtWidgets.QLineEdit()
        self.bandpass_box_low.setPlaceholderText('1.0')
        self.bandpass_box_low.setText('1')
        self.bandpass_box_low.setStyleSheet("""
            QLineEdit {
                background-color: #4a4a4a;
                border: 2px solid #666;
                border-radius: 4px;
                padding: 6px;
                color: white;
                font-size: 11px;
            }
            QLineEdit:focus {
                border-color: #2196F3;
            }
        """)
        self.bandpass_box_low.setMaximumWidth(50)

        low_freq_layout.addWidget(low_label)
        low_freq_layout.addWidget(self.bandpass_box_low)

        high_freq_layout = QtWidgets.QVBoxLayout()
        high_freq_layout.setSpacing(2)
        high_label = QtWidgets.QLabel("High")
        high_label.setStyleSheet("color: #bbb; font-size: 10px;")
        high_label.setAlignment(QtCore.Qt.AlignCenter)

        self.bandpass_box_high = QtWidgets.QLineEdit()
        self.bandpass_box_high.setPlaceholderText('40.0')
        self.bandpass_box_high.setText('40')
        self.bandpass_box_high.setStyleSheet("""
            QLineEdit {
                background-color: #4a4a4a;
                border: 2px solid #666;
                border-radius: 4px;
                padding: 6px;
                color: white;
                font-size: 11px;
            }
            QLineEdit:focus {
                border-color: #2196F3;
            }
        """)
        self.bandpass_box_high.setMaximumWidth(50)

        high_freq_layout.addWidget(high_label)
        high_freq_layout.addWidget(self.bandpass_box_high)

        bandpass_inputs_layout.addLayout(low_freq_layout)
        bandpass_inputs_layout.addWidget(QtWidgets.QLabel("-"))
        bandpass_inputs_layout.addLayout(high_freq_layout)
        bandpass_inputs_layout.addStretch()

        bandpass_layout.addLayout(bandpass_inputs_layout)
        bandpass_container.setLayout(bandpass_layout)
        filters_layout.addWidget(bandpass_container)

        filters_group.setLayout(filters_layout)
        main_layout.addWidget(filters_group)

        main_layout.addStretch()

        parameters.setLayout(main_layout)
        parameters.setMinimumWidth(320)
        parameters.setMaximumWidth(380)
        parameters.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: white;
            }
            QLabel {
                color: white;
            }
        """)

        self.bandpass_checkbox.stateChanged.connect(self.on_bandpass_toggled)
        self.on_bandpass_toggled(self.bandpass_checkbox.checkState())

        return parameters

    def on_bandpass_toggled(self, state):
        """Enable/disable bandpass filter inputs based on checkbox"""
        enabled = state == QtCore.Qt.Checked
        self.bandpass_box_low.setEnabled(enabled)
        self.bandpass_box_high.setEnabled(enabled)

        if enabled:
            self.bandpass_box_low.setStyleSheet(
                self.bandpass_box_low.styleSheet().replace('color: #666', 'color: white'))
            self.bandpass_box_high.setStyleSheet(
                self.bandpass_box_high.styleSheet().replace('color: #666', 'color: white'))
        else:
            disabled_style = """
                QLineEdit {
                    background-color: #333;
                    border: 2px solid #555;
                    border-radius: 4px;
                    padding: 6px;
                    color: #666;
                    font-size: 11px;
                }
            """
            self.bandpass_box_low.setStyleSheet(disabled_style)
            self.bandpass_box_high.setStyleSheet(disabled_style)

    def update_data_buffer(self):
        """Update the data buffer with new data from LSL stream"""
        if not self.is_streaming or not self.lsl_inlet:
            self.app.processEvents()
            return

        try:
            # Pull samples from LSL stream
            samples, timestamps = self.lsl_inlet.pull_chunk(timeout=0.0, max_samples=self.num_points)
                        
            if samples:
                # Convert to numpy array
                data = np.array(samples).T  # Transpose to get channels x samples
                
                # Store for plotting
                self.current_data = data
                
        except Exception as e:
            logging.error(f"Error updating data buffer: {e}")
        
        self.app.processEvents()

    def init_plot(self):
        """Initialize the timeseries plot for the EEG channels"""
        self.eeg_plot = pg.PlotWidget()

        self.eeg_plot.showAxis('left', False)
        self.eeg_plot.setMenuEnabled('left', True)
        self.eeg_plot.showAxis('bottom', True)
        self.eeg_plot.setMenuEnabled('bottom', True)
        self.eeg_plot.showGrid(x=True, y=True)
        self.eeg_plot.setLabel('bottom', text='Time (s)')

        self.eeg_plot.setTitle('EEG Channels')

        self.offset_amplitude = 200
        self.curves = []
        self.current_data = None

        for i in range(len(self.eeg_channels)):
            curve = self.eeg_plot.plot(pen=colors[i])
            self.curves.append(curve)

            text_item = pg.TextItem(text=str(self.eeg_names[i]) if i < len(self.eeg_names) else f"Ch{i}", 
                                   anchor=(1, 0.5))
            text_item.setPos(-10, i * self.offset_amplitude)
            self.eeg_plot.addItem(text_item)

        return self.eeg_plot

    def update_plot(self):
        """Update the plot with new data"""
        if not self.is_streaming or not self.plot or self.current_data is None:
            return

        try:
            data = self.current_data
            
            # Apply filters if enabled
            if self.bandpass_checkbox.isChecked():
                start_freq = float(self.bandpass_box_low.text()) if self.bandpass_box_low.text() else 1.0
                stop_freq = float(self.bandpass_box_high.text()) if self.bandpass_box_high.text() else 40.0
                
                for i in range(min(len(self.eeg_channels), data.shape[0])):
                    if data.shape[1] > 0:
                        self.apply_bandpass_filter(data[i], start_freq, stop_freq)

            # Update curves
            for i in range(min(len(self.curves), data.shape[0])):
                if data.shape[1] > 0:
                    ch_data_offset = data[i] + i * self.offset_amplitude
                    self.curves[i].setData(ch_data_offset)

            # Adjust Y range
            min_display = -self.offset_amplitude
            max_display = len(self.eeg_channels) * self.offset_amplitude
            self.eeg_plot.setYRange(min_display, max_display)

        except Exception as e:
            logging.error(f"Error updating plot: {e}")

        self.app.processEvents()

    def apply_bandpass_filter(self, ch_data, start_freq, stop_freq, order=4,
                              filter_type=FilterTypes.BUTTERWORTH_ZERO_PHASE, ripple=0):
        """Apply bandpass filter to channel data"""
        if len(ch_data) == 0:
            return
            
        DataFilter.detrend(ch_data, DetrendOperations.CONSTANT.value)
        
        if start_freq >= stop_freq:
            logging.error("Band-pass Filter: Start frequency should be less than stop frequency")
            return
        if start_freq < 0 or stop_freq < 0:
            logging.error("Band-pass Filter: Frequency values should be positive")
            return
        if start_freq > self.sampling_rate / 2 or stop_freq > self.sampling_rate / 2:
            logging.error("Band-pass Filter: Frequency values exceed Nyquist limit")
            return
            
        try:
            DataFilter.perform_bandpass(ch_data, self.sampling_rate, start_freq, stop_freq, order, filter_type, ripple)
        except ValueError as e:
            logging.error(f"Invalid frequency value {e}")

    def toggle_plot(self):
        """Start or stop the plotting of data"""
        if self.plot:
            self.start_button.setText('Start Plot')
            self.plot = False
        else:
            self.start_button.setText('Stop Plot')
            self.plot = True

    def quit(self):
        """Quit the application cleanly"""
        self.is_streaming = False
        if self.lsl_streamer.logging_thread:
            self.lsl_streamer.keep_logging = False
            self.lsl_streamer.logging_thread.join(timeout=1.0)


if __name__ == "__main__":
    config_path = pkg_resources.files("opencortex.configs").joinpath("Default.yaml")
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    app = QtWidgets.QApplication(sys.argv)
    gui = StreamerGUI(config_file="C:\\Users\\Greg\\dev\\1_OpenCortexBCI\\OpenCortexBCI\\opencortex\\configs\\Default.yaml")
    sys.exit(app.exec_())