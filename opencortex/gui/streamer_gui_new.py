# """
# This class creates a GUI to plot and handle LSL events.
# Filters can be applied to the data.
# """
import sys
import importlib.resources as pkg_resources
import logging
import time
import yaml
import numpy as np
import pyqtgraph as pg
from pyqtgraph import mkBrush, ScatterPlotItem
from pyqtgraph.Qt import QtWidgets, QtCore
from PyQt5.QtCore import Qt

from opencortex.gui.widgets.stream_select_widget import LSLSelector, LSLReceiver  


class StreamerGUI(QtWidgets.QWidget):
    """Main GUI for real-time EEG LSL streaming and visualization."""

    def __init__(self, config_file='default_config.yaml'):
        super().__init__()
        logging.info("Initializing Streamer GUI...")

        # --- Load configuration ---
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        self.sampling_rate = config.get("sampling_rate", 250)
        self.window_size = config.get("window_size", 1)
        self.update_interval_ms = config.get("update_interval_ms", 100)
        self.num_points = int(self.window_size * self.sampling_rate)

        self.offset_amplitude = config.get("offset_amplitude", 200)
        self.eeg_names = config.get("channel_names", [])
        self.colors = [pg.intColor(i, hues=16) for i in range(16)]

        # --- State variables ---
        self.is_streaming = False
        self.lsl_inlet = None
        self.receiver = None
        self.data_buffer = None
        # For plotting
        self.curves = []

        # --- Initialize GUI ---
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title='OpenCortex Streamer', size=(1920, 1080))
        self.win.setWindowTitle('OpenCortex Streamer')
        self.win.setWindowIcon(QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.SP_ComputerIcon))
        self.main_layout = QtWidgets.QGridLayout()
        self.win.setLayout(self.main_layout)

        # --- LSL selector panel ---
        self.lsl_streamer = LSLSelector()
        self.lsl_streamer.setStyleSheet("background-color: #2b2b2b; color: white;")
        self.lsl_streamer.stream_selected.connect(self.on_lsl_stream_selected)

        # --- Add widgets ---
        self.plot = self.init_plot()
        side_panel_widget = QtWidgets.QWidget()
        side_panel_layout = QtWidgets.QVBoxLayout()
        side_panel_layout.addWidget(self.lsl_streamer)
        side_panel_widget.setLayout(side_panel_layout)

        self.main_layout.addWidget(self.plot, 0, 0)
        self.main_layout.addWidget(side_panel_widget, 0, 1, alignment=Qt.AlignCenter)

        self.win.setLayout(self.main_layout)
        self.win.show()

        # --- Timer for periodic plot updates ---
        self.plot_timer = QtCore.QTimer()
        self.plot_timer.timeout.connect(self.update_plot)
        self.plot_timer.start(self.update_interval_ms)

        # Run app
        self.app.exec_()

    # ----------------------------------------------------------------------
    # LSL CONNECTION HANDLING
    # ----------------------------------------------------------------------

    def on_lsl_stream_selected(self, inlet):
        """Callback when the user selects an LSL EEG stream."""
        self.lsl_inlet = inlet
        if self.lsl_inlet is None:
            logging.warning("No LSL inlet selected.")
            self.curves = []
            self.is_streaming = False
            return
        info = inlet.info()

        self.lsl_channel_count = info.channel_count()
        self.sampling_rate = info.nominal_srate()
        self.num_points = int(self.window_size * self.sampling_rate)

        # Update EEG names from LSL metadata if available
        desc = info.desc()
        channel_names = []
        if desc.child("channels"):
            ch = desc.child("channels").child("channel")
            while not ch.empty():
                channel_names.append(ch.child_value("name"))
                ch = ch.next_sibling("channel")
        if channel_names:
            self.eeg_names = channel_names

        logging.info(f"Connected to LSL stream: {info.name()} ({self.lsl_channel_count} ch @ {self.sampling_rate} Hz)")

        # Initialize data buffer
        self.data_buffer = np.zeros((self.lsl_channel_count, self.num_points), dtype=np.float32)

        # Reinitialize plot
        self.refresh_plot()

        # Start LSL receiver thread
        self.receiver = LSLReceiver(inlet)
        self.receiver.data_received.connect(self.on_new_sample)
        self.receiver.start()
        self.is_streaming = True

    # ----------------------------------------------------------------------
    # DATA HANDLING
    # ----------------------------------------------------------------------

    def on_new_sample(self, sample, timestamp):
        """Receive a new EEG sample from LSLReceiver and update the circular buffer."""
        if not self.is_streaming or self.data_buffer is None:
            return
        sample = np.array(sample, dtype=np.float32)
        self.data_buffer = np.roll(self.data_buffer, -1, axis=1)
        self.data_buffer[:, -1] = sample

    # ----------------------------------------------------------------------
    # PLOTTING
    # ----------------------------------------------------------------------

    def init_plot(self):
        """Initialize the pyqtgraph plot widget."""
        eeg_plot = pg.PlotWidget()
        eeg_plot.showGrid(x=True, y=True)
        eeg_plot.setLabel('bottom', 'Time (s)')
        eeg_plot.setYRange(-self.offset_amplitude, 16 * self.offset_amplitude)
        eeg_plot.setTitle("EEG Stream")

        self.curves = []
        for i, name in enumerate(self.eeg_names or [f"Ch{i+1}" for i in range(16)]):
            curve = eeg_plot.plot(pen=pg.mkPen(self.colors[i % len(self.colors)], width=1))
            text = pg.TextItem(text=name, anchor=(1, 0.5))
            text.setPos(-10, i * self.offset_amplitude)
            eeg_plot.addItem(text)
            self.curves.append(curve)
        return eeg_plot

    def refresh_plot(self):
        """Recreate the plot when a new stream is connected."""
        old_plot = self.plot
        self.plot = self.init_plot()
        self.main_layout.replaceWidget(old_plot, self.plot)
        old_plot.deleteLater()

    def update_plot(self):
        """Efficiently update curves from the EEG data buffer."""
        if not self.is_streaming or self.data_buffer is None:
            return
        for i, curve in enumerate(self.curves):
            ch_data = self.data_buffer[i] + i * self.offset_amplitude
            curve.setData(ch_data)

    # ----------------------------------------------------------------------
    # CLEANUP
    # ----------------------------------------------------------------------

    def closeEvent(self, event):
        """Stop the LSL receiver thread cleanly on exit."""
        if self.receiver:
            self.receiver.stop()
        self.is_streaming = False
        super().closeEvent(event)


if __name__ == "__main__":
    config_path = pkg_resources.files("opencortex.configs").joinpath("Default.yaml")
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    app = QtWidgets.QApplication(sys.argv)
    gui = StreamerGUI(config_file="C:\\Users\\Greg\\dev\\1_OpenCortexBCI\\OpenCortexBCI\\opencortex\\configs\\Default.yaml")
    sys.exit(app.exec_())
