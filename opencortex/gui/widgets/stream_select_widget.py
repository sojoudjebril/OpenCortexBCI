import sys
import time
import logging
from PyQt5.QtCore import pyqtSignal, QThread, QObject
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QRadioButton,
    QGroupBox, QPushButton, QLabel
)
from pylsl import resolve_streams, StreamInlet


class LSLSelector(QWidget):
    # Signal emitted when a valid LSL inlet is created
    stream_selected = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LSL EEG Stream Selector")

        layout = QVBoxLayout()

        self.label = QLabel("Select an EEG stream:")
        layout.addWidget(self.label)

        self.groupBox = QGroupBox("Available EEG Streams")
        self.vbox = QVBoxLayout()
        self.groupBox.setLayout(self.vbox)
        layout.addWidget(self.groupBox)

        self.refresh_button = QPushButton("Refresh Streams")
        self.refresh_button.clicked.connect(self.load_streams)
        layout.addWidget(self.refresh_button)

        self.confirm_button = QPushButton("Confirm Selection")
        self.confirm_button.clicked.connect(self.confirm_selection)
        layout.addWidget(self.confirm_button)

        self.setLayout(layout)

        self.stream_buttons = []
        self.selected_stream = None
        self.inlet = None

        self.load_streams()

    def load_streams(self):
        '''Clean up previous streams and scan for new ones.'''
        if self.inlet:
            self.inlet.close_stream()
            self.inlet = None
            self.stream_selected.emit(self.inlet)
            
        
        """Scan for EEG LSL streams."""
        for btn in self.stream_buttons:
            self.vbox.removeWidget(btn)
            btn.deleteLater()
        self.stream_buttons.clear()

        self.label.setText("Scanning for EEG streams...")
        QApplication.processEvents()

        try:
            streams = resolve_streams()
        except Exception as e:
            self.label.setText(f"Error: {e}")
            return

        eeg_streams = [s for s in streams if s.type().lower() == "eeg"]

        if not eeg_streams:
            self.label.setText("No EEG streams found.")
            return

        self.label.setText("Select an EEG stream:")
        for s in eeg_streams:
            rb = QRadioButton(f"{s.name()} (ID: {s.source_id()})")
            rb.stream = s
            self.vbox.addWidget(rb)
            self.stream_buttons.append(rb)

    def confirm_selection(self):
        """Return the selected inlet to the parent."""
        for rb in self.stream_buttons:
            if rb.isChecked():
                self.selected_stream = rb.stream
                self.label.setText(f"Selected: {rb.text()}")
                self.inlet = StreamInlet(self.selected_stream)
                logging.info(f"Connected to stream {rb.stream.name()}")
                self.stream_selected.emit(self.inlet)
                break


class LSLReceiver(QThread):
    """A QThread-based LSL receiver that continuously pulls EEG samples."""
    data_received = pyqtSignal(object, float)

    def __init__(self, inlet: StreamInlet, parent=None):
        super().__init__(parent)
        self.inlet = inlet
        self.running = True

    def run(self):
        while self.running:
            sample, ts = self.inlet.pull_sample(timeout=0.05)
            if sample is not None:
                self.data_received.emit(sample, ts)
            time.sleep(0.01)

    def stop(self):
        self.running = False
        self.wait(timeout=500)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    selector = LSLSelector()
    selector.show()

    def on_stream(inlet):
        print(f"Stream selected: {inlet.info().name()} ({inlet.info().source_id()})")

    selector.stream_selected.connect(on_stream)
    sys.exit(app.exec_())

