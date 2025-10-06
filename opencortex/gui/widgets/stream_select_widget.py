import sys
import threading
import time
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QRadioButton,
    QGroupBox, QPushButton, QLabel
)
from pylsl import resolve_streams, StreamInlet

class LSLSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LSL EEG Stream Selector")

        layout = QVBoxLayout()

        # Label for info
        self.label = QLabel("Select an EEG stream:")
        layout.addWidget(self.label)

        # GroupBox to hold radio buttons
        self.groupBox = QGroupBox("Available EEG Streams")
        self.vbox = QVBoxLayout()
        self.groupBox.setLayout(self.vbox)
        layout.addWidget(self.groupBox)

        # Refresh button
        self.refresh_button = QPushButton("Refresh Streams")
        self.refresh_button.clicked.connect(self.load_streams)
        layout.addWidget(self.refresh_button)

        # Confirm button
        self.confirm_button = QPushButton("Confirm Selection")
        self.confirm_button.clicked.connect(self.confirm_selection)
        layout.addWidget(self.confirm_button)

        self.setLayout(layout)
        self.stream_buttons = []
        self.selected_stream = None
        self.inlet = None
        self.logging_thread = None
        self.keep_logging = False

        self.load_streams()  # Initial load

    def load_streams(self):
        # Clear previous buttons
        for btn in self.stream_buttons:
            self.vbox.removeWidget(btn)
            btn.deleteLater()
        self.stream_buttons.clear()

        # Resolve available streams
        streams = resolve_streams()
        eeg_streams = [s for s in streams if s.type() == "EEG"]

        if not eeg_streams:
            self.label.setText("No EEG streams found.")
            return
        else:
            self.label.setText("Select an EEG stream:")

        for i, s in enumerate(eeg_streams):
            name = s.name()
            source_id = s.source_id()
            rb = QRadioButton(f"{name} (ID: {source_id})")
            rb.stream = s  # store the stream object
            self.vbox.addWidget(rb)
            self.stream_buttons.append(rb)

    def confirm_selection(self):
        for rb in self.stream_buttons:
            if rb.isChecked():
                self.selected_stream = rb.stream
                self.label.setText(f"Selected: {rb.text()}")
                print(f"Selected stream: {rb.stream.name()} (ID: {rb.stream.source_id()})")
                self.attach_and_log()
                break

    def attach_and_log(self):
        if not self.selected_stream:
            return

        # Create inlet
        self.inlet = StreamInlet(self.selected_stream)
        info = self.inlet.info()
        
        info_desc = info.desc()
        print(info_desc)
        
        channel_names = []
        if info_desc.child("channels"):
            ch = info_desc.child("channels").child("channel")
            while not ch.empty():
                channel_names.append(ch.child_value("name"))
                ch = ch.next_sibling("channel")
                    
        print(f"Channel names: {channel_names}")

        # Log stream metadata
        print("Attached to stream:")
        print(f"  Name: {info.name()}")
        print(f"  Type: {info.type()}")
        print(f"  Channel count: {info.channel_count()}")
        print(f"  Sampling rate: {info.nominal_srate()} Hz")
        print(f"  Source ID: {info.source_id()}")

        # Start background logging
        self.keep_logging = True
        self.logging_thread = threading.Thread(target=self.pull_data, daemon=True)
        self.logging_thread.start()

    def pull_data(self):
        """ Continuously pull samples and log them. """
        while self.keep_logging:
            sample, timestamp = self.inlet.pull_sample(timeout=1.0)
            if sample is not None:
                # print(f"[{timestamp:.3f}] {sample}")
                pass
            time.sleep(0.01)  # avoid hammering CPU
        return sample

    def closeEvent(self, event):
        """ Stop logging cleanly when GUI is closed. """
        self.keep_logging = False
        if self.logging_thread:
            self.logging_thread.join(timeout=1.0)
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = LSLSelector()
    win.show()
    sys.exit(app.exec_())
