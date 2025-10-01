# import re
# import bluetooth
# import argparse
# import time
# import mne
# import matplotlib.pyplot as plt
# import matplotlib
# from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets

# matplotlib.use("Qt5Agg")


# def retrieve_unicorn_devices():
#     saved_devices = bluetooth.discover_devices(duration=1, lookup_names=True, lookup_class=True)
#     unicorn_devices = filter(lambda x: re.search(r'UN-\d{4}.\d{2}.\d{2}', x[1]), saved_devices)
#     return list(unicorn_devices)


# def main():
#     BoardShim.enable_dev_board_logger()
#     # use synthetic board for demo
#     params = BrainFlowInputParams()

#     # Get bluetooth devices that match the UN-XXXX.XX.XX pattern
#     print(retrieve_unicorn_devices())
#     params.serial_number = retrieve_unicorn_devices()[0][1]

#     # Create a board object and prepare the session
#     board = BoardShim(BoardIds.UNICORN_BOARD.value, params)
#     board.prepare_session()
#     board.start_stream()

#     # Get data from the board, 10 seconds in this example, then close the session
#     time.sleep(10)
#     data = board.get_board_data()
#     board.stop_stream()
#     board.release_session()
#     eeg_channels = BoardShim.get_eeg_channels(BoardIds.UNICORN_BOARD.value)
#     eeg_data = data[eeg_channels, :]
#     eeg_data = eeg_data / 1e6  # BrainFlow returns uV, convert to V for MNE

#     # Creating MNE objects from brainflow data arrays
#     ch_types = ['eeg'] * len(eeg_channels)
#     ch_names = BoardShim.get_eeg_names(BoardIds.UNICORN_BOARD.value)
#     sfreq = BoardShim.get_sampling_rate(BoardIds.UNICORN_BOARD.value)
#     info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
#     raw = mne.io.RawArray(eeg_data, info)

#     # Plot the data using MNE
#     raw.plot()
#     raw.compute_psd().plot(average=True)
#     plt.show()
#     plt.savefig('psd.png')


# if __name__ == '__main__':
#     main()


# Unicorn.py
import re
import argparse
import time
import mne
import matplotlib.pyplot as plt
import matplotlib
from PyQt5 import QtBluetooth as QtBt
from PyQt5 import QtCore
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets

matplotlib.use("Qt5Agg")


class BluetoothScanner(QtCore.QObject):
    """Helper class to perform synchronous Bluetooth scanning using Qt's async API"""
    
    devices_found = QtCore.pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.discovered_devices = []
        self.agent = None
        
    def scan_devices(self, timeout_ms=5000):
        """Scan for Bluetooth devices with a timeout"""
        self.discovered_devices = []
        self.agent = QtBt.QBluetoothDeviceDiscoveryAgent(self)
        
        # Connect signals
        self.agent.deviceDiscovered.connect(self._on_device_discovered)
        self.agent.finished.connect(self._on_scan_finished)
        self.agent.error.connect(self._on_scan_error)
        
        # Set timeout for Low Energy devices
        self.agent.setLowEnergyDiscoveryTimeout(timeout_ms)
        
        # Start discovery
        self.agent.start()
        
    def _on_device_discovered(self, device_info):
        """Slot called when a device is discovered"""
        # Store device info as tuple: (address, name, device_class)
        address = device_info.address().toString()
        name = device_info.name()
        device_class = str(device_info.deviceUuid().toString())
        
        self.discovered_devices.append((address, name, device_class))
        
    def _on_scan_finished(self):
        """Slot called when scanning finishes"""
        self.devices_found.emit(self.discovered_devices)
        
    def _on_scan_error(self, error):
        """Slot called when an error occurs"""
        print(f"Bluetooth scan error: {self.agent.errorString()}")
        self.devices_found.emit(self.discovered_devices)


def retrieve_unicorn_devices():
    """Synchronous wrapper for Bluetooth device discovery"""
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QtCore.QCoreApplication([])
    
    scanner = BluetoothScanner()
    devices = []
    
    # Create event loop to wait for scan completion
    loop = QtCore.QEventLoop()
    
    def on_devices_found(found_devices):
        nonlocal devices
        devices = found_devices
        loop.quit()
    
    scanner.devices_found.connect(on_devices_found)
    scanner.scan_devices(timeout_ms=5000)
    
    # Wait for scan to complete
    loop.exec_()
    
    # Filter for Unicorn devices
    unicorn_devices = list(filter(lambda x: re.search(r'UN-\d{4}.\d{2}.\d{2}', x[1]), devices))
    return unicorn_devices


def main():
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()

    # Get bluetooth devices that match the UN-XXXX.XX.XX pattern
    print("Scanning for Unicorn devices...")
    unicorn_devices = retrieve_unicorn_devices()
    print(unicorn_devices)
    
    if not unicorn_devices:
        print("No Unicorn devices found!")
        return
    
    params.serial_number = unicorn_devices[0][1]

    # Create a board object and prepare the session
    board = BoardShim(BoardIds.UNICORN_BOARD.value, params)
    board.prepare_session()
    board.start_stream()

    # Get data from the board, 10 seconds in this example, then close the session
    time.sleep(10)
    data = board.get_board_data()
    board.stop_stream()
    board.release_session()
    
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.UNICORN_BOARD.value)
    eeg_data = data[eeg_channels, :]
    eeg_data = eeg_data / 1e6  # BrainFlow returns uV, convert to V for MNE

    # Creating MNE objects from brainflow data arrays
    ch_types = ['eeg'] * len(eeg_channels)
    ch_names = BoardShim.get_eeg_names(BoardIds.UNICORN_BOARD.value)
    sfreq = BoardShim.get_sampling_rate(BoardIds.UNICORN_BOARD.value)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(eeg_data, info)

    # Plot the data using MNE
    raw.plot()
    raw.compute_psd().plot(average=True)
    plt.show()
    plt.savefig('psd.png')


if __name__ == '__main__':
    main()
