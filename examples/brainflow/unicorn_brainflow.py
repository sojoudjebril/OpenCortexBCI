import re
import time
import sys
import mne
import matplotlib
import matplotlib.pyplot as plt
from PyQt5.QtCore import QCoreApplication, QEventLoop, QTimer
from PyQt5.QtBluetooth import QBluetoothDeviceDiscoveryAgent
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

matplotlib.use("Qt5Agg")


# TODO: check if this works on Mac, Linux, Windows with real devices. (For Mike)

def retrieve_unicorn_devices():
    app = QCoreApplication(sys.argv)  # Required Qt event loop

    discovered_devices = []

    loop = QEventLoop()

    def device_found(device):
        # Append devices whose name matches the UN-XXXX.XX.XX pattern
        if re.search(r'UN-\d{4}.\d{2}.\d{2}', device.name()):
            # Append tuple (address, name, device class)
            discovered_devices.append((device.address().toString(), device.name(), device.deviceClass()))

    def finished():
        loop.quit()

    agent = QBluetoothDeviceDiscoveryAgent()
    agent.deviceDiscovered.connect(device_found)
    agent.finished.connect(finished)
    agent.start()

    # Use a timeout in case discovery never finishes (10 seconds here)
    QTimer.singleShot(10000, loop.quit)

    loop.exec_()  # Run the event loop until discovery finishes or timeout

    return discovered_devices


def main():
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()

    unicorn_devices = retrieve_unicorn_devices()
    print(unicorn_devices)
    if unicorn_devices:
        params.serial_number = unicorn_devices[0][1]
    else:
        print("No Unicorn devices found.")
        return

    board = BoardShim(BoardIds.UNICORN_BOARD.value, params)
    board.prepare_session()
    board.start_stream()

    time.sleep(10)
    data = board.get_board_data()
    board.stop_stream()
    board.release_session()
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.UNICORN_BOARD.value)
    eeg_data = data[eeg_channels, :]
    eeg_data = eeg_data / 1e6

    ch_types = ['eeg'] * len(eeg_channels)
    ch_names = BoardShim.get_eeg_names(BoardIds.UNICORN_BOARD.value)
    sfreq = BoardShim.get_sampling_rate(BoardIds.UNICORN_BOARD.value)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(eeg_data, info)

    raw.plot()
    raw.compute_psd().plot(average=True)
    plt.show()
    plt.savefig('psd.png')


if __name__ == '__main__':
    main()

