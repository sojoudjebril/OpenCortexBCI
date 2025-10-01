import opencortex.neuroengine.flux.base.operators  # Enable >>
from opencortex.neuroengine.flux.preprocessing.bandpass import BandPassFilterNode
from opencortex.neuroengine.flux.preprocessing.notch import NotchFilterNode
from opencortex.utils.loader import load_data, convert_to_mne
import matplotlib.pyplot as plt

fs = 250
chs = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]


eeg, trigger, dataframe = load_data("../../../data/aep/auditory_erp_eyes_open_S1.csv", fs=fs, skiprows=5, delimiter=',')
print("Loaded data with shape:" + str(eeg.shape) + " and trigger shape: " + str(trigger.shape))
print("That means we have " + str(eeg.shape[0]) + " samples and " + str(eeg.shape[1]) + " channels.")

 # Convert to MNE format
raw_data = convert_to_mne(eeg, trigger, fs=fs, chs=chs, recompute=False) # recompute=True to recalculate the event labels if the values are negative
plt.plot(trigger)
plt.show()

Pxx = raw_data.compute_psd(fmin=0, fmax=fs/2)
Pxx.plot()
plt.show()

preprocessing = NotchFilterNode((50, 60)) >> BandPassFilterNode(1.0, 30.0)
filtered_data = preprocessing(raw_data)

Pxx = filtered_data.compute_psd(fmin=0, fmax=fs/2)
Pxx.plot()
plt.show()