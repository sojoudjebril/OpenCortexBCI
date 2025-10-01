import opencortex.neuroengine.flux.base.operators  # Enable >>
import matplotlib.pyplot as plt
import numpy as np
from opencortex.neuroengine.flux.preprocessing.bandpass import BandPassFilterNode
from opencortex.neuroengine.flux.preprocessing.notch import NotchFilterNode
from opencortex.utils.loader import load_data, convert_to_mne
from opencortex.neuroengine.flux.preprocessing.events import ExtractEventsNode

if __name__ == "__main__":

    # Configuration
    fs = 250
    chs = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]

    # Load data
    print("=" * 60)
    print("Loading EEG data...")
    print("=" * 60)
    eeg, trigger, dataframe = load_data(
        "../../data/aep/auditory_erp_eyes_open_S1.csv",
        fs=fs,
        skiprows=5,
        delimiter=','
    )
    print(f"Loaded data with shape: {eeg.shape} and trigger shape: {trigger.shape}")
    print(f"That means we have {eeg.shape[0]} samples and {eeg.shape[1]} channels.\n")

    # Convert to MNE format
    raw_data = convert_to_mne(
        eeg, trigger, fs=fs, chs=chs, recompute=False
    )

    # Plot original trigger channel
    plt.figure(figsize=(12, 3))
    plt.plot(trigger)
    plt.title("Original Trigger Channel")
    plt.xlabel("Samples")
    plt.ylabel("Trigger Value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot original PSD
    print("Computing PSD for raw data...")
    Pxx = raw_data.compute_psd(fmin=0, fmax=fs / 2)
    Pxx.plot()

    # Create preprocessing pipeline with event extraction
    print("\nCreating preprocessing pipeline...")
    ev_ids = {'T': 1, 'NT': 2}
    event_colors = {1: 'g', 2: 'r'}  # Green for target, Red for non-target
    preprocessing = (
            NotchFilterNode((50, 60), name="PowerlineNotch") >>
            BandPassFilterNode(1.0, 30.0, name="BetaBand") >>
            ExtractEventsNode(stim_channel='STI',
                              auto_label=True,
                              name="EventExtractor",
                              ev_ids=ev_ids,
                              event_color=event_colors)
    )

    # Apply preprocessing
    print("Applying preprocessing pipeline...")
    filtered_data = preprocessing(raw_data)

    # Get the event extraction node (last in pipeline)
    event_extractor = preprocessing.steps[-1]
    events, event_ids, event_colors = event_extractor.get_events()

    print(f"\n{event_extractor}")
    print(f"Event IDs: {event_ids}")
    print(f"Event colors: {event_colors}")
    print(f"Number of events: {len(events)}")

    # Plot filtered PSD
    print("\nComputing PSD for filtered data...")
    Pxx_filtered = filtered_data.compute_psd(fmin=0, fmax=fs / 2)
    Pxx_filtered.plot()

    # Plot raw data with events
    print("\nPlotting raw data with extracted events...")
    fig = filtered_data.plot(
        events=events,
        event_id=event_ids,
        event_color=event_colors,
        duration=10,  # Show 10 seconds at a time
        scalings='auto',
        title="Filtered EEG Data with Events"
    )
    plt.show()

    # Plot events only
    print("\nPlotting event distribution...")
    plt.figure(figsize=(14, 4))

    # Event timeline
    plt.subplot(1, 2, 1)
    for event_name, event_id in event_ids.items():
        event_times = events[events[:, 2] == event_id, 0] / fs  # Convert to seconds
        plt.scatter(
            event_times,
            [event_id] * len(event_times),
            label=event_name,
            c=event_colors.get(event_id, 'b'),
            s=50,
            alpha=0.7
        )
    plt.xlabel("Time (s)")
    plt.ylabel("Event ID")
    plt.title("Event Timeline")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Event histogram
    plt.subplot(1, 2, 2)
    event_types = events[:, 2]
    unique_events, counts = np.unique(event_types, return_counts=True)
    colors = [event_colors.get(ev_id, 'b') for ev_id in unique_events]
    labels = [event_name for event_name, ev_id in event_ids.items() if ev_id in unique_events]

    plt.bar(range(len(unique_events)), counts, color=colors, alpha=0.7)
    plt.xticks(range(len(unique_events)), labels)
    plt.xlabel("Event Type")
    plt.ylabel("Count")
    plt.title("Event Distribution")
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
