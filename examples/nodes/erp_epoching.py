import logging

import numpy as np

from opencortex.neuroengine.flux.preprocessing.epochs import EpochingNode
from opencortex.neuroengine.flux.preprocessing.events import ExtractEventsNode, FilterEventsNode, RelabelEventsNode
from opencortex.neuroengine.flux.base.sequential import Sequential

if __name__ == "__main__":
    import opencortex.neuroengine.flux.base.operators  # Enable >>
    from opencortex.neuroengine.flux.preprocessing.bandpass import BandPassFilterNode
    from opencortex.neuroengine.flux.preprocessing.notch import NotchFilterNode
    from opencortex.utils.loader import load_data, convert_to_mne
    import matplotlib.pyplot as plt

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Configuration
    fs = 250
    chs = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]

    # Load data
    print("=" * 60)
    print("Loading EEG data for ERP analysis...")
    print("=" * 60)
    eeg, trigger, dataframe = load_data(
        "../../data/aep/auditory_erp_eyes_open_S1.csv",
        fs=fs,
        skiprows=5,
        delimiter=','
    )

    # Convert to MNE format
    raw_data = convert_to_mne(
        eeg, trigger, fs=fs, chs=chs, recompute=False
    )

    # ========================================================================
    # APPROACH 1: Manual chaining (explicit)
    # ========================================================================
    print("\n" + "=" * 60)
    print("APPROACH 1: Manual pipeline with explicit chaining")
    print("=" * 60)

    # Signal preprocessing (traditional pipeline with >>)
    filter_pipeline = (
            NotchFilterNode((50, 60), name="PowerlineNotch") >>
            BandPassFilterNode(0.1, 15.0, name="ERPBand")
    )

    # Apply signal filters
    print("\nApplying signal filters...")
    filtered_data = filter_pipeline(raw_data)

    # Event processing nodes
    extract_events = ExtractEventsNode(stim_channel='STI', name="EventExtractor")
    filter_events = FilterEventsNode(max_event_id=90, name="EventFilter")
    relabel_events = RelabelEventsNode(target_class=1, nontarget_label=3, name="EventRelabeler")
    epoching = EpochingNode(
        tmin=-0.2,
        tmax=0.8,
        baseline=(-0.1, 0.0),
        event_id={'T': 1, 'NT': 3},
        name="Epocher"
    )

    # Chain event processing manually
    print("\nProcessing events...")
    result = extract_events(filtered_data)
    result = filter_events(result)
    result = relabel_events(result)
    epochs_data, labels = epoching(result)

    print(f"\nFinal epochs shape: {epochs_data.shape}")
    print(f"Labels distribution: {np.unique(labels, return_counts=True)}")

    # ========================================================================
    # APPROACH 2: Using Sequential for event pipeline
    # ========================================================================
    print("\n" + "=" * 60)
    print("APPROACH 2: Using Sequential for complete pipeline")
    print("=" * 60)


    # Create a complete pipeline using Sequential
    complete_pipeline = Sequential(
        NotchFilterNode((50, 60), name="PowerlineNotch"),
        BandPassFilterNode(0.1, 15.0, name="ERPBand"),
        ExtractEventsNode(stim_channel='STI', name="EventExtractor"),
        FilterEventsNode(max_event_id=90, name="EventFilter"),
        RelabelEventsNode(target_class=1, nontarget_label=3, name="EventRelabeler"),
        EpochingNode(
            tmin=-0.2,
            tmax=0.8,
            baseline=(-0.1, 0.0),
            event_id={'T': 1, 'NT': 3},
            name="Epocher"
        ),
        name="CompleteERPPipeline"
    )

    print("\nApplying complete pipeline...")
    epochs_data2, labels2 = complete_pipeline(raw_data)

    print(f"\nFinal epochs shape: {epochs_data2.shape}")
    print(f"Labels distribution: {np.unique(labels2, return_counts=True)}")

    # ========================================================================
    # Visualization
    # ========================================================================
    print("\nPlotting averaged ERPs...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    times = np.linspace(epoching.tmin, epoching.tmax, epochs_data.shape[2])

    for ch_idx, ch_name in enumerate(chs):
        ax = axes[ch_idx]

        # Plot target and non-target separately
        target_epochs = epochs_data[labels == 1, ch_idx, :]
        nontarget_epochs = epochs_data[labels == 3, ch_idx, :]

        if len(target_epochs) > 0:
            ax.plot(times, target_epochs.mean(axis=0), 'r-',
                    label=f'Target (n={len(target_epochs)})', linewidth=2)
            ax.fill_between(
                times,
                target_epochs.mean(axis=0) - target_epochs.std(axis=0),
                target_epochs.mean(axis=0) + target_epochs.std(axis=0),
                alpha=0.3, color='r'
            )

        if len(nontarget_epochs) > 0:
            ax.plot(times, nontarget_epochs.mean(axis=0), 'b-',
                    label=f'Non-Target (n={len(nontarget_epochs)})', linewidth=2)
            ax.fill_between(
                times,
                nontarget_epochs.mean(axis=0) - nontarget_epochs.std(axis=0),
                nontarget_epochs.mean(axis=0) + nontarget_epochs.std(axis=0),
                alpha=0.3, color='b'
            )

        ax.axvline(0, color='k', linestyle='--', alpha=0.5, label='Stimulus')
        ax.axhline(0, color='k', linestyle='-', alpha=0.3)
        ax.set_title(f'{ch_name}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (µV)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Event-Related Potentials (ERPs) - Modular Pipeline',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Summary
    print("\n" + "=" * 60)
    print("Pipeline Summary:")
    print("=" * 60)
    print(complete_pipeline)
    print("=" * 60)