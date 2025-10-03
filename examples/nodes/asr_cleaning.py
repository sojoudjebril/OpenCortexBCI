from opencortex.neuroengine.flux.preprocessing.asr import ASRNode

if __name__ == "__main__":
    import opencortex.neuroengine.flux.base.operators  # Enable >>
    from opencortex.neuroengine.flux.preprocessing.bandpass import BandPassFilterNode
    from opencortex.neuroengine.flux.preprocessing.notch import NotchFilterNode
    from opencortex.utils.loader import load_data, convert_to_mne
    import matplotlib.pyplot as plt
    import numpy as np
    import logging

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Configuration
    fs = 250
    chs = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]

    # Load data
    print("=" * 60)
    print("Loading EEG data for ASR cleaning...")
    print("=" * 60)
    eeg, trigger, dataframe = load_data(
        "../../data/aep/auditory_erp_eyes_open_S1.csv",
        fs=fs,
        skiprows=5,
        delimiter=','
    )
    print(f"Loaded data with shape: {eeg.shape}")
    print(f"Total recording time: {eeg.shape[0] / fs:.1f} seconds\n")

    # Convert to MNE format
    raw_data = convert_to_mne(
        eeg, trigger, fs=fs, chs=chs, recompute=False, rescale=1e6
    )

    # ========================================================================
    # Visualization: Before ASR
    # ========================================================================
    print("Plotting raw data before ASR...")

    # Plot a segment of raw data
    fig = plt.figure(figsize=(15, 8))

    # Time series plot
    plt.subplot(2, 1, 1)
    times = np.arange(raw_data.n_times) / fs
    max_peak = np.max(np.abs(raw_data.get_data(picks=chs)))
    for i, ch in enumerate(chs):
        ch_data = raw_data.get_data(picks=[ch])[0]
        plt.plot(times, ch_data + max_peak * i, label=ch, alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.title('Raw EEG Data - Before ASR')
    plt.legend(loc='upper right', ncol=len(chs))
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 50)  # Show first 20 seconds

    # PSD plot
    plt.subplot(2, 1, 2)
    psd_raw = raw_data.compute_psd(fmin=0, fmax=fs / 2)
    psd_raw.plot(axes=plt.gca(), show=False)
    plt.title('Power Spectral Density - Before ASR')
    plt.tight_layout()
    plt.show()

    # ========================================================================
    # Apply ASR with calibration on first 15 seconds
    # ========================================================================
    print("\n" + "=" * 60)
    print("Applying ASR with 15-second calibration")
    print("=" * 60)

    # Create preprocessing pipeline with ASR
    # Use first 15 seconds for calibration
    preprocessing_with_asr = (
            NotchFilterNode((50, 60), name="PowerlineNotch") >>
            BandPassFilterNode(0.5, 40.0, name="BroadBand") >>
            ASRNode(
                sfreq=fs,
                cutoff=5.0,  # Standard cutoff - lower = more aggressive
                calibration_time=5.0,  # Use first 5 seconds,
                calibrate=True,
                name="ASR_Cleaner"
            )
    )

    # Apply pipeline
    print("\nApplying preprocessing pipeline with ASR...")
    cleaned_data = preprocessing_with_asr(raw_data)

    # ========================================================================
    # Visualization: After ASR
    # ========================================================================
    print("\nPlotting cleaned data after ASR...")

    fig = plt.figure(figsize=(15, 8))

    # Time series plot
    plt.subplot(2, 1, 1)
    times = np.arange(cleaned_data.n_times) / fs
    max_peak = np.max(np.abs(cleaned_data.get_data(picks=chs)))
    for i, ch in enumerate(chs):
        ch_data = cleaned_data.get_data(picks=[ch])[0]
        plt.plot(times, ch_data + max_peak * i, label=ch, alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.title('Cleaned EEG Data - After ASR')
    plt.legend(loc='upper right', ncol=len(chs))
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 50)  # Show first 20 seconds

    # PSD plot
    plt.subplot(2, 1, 2)
    psd_cleaned = cleaned_data.compute_psd(fmin=0, fmax=fs / 2)
    psd_cleaned.plot(axes=plt.gca(), show=False)
    plt.title('Power Spectral Density - After ASR')

    plt.tight_layout()
    plt.show()

    # ========================================================================
    # Comparison plot
    # ========================================================================
    print("\nCreating comparison plot...")

    fig, axes = plt.subplots(len(chs), 1, figsize=(15, 12))

    # Show a 10-second window where artifacts might be visible
    start_time = 20  # seconds
    end_time = 50  # seconds
    start_sample = int(start_time * fs)
    end_sample = int(end_time * fs)
    times_segment = np.arange(start_sample, end_sample) / fs

    for i, ch in enumerate(chs):
        ax = axes[i]

        # Get data segments
        raw_segment = raw_data.get_data(picks=[ch])[0, start_sample:end_sample]
        cleaned_segment = cleaned_data.get_data(picks=[ch])[0, start_sample:end_sample]

        # Plot
        ax.plot(times_segment, raw_segment, 'r-', alpha=0.5, label='Before ASR', linewidth=1)
        ax.plot(times_segment, cleaned_segment, 'b-', alpha=0.7, label='After ASR', linewidth=1)

        ax.set_ylabel(f'{ch}\n(µV)', rotation=0, ha='right', va='center')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        if i == 0:
            ax.set_title(f'ASR Artifact Removal Comparison ({start_time}-{end_time}s)')
        if i == len(chs) - 1:
            ax.set_xlabel('Time (s)')
        else:
            ax.set_xticks([])

    plt.tight_layout()
    plt.show()

    # ========================================================================
    # Compare artifact statistics
    # ========================================================================
    print("\n" + "=" * 60)
    print("Artifact Removal Statistics")
    print("=" * 60)

    for ch in chs:
        raw_ch = raw_data.get_data(picks=[ch])[0]
        cleaned_ch = cleaned_data.get_data(picks=[ch])[0]

        # Calculate statistics
        raw_std = np.std(raw_ch)
        cleaned_std = np.std(cleaned_ch)
        raw_max = np.max(np.abs(raw_ch))
        cleaned_max = np.max(np.abs(cleaned_ch))

        reduction_std = ((raw_std - cleaned_std) / raw_std) * 100
        reduction_max = ((raw_max - cleaned_max) / raw_max) * 100

        print(f"\n{ch}:")
        print(f"  Std Dev:  {raw_std:.2f} → {cleaned_std:.2f} µV ({reduction_std:+.1f}%)")
        print(f"  Max Amp:  {raw_max:.2f} → {cleaned_max:.2f} µV ({reduction_max:+.1f}%)")

    # ========================================================================
    # Try different cutoff values
    # ========================================================================
    print("\n" + "=" * 60)
    print("Comparing different ASR cutoff values")
    print("=" * 60)

    cutoffs = [3.0, 5.0, 10.0, 20.0]

    fig, axes = plt.subplots(len(cutoffs), 1, figsize=(15, 12))

    # Use a channel with visible artifacts
    test_channel = "Cz"
    raw_segment = raw_data.get_data(picks=[test_channel])[0, start_sample:end_sample]

    for i, cutoff in enumerate(cutoffs):
        ax = axes[i]

        # Apply ASR with this cutoff
        asr_pipeline = (
                NotchFilterNode((50, 60)) >>
                BandPassFilterNode(0.5, 40.0) >>
                ASRNode(sfreq=fs, cutoff=cutoff, calibration_time=5.0, calibrate=True, name=f"ASR_{cutoff}")
        )

        cleaned = asr_pipeline(raw_data)
        cleaned_segment = cleaned.get_data(picks=[test_channel])[0, start_sample:end_sample]

        # Plot
        ax.plot(times_segment, raw_segment, 'r-', alpha=0.3, label='Raw', linewidth=1)
        ax.plot(times_segment, cleaned_segment, 'b-', alpha=0.7,
                label=f'ASR (cutoff={cutoff})', linewidth=1.5)

        ax.set_ylabel(f'Cutoff={cutoff}\n(µV)', rotation=0, ha='right', va='center')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        if i == 0:
            ax.set_title(f'Effect of ASR Cutoff Parameter - Channel {test_channel}')
        if i == len(cutoffs) - 1:
            ax.set_xlabel('Time (s)')
        else:
            ax.set_xticks([])

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("ASR Analysis Complete!")
    print("=" * 60)
    print("\nNote: Lower cutoff values (e.g., 3-5) are more aggressive,")
    print("while higher values (e.g., 10-20) are more conservative.")
    print("Recommended starting point: cutoff=5.0")
    print("=" * 60)