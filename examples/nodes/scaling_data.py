if __name__ == "__main__":
    import opencortex.neuroengine.flux.base.operators  # Enable >>
    from opencortex.neuroengine.flux.preprocessing.bandpass import BandPassFilterNode
    from opencortex.neuroengine.flux.preprocessing.notch import NotchFilterNode
    from opencortex.neuroengine.flux.preprocessing.events import (
        ExtractEventsNode, FilterEventsNode, RelabelEventsNode
    )
    from opencortex.neuroengine.flux.preprocessing.epochs import EpochingNode
    from opencortex.utils.loader import load_data, convert_to_mne
    from opencortex.neuroengine.flux.preprocessing.extract import ExtractNode
    from opencortex.neuroengine.flux.preprocessing.scaler import ScalerNode, ChannelwiseStandardScaler, ChannelwiseRobustScaler
    from opencortex.neuroengine.flux.base.sequential import Sequential
    import numpy as np
    import logging
    import matplotlib.pyplot as plt

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Configuration
    fs = 250
    chs = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]

    # Load data
    print("=" * 60)
    print("Loading EEG data for feature extraction and scaling...")
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
    # Build complete preprocessing pipeline with GENERIC ScalerNode
    # ========================================================================
    print("\n" + "=" * 60)
    print("Building pipelines with generic ScalerNode")
    print("=" * 60)

    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder

    # Example 1: Using sklearn StandardScaler
    pipeline_sklearn_standard = Sequential(
        NotchFilterNode((50, 60), name="PowerlineNotch"),
        BandPassFilterNode(0.1, 30.0, name="ERPBand"),
        ExtractEventsNode(stim_channel='STI', auto_label=True, name="EventExtractor"),
        FilterEventsNode(max_event_id=90, name="EventFilter"),
        RelabelEventsNode(target_class=1, nontarget_label=3, name="EventRelabeler"),
        EpochingNode(
            tmin=-0.2,
            tmax=0.8,
            baseline=(-0.1, 0.0),
            event_id={'T': 1, 'NT': 3},
            name="Epocher"
        ),
        ExtractNode(None, apply_label_encoding=True, label_mapping={1: 0, 3: 1}, name="ExtractXy"),
        ScalerNode(
            scaler=StandardScaler(),
            per_channel=True,
            name="SklearnStandardScaler"
        ),
        name="Pipeline_SklearnStandard"
    )

    # Example 2: Using sklearn RobustScaler
    pipeline_sklearn_robust = Sequential(
        NotchFilterNode((50, 60), name="PowerlineNotch"),
        BandPassFilterNode(0.1, 30.0, name="ERPBand"),
        ExtractEventsNode(stim_channel='STI', auto_label=True, name="EventExtractor"),
        FilterEventsNode(max_event_id=90, name="EventFilter"),
        RelabelEventsNode(target_class=1, nontarget_label=3, name="EventRelabeler"),
        EpochingNode(
            tmin=-0.2,
            tmax=0.8,
            baseline=(-0.1, 0.0),
            event_id={'T': 1, 'NT': 3},
            name="Epocher"
        ),
        ExtractNode(None, apply_label_encoding=True, label_mapping={1: 0, 3: 1}, name="ExtractXy"),
        ScalerNode(
            scaler=RobustScaler(),
            per_channel=True,
            name="SklearnRobustScaler"
        ),
        name="Pipeline_SklearnRobust"
    )

    # Example 3: Using custom ChannelwiseStandardScaler
    pipeline_custom_standard = Sequential(
        NotchFilterNode((50, 60), name="PowerlineNotch"),
        BandPassFilterNode(0.1, 30.0, name="ERPBand"),
        ExtractEventsNode(stim_channel='STI', auto_label=True, name="EventExtractor"),
        FilterEventsNode(max_event_id=90, name="EventFilter"),
        RelabelEventsNode(target_class=1, nontarget_label=3, name="EventRelabeler"),
        EpochingNode(
            tmin=-0.2,
            tmax=0.8,
            baseline=(-0.1, 0.0),
            event_id={'T': 1, 'NT': 3},
            name="Epocher"
        ),
        ExtractNode(None, apply_label_encoding=True, label_mapping={1: 0, 3: 1}, name="ExtractXy"),
        ScalerNode(
            scaler=ChannelwiseStandardScaler(),
            per_channel=True,
            name="CustomStandardScaler"
        ),
        name="Pipeline_CustomStandard"
    )

    # Example 3b: Using custom ChannelwiseRobustScaler
    pipeline_custom_robust = Sequential(
        NotchFilterNode((50, 60), name="PowerlineNotch"),
        BandPassFilterNode(0.1, 30.0, name="ERPBand"),
        ExtractEventsNode(stim_channel='STI', auto_label=True, name="EventExtractor"),
        FilterEventsNode(max_event_id=90, name="EventFilter"),
        RelabelEventsNode(target_class=1, nontarget_label=3, name="EventRelabeler"),
        EpochingNode(
            tmin=-0.2,
            tmax=0.8,
            baseline=(-0.1, 0.0),
            event_id={'T': 1, 'NT': 3},
            name="Epocher"
        ),
        ExtractNode(None, apply_label_encoding=True, label_mapping={1: 0, 3: 1}, name="ExtractXy"),
        ScalerNode(
            scaler=ChannelwiseRobustScaler(),
            per_channel=True,
            name="CustomRobustScaler"
        ),
        name="Pipeline_CustomRobust")

    # Example 4: Using sklearn MinMaxScaler
    pipeline_sklearn_minmax = Sequential(
        NotchFilterNode((50, 60), name="PowerlineNotch"),
        BandPassFilterNode(0.1, 30.0, name="ERPBand"),
        ExtractEventsNode(stim_channel='STI', auto_label=True, name="EventExtractor"),
        FilterEventsNode(max_event_id=90, name="EventFilter"),
        RelabelEventsNode(target_class=1, nontarget_label=3, name="EventRelabeler"),
        EpochingNode(
            tmin=-0.2,
            tmax=0.8,
            baseline=(-0.1, 0.0),
            event_id={'T': 1, 'NT': 3},
            name="Epocher"
        ),
        ExtractNode(label_encoder=LabelEncoder(), apply_label_encoding=True, label_mapping={1: 0, 3: 1}, name="ExtractXy"),
        ScalerNode(
            scaler=MinMaxScaler(),
            per_channel=True,
            name="SklearnMinMaxScaler"
        ),
        name="Pipeline_MinMaxScaler"
    )

    # Apply all pipelines
    print("\nApplying sklearn StandardScaler pipeline...")
    X_sklearn_std, y1 = pipeline_sklearn_standard(raw_data)

    print("\nApplying sklearn RobustScaler pipeline...")
    X_sklearn_rob, y2 = pipeline_sklearn_robust(raw_data)

    print("\nApplying custom StandardScaler pipeline...")
    X_custom_std, y3 = pipeline_custom_standard(raw_data)

    print("\nApplying custom RobustScaler pipeline...")
    X_custom_rob, y3b = pipeline_custom_robust(raw_data)

    print("\nApplying sklearn MinMaxScaler pipeline...")
    X_sklearn_minmax, y4 = pipeline_sklearn_minmax(raw_data)

    print(f"\n" + "=" * 60)
    print("Scaler Comparison Results:")
    print("=" * 60)

    scalers_data = [
        ("sklearn StandardScaler", X_sklearn_std),
        ("sklearn RobustScaler", X_sklearn_rob),
        ("Custom StandardScaler", X_custom_std),
        ("Custom RobustScaler", X_custom_rob),
        ("sklearn MinMaxScaler", X_sklearn_minmax),
    ]

    for scaler_name, X_scaled in scalers_data:
        print(f"\n{scaler_name}:")
        print(f"  Shape: {X_scaled.shape}")
        print(f"  Mean:  {X_scaled.mean():.6f}")
        print(f"  Std:   {X_scaled.std():.6f}")
        print(f"  Min:   {X_scaled.min():.6f}")
        print(f"  Max:   {X_scaled.max():.6f}")

    # ========================================================================
    # Visualize scaler comparison
    # ========================================================================
    print("\n" + "=" * 60)
    print("Creating scaler comparison visualization")
    print("=" * 60)

    # Plot comparison
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    axes = axes.flatten()

    # Choose a representative channel and epoch
    channel_idx = 6  # Cz
    epoch_idx = 0

    times = np.linspace(-0.2, 0.8, X_sklearn_std.shape[2])

    for idx, (scaler_name, X_scaled) in enumerate(scalers_data):
        ax = axes[idx]

        # Plot multiple epochs
        for ep_idx in range(min(5, X_scaled.shape[0])):
            ax.plot(times, X_scaled[ep_idx, channel_idx, :], alpha=0.5)

        ax.axvline(0, color='k', linestyle='--', alpha=0.5, label='Stimulus')
        ax.axhline(0, color='r', linestyle='-', alpha=0.3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Scaled Amplitude')
        ax.set_title(f'{scaler_name} - Channel {chs[channel_idx]}')
        ax.grid(True, alpha=0.3)
        #ax.set_ylim(-5, 5)  # Standardize y-axis for comparison
        ax.legend(loc='best')

    plt.tight_layout()
    plt.show()

    # Distribution comparison
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (scaler_name, X_scaled) in enumerate(scalers_data):
        ax = axes[idx]

        # Histogram of all values
        ax.hist(X_scaled.flatten(), bins=100, alpha=0.7, density=True)
        ax.set_xlabel('Scaled Value')
        ax.set_ylabel('Density')
        ax.set_title(f'{scaler_name} - Value Distribution')
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero')
        ax.legend()

    plt.tight_layout()
    plt.show()
