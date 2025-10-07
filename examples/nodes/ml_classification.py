from opencortex.neuroengine.flux.base.sequential import Sequential
from opencortex.neuroengine.flux.estimation.lightning import LightningNode
from opencortex.neuroengine.flux.estimation.scikit import ScikitNode
from opencortex.neuroengine.flux.evaluation.metrics import MetricNode
from opencortex.neuroengine.flux.preprocessing.dataset import DatasetNode
from opencortex.neuroengine.flux.preprocessing.epochs import EpochingNode
from opencortex.neuroengine.flux.preprocessing.extract import ExtractNode

if __name__ == "__main__":
    import opencortex.neuroengine.flux.base.operators
    import logging
    from opencortex.neuroengine.flux.preprocessing.bandpass import BandPassFilterNode
    from opencortex.neuroengine.flux.preprocessing.notch import NotchFilterNode
    from opencortex.neuroengine.flux.preprocessing.events import (
        ExtractEventsNode, FilterEventsNode, RelabelEventsNode
    )
    from opencortex.neuroengine.flux.preprocessing.scaler import ScalerNode
    from opencortex.utils.loader import load_data, convert_to_mne
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    logging.basicConfig(level=logging.INFO)

    # ========================================================================
    # Load Data
    # ========================================================================

    fs = 250
    chs = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]

    print("=" * 60)
    print("Loading EEG data...")
    print("=" * 60)

    eeg, trigger, _ = load_data(
        "../../data/aep/auditory_erp_eyes_open_S1.csv",
        fs=fs,
        skiprows=5,
        delimiter=','
    )

    raw_data = convert_to_mne(eeg, trigger, fs=fs, chs=chs, recompute=False)

    # ========================================================================
    # Example 1: Scikit-learn with Metrics
    # ========================================================================

    print("\n" + "=" * 60)
    print("Example 1: Scikit-learn Pipeline with Metrics")
    print("=" * 60)

    # Preprocessing pipeline
    preprocessing = Sequential(
        NotchFilterNode((50, 60)),
        BandPassFilterNode(0.1, 30.0),
        ExtractEventsNode(stim_channel='STI', auto_label=True),
        FilterEventsNode(max_event_id=90),
        RelabelEventsNode(target_class=1, nontarget_label=3),
        EpochingNode(tmin=-0.2, tmax=0.8, baseline=(-0.1, 0.0), event_id={'T': 1, 'NT': 3}),
        ExtractNode(None, apply_label_encoding=True, label_mapping={1: 0, 3: 1}),
        ScalerNode(scaler=StandardScaler(), per_channel=True),
        name="Preprocessing"
    )

    X, y = preprocessing(raw_data)

    # Create dataset and loaders
    dataset_node = DatasetNode(
        split_size=0.2,
        batch_size=8,
        shuffle=True,
        num_workers=4
    )
    train_loader, val_loader = dataset_node((X, y))

    # Train sklearn model
    sklearn_node = ScikitNode(
        model=RandomForestClassifier(n_estimators=100, random_state=42),
        fit_params={}
    )
    model = sklearn_node((train_loader, val_loader))

    # Make predictions
    y_pred = sklearn_node.predict(val_loader)

    # Extract ground truth
    import torch

    y_true = torch.cat([batch[1] for batch in val_loader], dim=0).numpy()

    # Compute metrics
    metric_node = MetricNode(
        scorers={
            'accuracy': accuracy_score,
            'f1': f1_score,
            'precision': precision_score,
            'recall': recall_score
        }
    )
    metrics = metric_node((y_true, y_pred))

    print("\nMetrics:")
    for name, score in metrics.items():
        print(f"  {name}: {score:.4f}")

    # ========================================================================
    # Example 2: PyTorch Lightning
    # ========================================================================

    print("\n" + "=" * 60)
    print("Example 2: PyTorch Lightning Pipeline")
    print("=" * 60)

    try:
        import torch
        import torch.nn as nn
        import pytorch_lightning as pl


        class SimpleEEGNet(pl.LightningModule):
            def __init__(self, n_channels=8, n_times=250, n_classes=2, lr=0.001):
                super().__init__()
                self.save_hyperparameters()
                self.lr = lr

                self.conv1 = nn.Conv1d(n_channels, 32, 5, padding=2)
                self.pool = nn.MaxPool1d(2)
                self.conv2 = nn.Conv1d(32, 64, 5, padding=2)
                self.fc = nn.Linear(64 * (n_times // 4), n_classes)
                self.criterion = nn.CrossEntropyLoss()

            def forward(self, x):
                x = self.pool(torch.relu(self.conv1(x)))
                x = self.pool(torch.relu(self.conv2(x)))
                x = x.view(x.size(0), -1)
                return self.fc(x)

            def training_step(self, batch, batch_idx):
                x, y = batch
                logits = self(x)
                loss = self.criterion(logits, y)
                self.log('train_loss', loss)
                return loss

            def validation_step(self, batch, batch_idx):
                x, y = batch
                logits = self(x)
                loss = self.criterion(logits, y)
                acc = (logits.argmax(1) == y).float().mean()
                self.log('val_loss', loss)
                self.log('val_acc', acc)

            def predict_step(self, batch, batch_idx):
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                return self(x)

            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=self.lr)


        # Create fresh loaders for Lightning
        train_loader2, val_loader2 = dataset_node((X, y))

        # Train Lightning model
        lightning_node = LightningNode(
            model=SimpleEEGNet(n_channels=len(chs), n_times=250),
            trainer_config={
                'max_epochs': 5,
                'accelerator': 'cpu',
                'enable_progress_bar': True,
                'enable_model_summary': False,
                'log_every_n_steps': 1,
            }
        )

        trained_model = lightning_node((train_loader2, val_loader2))

        # Predictions
        y_pred_lightning = lightning_node.predict(val_loader2)

        # Metrics
        metrics_lightning = metric_node((y_true, y_pred_lightning))

        print("\nLightning Metrics:")
        for name, score in metrics_lightning.items():
            print(f"  {name}: {score:.4f}")

    except ImportError:
        print("\nSkipping Lightning example (pytorch-lightning not installed)")

    # ========================================================================
    # Example 3: Custom Metrics
    # ========================================================================

    print("\n" + "=" * 60)
    print("Example 3: Custom Metrics")
    print("=" * 60)


    def balanced_accuracy(y_true, y_pred):
        """Custom balanced accuracy scorer."""
        from sklearn.metrics import balanced_accuracy_score
        return balanced_accuracy_score(y_true, y_pred)


    def confusion_matrix_metric(y_true, y_pred):
        """Return flattened confusion matrix as single score (for demo)."""
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        return float(cm[0, 0] + cm[1, 1])  # True positives


    custom_metric_node = MetricNode(
        scorers={'accuracy': accuracy_score},
        custom_scorers={
            'balanced_acc': balanced_accuracy,
            'correct_predictions': confusion_matrix_metric
        }
    )

    custom_metrics = custom_metric_node((y_true, y_pred))

    print("\nCustom Metrics:")
    for name, score in custom_metrics.items():
        print(f"  {name}: {score:.4f}")

    print("\n" + "=" * 60)
    print("Examples Complete!")