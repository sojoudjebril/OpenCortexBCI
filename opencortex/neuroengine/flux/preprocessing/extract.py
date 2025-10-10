"""
Nodes for extracting features/labels and scaling EEG data
"""
import numpy as np
import logging
from typing import Optional, Dict, Tuple, Literal, Union, List
from mne import Epochs
from opencortex.neuroengine.flux.base.node import Node



class ExtractNode(Node):
    """
    A node that extracts X (features) and y (labels) from epoched data.
    Optionally applies label encoding.
    """

    def __init__(self,
                 label_encoder,
                 apply_label_encoding: bool = True,
                 label_mapping: Optional[Dict[int, int]] = None,
                 picks: Union[str, List[str]] = 'eeg',
                 name: str = None):
        """
        Initialize the ExtractXyNode.

        Args:
            label_encoder: An instance of sklearn's LabelEncoder.
            apply_label_encoding: If True, apply sklearn LabelEncoder to labels.
            label_mapping: Optional dictionary to map original labels to new values
                          before encoding. Example: {1: 0, 3: 1} maps targets to 0, non-targets to 1.
            picks: Channels to include ('eeg', 'all', or list of channel names).
            name: Optional name for this node.
        """
        super().__init__(name or "ExtractXy")
        self.apply_label_encoding = apply_label_encoding
        self.label_mapping = label_mapping
        self.label_encoder = label_encoder if apply_label_encoding else None
        self.is_fitted = False
        self.picks = picks

    def __call__(
            self,
            epochs_data: Epochs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract and potentially encode X and y.

        Args:
            epochs_data: mne.Epochs object containing epoched data and events.

        Returns:
            Tuple of (X, y)
            - X: shape (n_epochs, n_channels, n_times)
            - y: shape (n_epochs,) - potentially encoded
        """

        X = epochs_data.get_data(picks=self.picks)
        y = epochs_data.events[:, -1]
        logging.info(f"Extracting X and y: X shape={X.shape}, y shape={y.shape}")

        # Apply label mapping if specified
        if self.label_mapping is not None:
            y_mapped = y.copy()
            for old_label, new_label in self.label_mapping.items():
                y_mapped[y == old_label] = new_label
            y = y_mapped
            logging.debug(f"Applied label mapping: {self.label_mapping}")
            logging.debug(f"Mapped labels: {np.unique(y)}")

        # Apply label encoding if specified
        if self.apply_label_encoding:
            if not self.is_fitted:
                y = self.label_encoder.fit_transform(y)
                self.is_fitted = True
                logging.info(
                    f"Fitted LabelEncoder. Classes: {self.label_encoder.classes_} "
                    f"-> {np.unique(y)}"
                )
            else:
                y = self.label_encoder.transform(y)
                logging.debug(f"Transformed labels using fitted encoder")

        logging.info(f"Final: X shape={X.shape}, y shape={y.shape}, unique labels={np.unique(y)}")

        return X, y

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """
        Inverse transform encoded labels back to original labels.

        Args:
            y: Encoded labels

        Returns:
            Original labels
        """
        if self.label_encoder is None or not self.is_fitted:
            return y
        return self.label_encoder.inverse_transform(y)


    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "apply_label_encoding": self.apply_label_encoding,
            "label_mapping": self.label_mapping,
            "is_fitted": self.is_fitted,
            "picks": self.picks,
            "label_encoder": self.label_encoder.__class__.__name__ if self.label_encoder else None
        })
        return config

    def __str__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        return (f"{self.__class__.__name__}"
                f"(encoding={self.apply_label_encoding}, status={status}, name={self.name})")
