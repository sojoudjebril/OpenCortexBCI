import logging
import numpy as np
from opencortex.neuroengine.flux.base.node import Node
from typing import Optional, Tuple
from sklearn.base import clone, BaseEstimator

class ScalerNode(Node):
    """
    A generic node that applies any scaler to EEG data.
    Supports custom scalers or scikit-learn scalers.

    The scaler must implement:
    - fit(X) or fit(X, y) method
    - transform(X) method
    - Optionally: fit_transform(X) method
    """

    def __init__(
            self,
            scaler,
            per_channel: bool = True,
            name: str = None
    ):
        """
        Initialize the ScalerNode.

        Args:
            scaler: Scaler instance (e.g., StandardScaler(), RobustScaler(), or custom scaler).
                   Must have fit() and transform() methods.
            per_channel: If True, fit/transform each channel independently.
                        If False, flatten all data and scale globally.
            name: Optional name for this node.
        """
        super().__init__(name or f"Scaler_{scaler.__class__.__name__}")
        self.scaler = scaler
        self.per_channel = per_channel
        self.is_fitted = False
        self.channel_scalers = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'ScalerNode':
        """
        Fit the scaler on training data.

        Args:
            X: Training data of shape (n_epochs, n_channels, n_times)
            y: Optional labels (unused, for sklearn compatibility)

        Returns:
            self
        """
        logging.info(f"Fitting {self.scaler.__class__.__name__} on data with shape {X.shape}")

        if self.per_channel:
            # Fit a separate scaler for each channel
            n_channels = X.shape[1]
            self.channel_scalers = []

            for ch_idx in range(n_channels):
                # Get all data for this channel: (n_epochs, n_times)
                # Reshape to (n_epochs * n_times, 1) for sklearn compatibility
                ch_data = X[:, ch_idx, :].reshape(-1, 1)

                # Clone the scaler for this channel
                try:
                    ch_scaler = clone(self.scaler)
                except:
                    # If clone fails (custom scaler), create new instance
                    ch_scaler = self.scaler.__class__(**self.scaler.get_params())

                ch_scaler.fit(ch_data)
                self.channel_scalers.append(ch_scaler)

            logging.debug(f"Fitted {len(self.channel_scalers)} channel-wise scalers")

        else:
            # Fit single scaler on all data
            # Reshape to (n_epochs * n_channels * n_times, 1)
            X_flat = X.reshape(-1, 1)
            self.scaler.fit(X_flat)
            logging.debug(f"Fitted global scaler on {X_flat.shape[0]} samples")

        self.is_fitted = True
        logging.info(f"Scaler fitted successfully")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted scaler.

        Args:
            X: Data of shape (n_epochs, n_channels, n_times)

        Returns:
            Scaled data of same shape
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before transform. Call fit() first.")

        X_scaled = np.zeros_like(X)

        if self.per_channel:
            # Transform each channel independently
            for ch_idx, ch_scaler in enumerate(self.channel_scalers):
                ch_data = X[:, ch_idx, :].reshape(-1, 1)
                ch_scaled = ch_scaler.transform(ch_data)
                X_scaled[:, ch_idx, :] = ch_scaled.reshape(X.shape[0], X.shape[2])
        else:
            # Transform using global scaler
            n_epochs, n_channels, n_times = X.shape
            X_flat = X.reshape(-1, 1)
            X_scaled_flat = self.scaler.transform(X_flat)
            X_scaled = X_scaled_flat.reshape(n_epochs, n_channels, n_times)

        logging.debug(f"Transformed data: shape={X_scaled.shape}, range=[{X_scaled.min():.3f}, {X_scaled.max():.3f}]")

        return X_scaled

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            X: Data of shape (n_epochs, n_channels, n_times)
            y: Optional labels (unused, for sklearn compatibility)

        Returns:
            Scaled data of same shape
        """
        return self.fit(X, y).transform(X)

    def __call__(
            self,
            data: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply scaling to X, pass through y unchanged.

        Args:
            data: Tuple of (X, y)

        Returns:
            Tuple of (X_scaled, y)
        """
        X, y = data

        # Fit on first call, transform on all calls
        if not self.is_fitted:
            X_scaled = self.fit_transform(X, y)
        else:
            X_scaled = self.transform(X)

        return X_scaled, y

    def __str__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        mode = "per-channel" if self.per_channel else "global"
        return (f"{self.__class__.__name__}"
                f"(scaler={self.scaler.__class__.__name__}, mode={mode}, "
                f"status={status}, name={self.name})")


class ChannelwiseStandardScaler:
    """
    Channel-wise standard scaler (mean=0, std=1).
    Numpy implementation compatible with ScalerNode.
    """

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def get_params(self, deep=True):
        return {
            'mean': self.mean,
            'std': self.std
        }

    def fit(self, X: np.ndarray, y=None):
        """
        X: shape (n_samples, 1) - flattened channel data
        """
        self.mean = X.mean()
        self.std = X.std()
        self.std = max(self.std, 1e-6)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        X: shape (n_samples, 1)
        """
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class ChannelwiseRobustScaler:
    """
    Channel-wise robust scaler (median, IQR).
    Numpy implementation compatible with ScalerNode.
    """

    def __init__(self, quantile_range=(0.25, 0.75), clamp_min=1e-6, median=None, iqr=None):
        self.quantile_range = quantile_range
        self.clamp_min = clamp_min
        self.median = median
        self.iqr = iqr

    def get_params(self, deep=True):
        return {
            'quantile_range': self.quantile_range,
            'clamp_min': self.clamp_min,
            'median': self.median,
            'iqr': self.iqr
        }

    def fit(self, X: np.ndarray, y=None):
        """
        X: shape (n_samples, 1) - flattened channel data
        """
        self.median = np.median(X)
        q_min, q_max = self.quantile_range
        qmin = np.quantile(X, q_min)
        qmax = np.quantile(X, q_max)
        self.iqr = np.clip(qmax - qmin, a_min=self.clamp_min, a_max=None)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        X: shape (n_samples, 1)
        """
        return (X - self.median) / self.iqr

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)