"""
Author: Michele Romani
Email: michele.romani.zaltieri@gmail.com
Copyright 2025 Michele Romani
"""
import torch
import numpy as np
import logging
import threading
from typing import Optional, Callable, Tuple, Any, Dict, Union

from scipy.special import softmax

from opencortex.neuroengine.flux.base.node import Node
from pathlib import Path


# TODO check scikit2onnx
class ONNXNode(Node):
    """
    ONNX inference only.
    """

    def __init__(
            self,
            model_path: Union[str, Path],
            session = None,
            return_proba: bool = True,
            binary_threshold: float = 0.5,
            binary_pos_label: int = 1,
            add_batch_dim: bool = True,
            name: str = None
    ):
        """
        Args:
            model_path: Path to ONNX model (.onnx)
            session: Pre-initialized ONNX Runtime session (optional)
            return_proba: If True, return probabilities; else return class labels
            binary_threshold: Threshold for binary classification (if return_proba is False)
            binary_pos_label: Index of positive class for binary classification
            add_batch_dim: If True, add batch dimension to input data
            name: Optional name for this node
        """
        super().__init__(name or "ONNX")
        self.model_path = Path(model_path)
        self.return_proba = return_proba
        self.threshold = binary_threshold
        self.binary_pos_label = binary_pos_label
        self.add_batch_dim = add_batch_dim

        if session is not None:
            self.session = session
            logging.info("Using provided ONNX Runtime session.")
        else:
            if threading.current_thread() is not threading.main_thread():
                raise RuntimeError("onnxruntime must be imported from the main thread")
            import onnxruntime as ort
            self.session = ort.InferenceSession(str(self.model_path))

        self.input_name = self.session.get_inputs()[0].name
        logging.info(f"Loaded ONNX model: {self.model_path}")

    def __call__(self, data: Any) -> np.ndarray:
        """
        Args:
            data: DataLoader or numpy array

        Returns:
            Predictions
        """
        try:
            # Handle DataLoader
            if hasattr(data, '__iter__') and not isinstance(data, np.ndarray):
                X = self._extract_data(data)
            else:
                X = data

            # Ensure float32
            X = X.astype(np.float32)

            if self.add_batch_dim:
                X = X.reshape(1, *X.shape)

            # Run inference
            try:
                outputs = self.session.run(None, {self.input_name: X})
            except Exception as e:
                logging.error(f"{self.name}: ONNX inference failed - {e} with input shape {X.shape} and {self.add_batch_dim} batch dim parameter activated. If your model does not expect a batch dimension, please set add_batch_dim to False.")
                raise e
            predictions = outputs[0]

            # Convert to class labels if needed
            if predictions.ndim > 1 and not self.return_proba:
                return np.argmax(predictions, axis=1)
            elif predictions.ndim == 1 and not self.return_proba:
                return (predictions > self.threshold).astype(int)
            else:
                predicted = softmax(predictions, axis=1)

                return np.squeeze(predicted)[self.binary_pos_label]
        except Exception as e:
            logging.error(f"{self.name}: Error during ONNX inference - {e}")
            raise e

    def _extract_data(self, loader: Any) -> np.ndarray:
        """Extract data from dataloader."""

        X_list = []
        for batch in loader:
            X_batch = batch[0] if isinstance(batch, (list, tuple)) else batch
            X_list.append(X_batch)

        X = torch.cat(X_list, dim=0).numpy()
        return X

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "_target_": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "model_path": str(self.model_path),
            "name": self.name
        })
        return config

    def __str__(self):
        return f"{self.__class__.__name__}(path={self.model_path.name})"
