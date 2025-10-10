"""
Author: Michele Romani
Email: michele.romani.zaltieri@gmail.com
Copyright 2025 Michele Romani
"""

import numpy as np
import logging
import onnxruntime as ort
from typing import Optional, Callable, Tuple, Any, Dict, Union
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
            name: str = None
    ):
        """
        Args:
            model_path: Path to ONNX model (.onnx)
        """
        super().__init__(name or "ONNX")
        self.model_path = Path(model_path)

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
        # Handle DataLoader
        if hasattr(data, '__iter__') and not isinstance(data, np.ndarray):
            X = self._extract_data(data)
        else:
            X = data

        # Ensure float32
        X = X.astype(np.float32)

        # Run inference
        outputs = self.session.run(None, {self.input_name: X})
        predictions = outputs[0]

        # Convert to class labels if needed
        if predictions.ndim > 1:
            predictions = predictions.argmax(axis=1)

        return predictions

    def _extract_data(self, loader: Any) -> np.ndarray:
        """Extract data from dataloader."""
        import torch

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
