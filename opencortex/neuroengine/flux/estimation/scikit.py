"""
Author: Michele Romani
Email: michele.romani.zaltieri@gmail.com
Copyright 2025 Michele Romani
"""

import numpy as np
import logging
import torch
from typing import Optional, Callable, Tuple, Any, Dict, Union
from opencortex.neuroengine.flux.base.node import Node


class ScikitNode(Node):
    """
    Trains/evaluates scikit-learn compatible models.
    """

    def __init__(
            self,
            model: Any,
            fit_params: Optional[Dict] = None,
            eval_func: Optional[Callable] = None,
            name: str = None
    ):
        """
        Args:
            model: sklearn-compatible model (fit/predict methods)
            fit_params: Parameters for model.fit()
            eval_func: Custom eval function (model, X_val, y_val) -> metrics
        """
        super().__init__(name or "Scikit")
        self.model = model
        self.fit_params = fit_params or {}
        self.eval_func = eval_func

    def __call__(
            self,
            data: Tuple[Any, Optional[Any]]
    ) -> Any:
        """
        Args:
            data: (train_loader, val_loader)

        Returns:
            Trained model
        """
        train_loader, val_loader = data

        # Extract data from loaders
        X_train, y_train = self._extract_data(train_loader)

        # Fit model
        self.model.fit(X_train, y_train, **self.fit_params)
        logging.info("Training complete")

        # Evaluate if provided
        if self.eval_func and val_loader:
            X_val, y_val = self._extract_data(val_loader)
            metrics = self.eval_func(self.model, X_val, y_val)
            logging.info(f"Validation metrics: {metrics}")

        return self.model

    def _extract_data(self, loader: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Extract all data from dataloader."""

        X_list, y_list = [], []
        for batch in loader:
            X_batch, y_batch = batch
            X_list.append(X_batch)
            y_list.append(y_batch)

        X = torch.cat(X_list, dim=0).numpy()
        y = torch.cat(y_list, dim=0).numpy()

        # Flatten if 3D (for sklearn)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

        return X, y

    def predict(self, loader: Any) -> np.ndarray:
        """Make predictions."""
        X, _ = self._extract_data(loader)
        return self.model.predict(X)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "_target_": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "model": str(self.model),
            "fit_params": self.fit_params,
            "eval_func": self.eval_func.__name__ if self.eval_func else None,
            "name": self.name
        })
        return config

    def __str__(self):
        return f"{self.__class__.__name__}(model={self.model.__class__.__name__})"
