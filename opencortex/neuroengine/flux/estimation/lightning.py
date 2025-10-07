"""
Author: Michele Romani
Email: michele.romani.zaltieri@gmail.com
Copyright 2025 Michele Romani
"""

import numpy as np
import logging
import pytorch_lightning as pl
import torch
from typing import Optional, Callable, Tuple, Any, Dict, Union
from opencortex.neuroengine.flux.base.node import Node
from pathlib import Path


class LightningNode(Node):
    """
    Trains/evaluates PyTorch Lightning models.
    """

    def __init__(
            self,
            model: Any,
            trainer_config: Optional[Dict] = None,
            train_func: Optional[Callable] = None,
            eval_func: Optional[Callable] = None,
            checkpoint_path: Optional[Union[str, Path]] = None,
            name: str = None
    ):
        """
        Args:
            model: PyTorch Lightning module
            trainer_config: Config dict for pl.Trainer
            train_func: Custom training function (model, train_loader, val_loader) -> model
            eval_func: Custom eval function (model, val_loader) -> metrics
            checkpoint_path: Path to load checkpoint from
        """
        super().__init__(name or "Lightning")
        self.model = model
        self.trainer_config = trainer_config or {}
        self.train_func = train_func
        self.eval_func = eval_func
        self.checkpoint_path = checkpoint_path
        self.trainer = None

        # Load checkpoint if provided
        if checkpoint_path:
            self.model = model.__class__.load_from_checkpoint(checkpoint_path)
            logging.info(f"Loaded checkpoint: {checkpoint_path}")

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

        if self.train_func:
            # Custom training
            self.model = self.train_func(self.model, train_loader, val_loader)
        else:
            # Default Lightning training
            self.trainer = pl.Trainer(**self.trainer_config)
            self.trainer.fit(
                self.model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader
            )

        logging.info("Training complete")

        # Evaluate if provided
        if self.eval_func and val_loader:
            metrics = self.eval_func(self.model, val_loader)
            logging.info(f"Validation metrics: {metrics}")

        return self.model

    def predict(self, dataloader: Any) -> np.ndarray:
        """Make predictions."""

        if self.trainer is None:
            self.trainer = pl.Trainer(**self.trainer_config)

        predictions = self.trainer.predict(self.model, dataloaders=dataloader)
        predictions = torch.cat(predictions, dim=0)

        if predictions.ndim > 1:
            predictions = predictions.argmax(dim=1)

        return predictions.cpu().numpy()

    def __str__(self):
        return f"{self.__class__.__name__}(model={self.model.__class__.__name__})"