"""
Author: Michele Romani
Email: michele.romani.zaltieri@gmail.com
Copyright 2025 Michele Romani
"""

import numpy as np
import logging
import pytorch_lightning as pl
import torch
from typing import Optional, Callable, Tuple, Any, Dict, Union, Literal
from opencortex.neuroengine.flux.base.node import Node
from pathlib import Path


class LightningNode(Node):
    """
    Trains/evaluates PyTorch Lightning models with train/inference modes.
    """

    def __init__(
            self,
            model: Any,
            mode: Literal['train', 'inference'] = 'train',
            trainer_config: Optional[Dict] = None,
            train_func: Optional[Callable] = None,
            eval_func: Optional[Callable] = None,
            checkpoint_path: Optional[Union[str, Path]] = None,
            return_true_labels: bool = False,
            name: str = None,
            log: Optional[logging.Logger] = None
    ):
        """
        Args:
            model: PyTorch Lightning module
            mode: 'train' (trains model) or 'inference' (uses existing model)
            trainer_config: Config dict for pl.Trainer
            train_func: Custom training function (model, train_loader, val_loader) -> model
            eval_func: Custom eval function (model, val_loader) -> metrics
            checkpoint_path: Path to load checkpoint from
            return_true_labels: If True, predict() returns (predictions, true_labels)
            name: Optional name for this node
            log: Optional logger
        """
        super().__init__(name or "Lightning")
        self.model = model
        self.mode = mode
        self.trainer_config = trainer_config or {}
        self.train_func = train_func
        self.eval_func = eval_func
        self.checkpoint_path = checkpoint_path
        self.trainer = None
        self.is_trained = False
        self.return_true_labels = return_true_labels
        self.log = log or logging.getLogger(__name__)

        # Load checkpoint if provided
        if checkpoint_path:
            self.model = model.__class__.load_from_checkpoint(checkpoint_path)
            self.is_trained = True
            self.log.debug(f"Loaded checkpoint: {checkpoint_path}")

    def __call__(
            self,
            data: Tuple[Any, Optional[Any]]
    ) -> Any:
        """
        Args:
            data: (train_loader, val_loader)

        Returns:
            - In 'train' mode: Trained model
            - In 'inference' mode: (train_loader, val_loader) passed through
        """

        #TODO why logger doesn't work? --> test new implementation
        if self.mode == 'train':
            train_loader, val_loader = data
            # Training mode
            if self.train_func:
                # Custom training
                self.model = self.train_func(self.model, train_loader, val_loader)
            else:
                # Default Lightning training
                self.trainer = pl.Trainer(**self.trainer_config)
                self.log.info(self.trainer_config)
                self.trainer.fit(
                    self.model,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader
                )

            self.is_trained = True
            self.log.debug("Training complete")

            # Evaluate if provided
            if self.eval_func and val_loader:
                metrics = self.eval_func(self.model, val_loader)
                self.log.debug(f"Validation metrics: {metrics}")
            return self.model

        elif self.mode == 'inference':
            test_loader = data
            # Inference mode - just pass through
            if not self.is_trained:
                logging.warning("Model not trained yet, predictions may be random")

            return self.predict(test_loader)



    def predict(self, dataloader: Any) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """Make predictions."""
        if self.trainer is None:
            self.trainer = pl.Trainer(**self.trainer_config)

        true_labels = dataloader.dataset.tensors[1].numpy()
        predictions = self.trainer.predict(self.model, dataloaders=dataloader)
        predictions = torch.cat(predictions, dim=0)

        if predictions.ndim > 1:
            predictions = predictions.argmax(dim=1)

        if self.return_true_labels:
            return predictions.cpu().numpy(), true_labels
        else:
            return predictions.cpu().numpy()

    def save_checkpoint(self, path: Union[str, Path]):
        """Save model checkpoint."""
        if not self.is_trained:
            logging.warning("Saving untrained model checkpoint")

        if self.trainer is None:
            self.trainer = pl.Trainer(**self.trainer_config)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.trainer.save_checkpoint(str(path))
        logging.info(f"Saved checkpoint: {path}")

    def get_state(self) -> Dict[str, Any]:
        """Get node state for serialization."""
        return {
            'checkpoint_path': str(self.checkpoint_path) if self.checkpoint_path else None,
            'mode': self.mode,
            'is_trained': self.is_trained,
            'trainer_config': self.trainer_config
        }

    def set_state(self, state: Dict[str, Any]):
        """Set node state from serialization."""
        if state.get('checkpoint_path'):
            self.checkpoint_path = Path(state['checkpoint_path'])
            self.model = self.model.__class__.load_from_checkpoint(self.checkpoint_path)
            self.is_trained = True
            logging.info(f"Restored model from checkpoint: {self.checkpoint_path}")

        self.mode = state.get('mode', 'train')
        self.is_trained = state.get('is_trained', False)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "_target_": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "mode": self.mode,
            "trainer_config": self.trainer_config,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "name": self.name
        })
        return config

    def __str__(self):
        status = "trained" if self.is_trained else "untrained"
        return f"{self.__class__.__name__}(model={self.model.__class__.__name__}, mode={self.mode}, status={status})"