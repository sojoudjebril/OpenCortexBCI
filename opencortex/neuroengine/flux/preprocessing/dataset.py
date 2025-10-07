"""
Author: Michele Romani
Email: michele.romani.zaltieri@gmail.com
Copyright 2025 Michele Romani
"""
import numpy as np
import logging
from typing import Optional, Callable, Tuple, Any
from opencortex.neuroengine.flux.base.node import Node


class DatasetNode(Node):
    """
    Creates train/val dataloaders from X, y data.
    """

    def __init__(
            self,
            split_size: float = 0.2,
            batch_size: int = 32,
            shuffle: bool = True,
            num_workers: int = 0,
            dataset_constructor: Optional[Callable] = None,
            name: str = None,
            **dataloader_kwargs
    ):
        """
        Args:
            split_size: Validation split ratio (0.0 to 1.0)
            batch_size: Batch size for dataloaders
            shuffle: Shuffle training data
            num_workers: Number of workers for data loading
            dataset_constructor: Custom TensorDataset constructor (optional)
            **dataloader_kwargs: Additional kwargs for DataLoader
        """
        super().__init__(name or "Dataset")
        self.split_size = split_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.dataset_constructor = dataset_constructor
        self.dataloader_kwargs = dataloader_kwargs

    def __call__(
            self,
            data: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[Any, Optional[Any]]:
        """
        Args:
            data: (X, y) tuple

        Returns:
            (train_loader, val_loader) - val_loader is None if split_size=0
        """
        import torch
        from torch.utils.data import TensorDataset, DataLoader, random_split

        X, y = data

        # Convert to tensors
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).long()

        # Create dataset
        if self.dataset_constructor:
            dataset = self.dataset_constructor(X_tensor, y_tensor)
        else:
            dataset = TensorDataset(X_tensor, y_tensor)

        # Split if needed
        if self.split_size > 0:
            n_val = int(len(dataset) * self.split_size)
            n_train = len(dataset) - n_val
            train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                **self.dataloader_kwargs
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                **self.dataloader_kwargs
            )

            logging.info(f"Created dataloaders: train={n_train}, val={n_val}")
            return train_loader, val_loader

        else:
            train_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                **self.dataloader_kwargs
            )

            logging.info(f"Created dataloader: train={len(dataset)}")
            return train_loader, None

    def __str__(self):
        return f"{self.__class__.__name__}(split={self.split_size}, batch={self.batch_size})"
