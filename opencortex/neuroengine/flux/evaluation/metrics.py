"""
Author: Michele Romani
Email: michele.romani.zaltieri@gmail.com
Copyright 2025 Michele Romani
"""

import numpy as np
import logging
from typing import Optional, Callable, Tuple, Dict
from opencortex.neuroengine.flux.base.node import Node


class MetricNode(Node):
    """
    Computes metrics on predictions.
    """

    def __init__(
            self,
            scorers: Optional[Dict[str, Callable]] = None,
            custom_scorers: Optional[Dict[str, Callable]] = None,
            name: str = None
    ):
        """
        Args:
            scorers: Dict of sklearn scorer names -> sklearn.metrics functions
                    Example: {'accuracy': accuracy_score, 'f1': f1_score}
            custom_scorers: Dict of custom scorer functions (y_true, y_pred) -> score
        """
        super().__init__(name or "Metrics")
        self.scorers = scorers or {}
        self.custom_scorers = custom_scorers or {}

    def __call__(
            self,
            data: Tuple[np.ndarray, np.ndarray]
    ) -> Dict[str, float]:
        """
        Args:
            data: (y_true, y_pred)

        Returns:
            Dict of metric_name -> score
        """
        y_true, y_pred = data

        results = {}

        # Compute sklearn scorers
        for name, scorer in self.scorers.items():
            try:
                score = scorer(y_true, y_pred)
                results[name] = float(score)
            except Exception as e:
                logging.warning(f"Failed to compute {name}: {e}")

        # Compute custom scorers
        for name, scorer in self.custom_scorers.items():
            try:
                score = scorer(y_true, y_pred)
                results[name] = float(score)
            except Exception as e:
                logging.warning(f"Failed to compute custom {name}: {e}")

        logging.info(f"Computed metrics: {results}")
        return results

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "_target_": f"{self.__class__.__module__}.{self.__class__.__qualname__}.from_scorers",
            "scorers": {name: scorer.__name__ for name, scorer in self.scorers.items()},
            "custom_scorers": {name: scorer.__name__ for name, scorer in self.custom_scorers.items()},
            "name": self.name,
        })
        return config

    def __str__(self):
        n_scorers = len(self.scorers) + len(self.custom_scorers)
        return f"{self.__class__.__name__}(n_scorers={n_scorers})"
