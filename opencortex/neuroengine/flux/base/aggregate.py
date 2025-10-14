"""
Author: Michele Romani
Email: michele.romani.zaltieri@gmail.com
Copyright 2025 Michele Romani
"""
import numpy as np
import logging
from typing import Dict, Any, Optional, Callable, List, Union
from opencortex.neuroengine.flux.base.node import Node


class Aggregate(Node):
    """
    A node that aggregates outputs from parallel branches.
    Can flatten nested dictionaries, apply transformations, and combine results.
    """

    def __init__(
            self,
            mode: str = 'flatten',
            aggregation_func: Optional[Callable] = None,
            flatten_keys: bool = True,
            prefix_separator: str = '_',
            filter_keys: Optional[List[str]] = None,
            exclude_keys: Optional[List[str]] = None,
            name: str = None
    ):
        """
        Args:
            mode: Aggregation mode:
                - 'flatten': Flatten nested dicts into single dict
                - 'list': Keep as list of outputs
                - 'dict': Keep as dict (pass-through)
                - 'mean': Average numeric values across branches
                - 'vote': Majority voting for classification
                - 'custom': Use custom aggregation_func
            aggregation_func: Custom function (dict) -> Any
            flatten_keys: If True, flatten nested dictionary keys
            prefix_separator: Separator for flattened keys (e.g., 'branch_key')
            filter_keys: Only include these keys (None = all)
            exclude_keys: Exclude these keys (None = none)
            name: Optional name for this node
        """
        super().__init__(name or "Aggregate")

        self.mode = mode
        self.aggregation_func = aggregation_func
        self.flatten_keys = flatten_keys
        self.prefix_separator = prefix_separator
        self.filter_keys = filter_keys
        self.exclude_keys = exclude_keys or []

        if mode == 'custom' and aggregation_func is None:
            raise ValueError("aggregation_func must be provided when mode='custom'")

    def __call__(self, data: Dict[str, Any]) -> Union[Dict[str, Any], List[Any], Any]:
        """
        Aggregate outputs from parallel branches.

        Args:
            data: Dictionary from Parallel node {branch_name: output}

        Returns:
            Aggregated result (type depends on mode)
        """
        if not isinstance(data, dict):
            logging.warning(f"AggregateNode expected dict input, got {type(data)}")
            return data

        # Filter keys if specified
        filtered_data = self._filter_data(data)

        if self.mode == 'flatten':
            return self._flatten(filtered_data)
        elif self.mode == 'list':
            return self._flatten_to_list(filtered_data)
        elif self.mode == 'dict':
            return filtered_data
        elif self.mode == 'mean':
            return self._mean_aggregation(filtered_data)
        elif self.mode == 'vote':
            return self._voting_aggregation(filtered_data)
        elif self.mode == 'custom':
            return self.aggregation_func(filtered_data)
        else:
            raise ValueError(f"Unknown aggregation mode: {self.mode}")

    def _filter_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter data based on include/exclude keys."""
        filtered = data.copy()

        # Include only specified keys
        if self.filter_keys:
            filtered = {k: v for k, v in filtered.items() if k in self.filter_keys}

        # Exclude specified keys
        if self.exclude_keys:
            filtered = {k: v for k, v in filtered.items() if k not in self.exclude_keys}

        return filtered

    def _flatten(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten nested dictionary structure.

        Example:
            {'arousal_model': {'prediction': 0.78}, 'valence_model': {'prediction': 0.71}}
            → {'arousal': 0.78, 'valence': 0.71}
        """
        result = {}

        for branch_name, branch_output in data.items():
            if isinstance(branch_output, dict):
                if self.flatten_keys:
                    # Flatten nested dict
                    for key, value in branch_output.items():
                        # Use branch name or nested key
                        if key in ['prediction', 'output', 'result', 'value']:
                            # Use branch name directly for common output keys
                            result[branch_name] = value
                        else:
                            # Combine branch name and key
                            result[f"{branch_name}{self.prefix_separator}{key}"] = value
                else:
                    # Keep nested structure but merge
                    result.update(branch_output)
            else:
                # Scalar output
                result[branch_name] = branch_output

        logging.debug(f"Flattened {len(data)} branches into {len(result)} keys")
        return result

    def _flatten_to_list(self, data: Dict[str, Any]) -> List[Any]:
        """
        Convert dictionary values to a list.

        Example:
            {'model1': 0.8, 'model2': 0.7} → [0.8, 0.7]
        """
        result = []
        for branch_name, branch_output in data.items():
            if isinstance(branch_output, (list, np.ndarray)):
                for item in branch_output:
                    result.append(item)
            else:
                result.append(branch_output)
        return result


    def _mean_aggregation(self, data: Dict[str, Any]) -> Union[Dict[str, float], float]:
        """
        Average numeric values across branches.

        Example:
            {'model1': 0.8, 'model2': 0.7, 'model3': 0.9} → 0.8
            {'model1': {'acc': 0.8}, 'model2': {'acc': 0.7}} → {'acc': 0.75}
        """
        # Check if all values are scalars
        values = list(data.values())

        if all(isinstance(v, (int, float)) for v in values):
            # Simple average of scalars
            mean_val = np.mean(values)
            logging.debug(f"Mean aggregation: {mean_val:.4f} from {len(values)} values")
            return float(mean_val)

        # Handle dictionary outputs
        if all(isinstance(v, dict) for v in values):
            # Get all unique keys
            all_keys = set()
            for v in values:
                all_keys.update(v.keys())

            # Average each key
            result = {}
            for key in all_keys:
                key_values = [v[key] for v in values if key in v and isinstance(v[key], (int, float))]
                if key_values:
                    result[key] = float(np.mean(key_values))

            logging.debug(f"Mean aggregation: averaged {len(result)} keys")
            return result

        logging.warning("Cannot compute mean for mixed types")
        return data

    def _voting_aggregation(self, data: Dict[str, Any]) -> Any:
        """
        Majority voting for classification results.

        Example:
            {'clf1': 0, 'clf2': 1, 'clf3': 0} → 0 (majority)
            {'clf1': 'happy', 'clf2': 'happy', 'clf3': 'sad'} → 'happy'
        """
        values = list(data.values())

        # Extract predictions from dicts if needed
        predictions = []
        for v in values:
            if isinstance(v, dict) and 'prediction' in v:
                predictions.append(v['prediction'])
            elif isinstance(v, dict) and 'class' in v:
                predictions.append(v['class'])
            else:
                predictions.append(v)

        # Count votes
        from collections import Counter
        vote_counts = Counter(predictions)
        winner = vote_counts.most_common(1)[0][0]

        logging.debug(
            f"Voting result: {winner} with {vote_counts[winner]}/{len(predictions)} votes"
        )

        return winner

    def get_config(self) -> dict:
        config = {
            "_target_": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "mode": self.mode,
            "flatten_keys": self.flatten_keys,
            "prefix_separator": self.prefix_separator,
            "name": self.name
        }

        if self.filter_keys:
            config["filter_keys"] = self.filter_keys
        if self.exclude_keys:
            config["exclude_keys"] = self.exclude_keys

        return config

    def __str__(self):
        return f"{self.__class__.__name__}(mode={self.mode}, name={self.name})"