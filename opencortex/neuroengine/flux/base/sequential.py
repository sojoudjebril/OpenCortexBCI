"""
Author: Michele Romani
Email: michele.romani.zaltieri@gmail.com
Copyright 2025 Michele Romani
"""
from opencortex.neuroengine.flux.base.node import Node
from typing import Any

class Sequential(Node):
    """
    A node that composes other nodes sequentially.
    """

    def __init__(self, *steps: Node, name: str = None):
        super().__init__(name or "Sequential")
        self.steps = steps
        
    @classmethod
    def hydra_inst(cls, steps: list, name: str = None) -> 'Sequential':
        """
        Factory method for Hydra to instantiate a Sequential node with steps.

        Args:
            steps: A list of Node instances
            name: Optional name for the node

        Returns:
            Sequential instance
        """
        return cls(*steps, name=name or "Sequential")


    def __call__(self, data: Any) -> Any:
        try:
            for step in self.steps:
                data = step(data)
            return data
        except Exception as e:
            raise RuntimeError(f"Error in Sequential node '{self.name}': {e}") from e

    def __repr__(self) -> str:
        return "Sequential({})".format(self.steps)
    
    def get_config(self) -> dict:
        return {
            "_target_": f"{self.__class__.__module__}.{self.__class__.__qualname__}.hydra_inst",
            "name": self.name,
            "steps": [step.get_config() for step in self.steps]
        }

    def get_node(self, node_name: str) -> Any:
        for node in self.steps:
            if node.name == node_name:
                return node
        raise ValueError(f"Node '{node_name}' not found in pipeline")