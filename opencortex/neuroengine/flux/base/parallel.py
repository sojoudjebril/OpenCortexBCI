"""
Author: Michele Romani
Email: michele.romani.zaltieri@gmail.com
Copyright 2025 Michele Romani
"""
from opencortex.neuroengine.flux.base.node import Node
from typing import Any, Dict

class Parallel(Node):
    """
    A node that runs multiple branches in parallel on the same input.
    Returns a dictionary with each branch's output.
    """

    def __init__(self, name:str=None,**branches: Node):
        super().__init__("Parallel")
        self.branches = branches
        self.name = name

    def __call__(self, data: Any) -> Dict[str, Any]:
        return {name: branch(data) for name, branch in self.branches.items()}

    def get_branches(self, branch_name):
        for name, branch in self.branches.items():
            if branch_name == name:
                return branch
        raise ValueError(f"Node '{branch_name}' not found in pipeline")

    def get_config(self) -> dict:
        return {
            "_target_": f"{self.__class__.__module__}.{self.__class__.__qualname__}.from_branches",
            "name": self.name,
            "branches": {name: branch.get_config() for name, branch in self.branches.items()}
        }

    @classmethod
    def from_branches(cls, branches: dict, name: str = None) -> "Parallel":
        """
        Factory method for Hydra to instantiate a Parallel node with named branches.

        Args:
            branches: A dictionary of {name: Node} pairs
            name: Optional name for the node

        Returns:
            Parallel instance
        """
        obj = cls(**branches)
        obj.name = name or "Parallel"
        return obj
