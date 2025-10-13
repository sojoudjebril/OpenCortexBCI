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

    def __init__(self, **branches: Node):
        super().__init__("Parallel")
        self.branches = branches
        
    @classmethod
    def hydra_inst(cls, branches: dict, name: str = None) -> "Parallel":
        """
        Factory method for Hydra to instantiate a Parallel node with named branches.

        Args:
            branches: A dictionary of {name: Node} pairs
            name: Optional name for the node

        Returns:
            Parallel instance
        """
        obj = cls(**branches, name=name or "Parallel")
        return obj      

    def __call__(self, data: Any) -> Dict[str, Any]:
        return {name: branch(data) for name, branch in self.branches.items()}

    def get_branches(self, branch_name):
        for name, branch in self.branches.items():
            if branch_name == name:
                return branch
        raise ValueError(f"Node '{branch_name}' not found in pipeline")

    def get_config(self) -> dict:
        return {
            "_target_": f"{self.__class__.__module__}.{self.__class__.__qualname__}.hydra_inst",
            "name": self.name,
            "branches": {name: branch.get_config() for name, branch in self.branches.items()}
        }
