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

    def __call__(self, data: Any) -> Any:
        for step in self.steps:
            data = step(data)
        return data

    def __repr__(self) -> str:
        return "Sequential({})".format(self.steps)

    def get_node(self, node_name: str) -> Any:
        for node in self.steps:
            if node.name == node_name:
                return node
        raise ValueError(f"Node '{node_name}' not found in pipeline")