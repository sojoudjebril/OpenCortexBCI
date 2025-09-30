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