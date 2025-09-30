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

    def __call__(self, data: Any) -> Dict[str, Any]:
        return {name: branch(data) for name, branch in self.branches.items()}