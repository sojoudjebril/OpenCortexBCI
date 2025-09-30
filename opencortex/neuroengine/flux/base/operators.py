# operators.py
from opencortex.neuroengine.flux.base.node import Node
from opencortex.neuroengine.flux.base.sequential import Sequential
from opencortex.neuroengine.flux.base.parallel import Parallel

def _rshift(self: Node, other: Node) -> Sequential:
    """Chain nodes: node1 >> node2"""
    return Sequential(self, other)

def _or(self: Node, other: Node) -> Parallel:
    """Parallel nodes: node1 | node2"""
    return Parallel(left=self, right=other)

Node.__rshift__ = _rshift
Node.__or__ = _or