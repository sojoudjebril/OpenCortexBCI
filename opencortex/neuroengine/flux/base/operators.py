# operators.py
from node import Node
from sequential import Sequential
from parallel import Parallel

def _rshift(self: Node, other: Node) -> Sequential:
    """Chain nodes: node1 >> node2"""
    return Sequential(self, other)

def _or(self: Node, other: Node) -> Parallel:
    """Parallel nodes: node1 | node2"""
    return Parallel(left=self, right=other)

Node.__rshift__ = _rshift
Node.__or__ = _or