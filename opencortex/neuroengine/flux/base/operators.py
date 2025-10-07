"""
Author: Michele Romani
Email: michele.romani.zaltieri@gmail.com
Copyright 2025 Michele Romani
"""
from opencortex.neuroengine.flux.base.node import Node
from opencortex.neuroengine.flux.base.sequential import Sequential
from opencortex.neuroengine.flux.base.parallel import Parallel

def __add__(a: Node, b: Node) -> Sequential:
    """Chain nodes: node1 + node2"""
    return Sequential(a, b, name=f"( {a.name} + {b.name} )")

def _rshift(self: Node, other: Node) -> Sequential:
    """Chain nodes: node1 >> node2"""
    return Sequential(self, other, name=f"( {self.name} >> {other.name} )")

def _or(self: Node, other: Node) -> Parallel:
    """Parallel nodes: node1 | node2"""
    return Parallel(left=self, right=other, name=f"( {self.name} || {other.name} )")

def _print(self: Node) -> str:
    return f"Node(name={self.name})"

Node.__rshift__ = _rshift
Node.__or__ = _or
Node.__add__ = __add__
Node.__repr__ = _print