"""
Author: Michele Romani
Email: michele.romani.zaltieri@gmail.com
Copyright 2025 Michele Romani
"""


from abc import ABC, abstractmethod
from typing import Any

class Node(ABC):
    """
    A base class for a processing unit in the BCI data flow.
    Each Node takes input(s) and produces output(s).
    """

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """
        Executes the node's computation.
        """
        pass

    def __str__(self):
        return f"{self.__class__.__name__}(name={self.name})"

    def __repr__(self):
        return self.__str__()


