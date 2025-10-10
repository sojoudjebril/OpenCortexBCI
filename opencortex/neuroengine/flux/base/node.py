"""
Author: Michele Romani
Email: michele.romani.zaltieri@gmail.com
Copyright 2025 Michele Romani
"""


from abc import ABC, abstractmethod
from typing import Any
from mne.io import RawArray

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

    @abstractmethod
    def get_config(self) -> dict:
        """
        Export the node's configuration for Hydra instantiation.
        Override in subclasses to include custom parameters.
        """
        return {
            "_target_": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "name": self.name
        }

    def __str__(self):
        return f"{self.__class__.__name__}(name={self.name})"

    def __repr__(self):
        return self.__str__()
    
    
    
class MNENode(Node):
    """
    A Node that processes MNE RawArray objects. It takes only one RawArray as input and returns a RawArray.
    """

    @abstractmethod
    def __call__(self, data: RawArray) -> RawArray:
        """
        Processes an MNE RawArray and returns a modified RawArray.
        """
        return data

    @abstractmethod
    def get_config(self) -> dict:
        config = super().get_config()
        # Add any RawNode-specific configuration here
        return config




