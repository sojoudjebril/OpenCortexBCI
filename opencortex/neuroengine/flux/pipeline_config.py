"""
Author: Michele Romani
Email: michele.romani.zaltieri@gmail.com
Copyright 2025 Michele Romani
"""
from typing import Dict, Any, Optional, Callable
import opencortex.neuroengine.flux
from opencortex.neuroengine.flux.base.node import Node
from hydra.utils import instantiate


class PipelineConfig:
    """Configuration for a single pipeline in the ProcessorGroup."""

    def __init__(
            self,
            pipeline: Node,
            config: Dict[str, Any] = None,
            callback: Optional[Callable[[str, Any], None]] = None,
            name: str = None
    ):
        """
        Args:
            pipeline: The Node/Sequential to execute
            config: Configuration dictionary for this pipeline
            callback: Function called with (pipeline_name, result) when complete
            name: Name identifier for this pipeline
        """
        self.pipeline = pipeline
        self.config = config or {}
        self.callback = callback
        self.name = name or f"pipeline_{id(pipeline)}"
        
    def __repr__(self):
        return f"PipelineConfig(name={self.name}, pipeline={self.pipeline}, config={self.config})"
    
    
    def _create_config(self) -> Dict[str, Any]:
        """Parse all nodes in the pipeline to create a configuration dictionary. Useful for saving/loading and exporting."""
        # self.pipeline is a Node or Sequential
        config = {}
        for node in self.pipeline:
            if hasattr(node, 'get_config'):
                config[node.name] = node.get_config()
        return config
    
        
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'PipelineConfig':
        """
        Create a PipelineConfig from a dictionary loaded via Hydra config.
        This instantiates the pipeline as a list of Node instances.
        """
        name = config.get("name", "Pipeline")
        node_configs = config.get("nodes", [])

        # Instantiate each node
        nodes = [instantiate(node_cfg) for node_cfg in node_configs]

        # If your pipeline is a sequence of nodes, wrap them in a SequentialNode
        from opencortex.neuroengine.flux.base.sequential import Sequential
        pipeline = Sequential(*nodes, name=name)

        return cls(pipeline=pipeline, config=config, name=name)
