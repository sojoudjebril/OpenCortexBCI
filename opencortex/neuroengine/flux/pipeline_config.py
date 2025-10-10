"""
Author: Michele Romani
Email: michele.romani.zaltieri@gmail.com
Copyright 2025 Michele Romani
"""
from typing import Dict, Any, Optional, Callable
from opencortex.neuroengine.flux.base.node import Node


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
    
        
    def _from_config_to_yaml(self) -> str:
        """Export the pipeline configuration to a YAML string."""
        import yaml
        config_dict = {
            'name': self.name,
            'pipeline': self.pipeline.name,
            'config': self._create_config()
        }
        return yaml.dump(config_dict)
    
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'PipelineConfig':
        """Create a PipelineConfig from a configuration dictionary."""
        pass