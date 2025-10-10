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
    """Configuration for a single pipeline in the PipelineGroup."""

    def __init__(
            self,
            pipeline: Node,
            name: str = None,
            callback: Optional[Callable[[str, Any], None]] = None
    ):
        """
        Args:
            pipeline: The Node/Sequential to execute
            callback: Function called with (pipeline_name, result) when complete
            name: Name identifier for this pipeline
        """
        self.pipeline = pipeline
        self.callback = callback
        self.name = name or f"pipeline_{id(pipeline)}"
        
    def __repr__(self):
        return f"PipelineConfig(name={self.name}, pipeline={self.pipeline}, config={self.get_config()})"
    
    
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
    

    def get_config(self) -> dict:
        """
        Returns a full dictionary representation of the pipeline,
        exportable to a YAML Hydra config.
        """
        # Handle Sequential/Parallel/etc.
        if hasattr(self.pipeline, "get_config"):
            structure = self.pipeline.get_config()
            return {
                "name": self.name,
                "nodes": [structure]  # wrap single top-level node (Sequential/Parallel) in a list
            }
        else:
            raise ValueError("Pipeline does not support config export.")
        

    def config_to_yaml(self, filepath: str):
        """
        Export pipeline config to a YAML file.
        """
        import yaml
        with open(filepath, 'w') as f:
            yaml.dump(self.get_config(), f)

