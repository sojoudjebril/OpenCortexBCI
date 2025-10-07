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