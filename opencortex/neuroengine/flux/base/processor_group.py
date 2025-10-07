"""
Author: Michele Romani
Email: michele.romani.zaltieri@gmail.com
Copyright 2025 Michele Romani
"""
import threading
from typing import Any, Dict, Callable, Optional, List
from concurrent.futures import ThreadPoolExecutor, Future
from opencortex.neuroengine.flux.base.node import Node
from opencortex.neuroengine.flux.base.pipe_config import PipelineConfig


class ProcessorGroup(Node):
    """
    A node that runs multiple pipelines in parallel threads.
    Each pipeline receives the same input data but different configurations.
    Results can be propagated via callbacks.
    """

    def __init__(
            self,
            pipelines: List[PipelineConfig],
            name: str = None,
            max_workers: Optional[int] = None,
            wait_for_all: bool = False
    ):
        """
        Args:
            pipelines: List of PipelineConfig objects
            name: Name for this processor group
            max_workers: Maximum number of threads (None = number of pipelines)
            wait_for_all: If True, waits for all pipelines to complete before returning
        """
        super().__init__(name or "ProcessorGroup")
        self.pipelines = pipelines
        self.max_workers = max_workers or len(pipelines)
        self.wait_for_all = wait_for_all
        self._results: Dict[str, Any] = {}
        self._lock = threading.Lock()
        # For safe pipeline addition/removal
        self._pipeline_lock = threading.Lock()
        
    def __iter__(self):
        return iter(self.pipelines)

    def _execute_pipeline(
            self,
            pipeline_config: PipelineConfig,
            data: Any
    ) -> tuple[str, Any]:
        """
        Execute a single pipeline with its configuration.

        Args:
            pipeline_config: The pipeline configuration
            data: Input data

        Returns:
            Tuple of (pipeline_name, result)
        """
        try:
            # You could inject config into data or pass it to pipeline
            # For now, we just execute the pipeline with the data
            result = pipeline_config.pipeline(data)

            # Call the callback if provided
            if pipeline_config.callback:
                pipeline_config.callback(pipeline_config.name, result)

            return pipeline_config.name, result

        except Exception as e:
            error_result = {"error": str(e), "pipeline": pipeline_config.name}
            if pipeline_config.callback:
                pipeline_config.callback(pipeline_config.name, error_result)
            return pipeline_config.name, error_result

    def __call__(self, data: Any) -> Dict[str, Any]:
        """
        Execute all pipelines in parallel threads.

        Args:
            data: Input data to be processed by all pipelines

        Returns:
            Dictionary mapping pipeline names to their results
        """
        self._results.clear()

        with self._pipeline_lock:
            actual_pipelines = list(self.pipelines)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all pipeline executions
            futures: Dict[Future, str] = {}
            for pipeline_config in actual_pipelines:
                future = executor.submit(
                    self._execute_pipeline,
                    pipeline_config,
                    data
                )
                futures[future] = pipeline_config.name

            # Collect results
            if self.wait_for_all:
                for future in futures:
                    pipeline_name, result = future.result()
                    with self._lock:
                        self._results[pipeline_name] = result
            else:
                # Return immediately, callbacks handle async results
                pass

        return self._results.copy()

    def get_results(self) -> Dict[str, Any]:
        """Get current results (useful for non-blocking mode)."""
        with self._lock:
            return self._results.copy()
        
    
    def get_pipelines(self) -> List[PipelineConfig]:
        """Get the current list of pipeline configurations."""
        with self._pipeline_lock:
            return list(self.pipelines)
        
    def remove_pipeline(self, pipeline_name: str) -> bool:
        """Remove a pipeline configuration by name. Returns True if removed."""
        with self._pipeline_lock:
            for i, pc in enumerate(self.pipelines):
                if pc.name == pipeline_name:
                    del self.pipelines[i]
                    return True
        return False
        
        
    def add_pipeline(self, pipeline_config: PipelineConfig):
        """Add a new pipeline configuration."""
        # Thread-safe addition
        with self._pipeline_lock:
            self.pipelines.append(pipeline_config)
            