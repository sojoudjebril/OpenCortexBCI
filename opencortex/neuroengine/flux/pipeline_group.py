"""
Author: Michele Romani
Email: michele.romani.zaltieri@gmail.com
Copyright 2025 Michele Romani
"""
import logging
import threading
from typing import Any, Dict, Callable, Optional, List
from concurrent.futures import ThreadPoolExecutor, Future
from opencortex.neuroengine.flux.base.node import Node
from opencortex.neuroengine.flux.pipeline_config import PipelineConfig


class PipelineGroup:
    """
    Manages and executes multiple pipelines in parallel threads.
    Each pipeline receives the same input data and processes it independently according to its own configuration.
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
            name: Name for this pipeline group
            max_workers: Maximum number of threads (None = number of pipelines)
            wait_for_all: If True, waits for all pipelines to complete before returning the results dictionary. If False, returns immediately and results are handled via callbacks.
        """
        self.name = name or "PipelineGroup"
        self.pipelines = pipelines
        self.max_workers = max_workers or len(pipelines)
        self.wait_for_all = wait_for_all
        # Executor
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._shutdown = False
        # Results storage and locks
        self._results = {}        
        self._lock = threading.Lock()
        # Cache pipeline list to avoid repeated lock acquisition
        self._cached_pipelines = list(pipelines)
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

    def __call__(self, data):
        """Execute all pipelines in parallel threads with optimized executor reuse."""
        if self._shutdown:
            raise RuntimeError("PipelineGroup has been shut down")
        
        # Use cached pipeline list if no modifications
        pipelines_to_run = self._cached_pipelines
        
        # Submit all pipeline executions to reused executor
        futures = {}
        for pipeline_config in pipelines_to_run:
            future = self.executor.submit(
                self._execute_pipeline,
                pipeline_config,
                data
            )
            futures[future] = pipeline_config.name

        # Collect results
        if self.wait_for_all:
            results = {}
            for future in futures:
                pipeline_name, result = future.result()
                results[pipeline_name] = result
            
            with self._lock:
                self._results = results
            return results.copy()
        
        return {}
    
    def shutdown(self):
        """Shut down the executor and clean up resources."""
        self._shutdown = True
        self.executor.shutdown(wait=True)
        logging.info(f"PipelineGroup '{self.name}' has been shut down.")
    
    
    def get_configs(self) -> List[Dict[str, Any]]:
        """Get configurations of all pipelines."""
        with self._pipeline_lock:
            return [pc.get_config() for pc in self.pipelines]

    def get_results(self) -> Dict[str, Any]:
        """Get current results (useful for non-blocking mode)."""
        with self._lock:
            return self._results.copy()

    def get_pipeline(self, pipeline_name: str) -> PipelineConfig | None:
        """Get a pipeline configuration by name."""
        with self._pipeline_lock:
            for pc in self.pipelines:
                if pc.name == pipeline_name:
                    return pc
            raise ValueError(f"Pipeline '{pipeline_name}' not found")
        
    def get_all_pipelines(self) -> List[PipelineConfig]:
        """Get a list of all pipeline configurations."""
        with self._pipeline_lock:
            return list(self.pipelines)

    def remove_pipeline(self, pipeline_name):
        """Remove a pipeline configuration by name."""
        with self._pipeline_lock:
            for i, pc in enumerate(self.pipelines):
                if pc.name == pipeline_name:
                    del self.pipelines[i]
                    self._cached_pipelines = list(self.pipelines)
                    return True
        return False
        
    def add_pipeline(self, pipeline_config: PipelineConfig):
        """Add a new pipeline configuration."""
        # Thread-safe addition
        with self._pipeline_lock:
            self.pipelines.append(pipeline_config)
            self._cached_pipelines = list(self.pipelines)
