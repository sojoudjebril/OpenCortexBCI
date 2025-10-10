# Create different pipelines
from opencortex.neuroengine.flux.pipeline_config import PipelineConfig
from opencortex.neuroengine.flux.pipeline_group import PipelineGroup
from opencortex.neuroengine.flux.base.simple_nodes import LogNode, MultiplyNode, AddNode
from omegaconf import OmegaConf


cfg = OmegaConf.load("examples/cortex_nodes/pipeline.yaml")
print(OmegaConf.to_yaml(cfg))


pipeline = PipelineConfig.from_config(cfg)


print(f"Loaded pipeline from config: {pipeline}")


# Process
if __name__ == "__main__":
    input_data = 5
    print(f"\nProcessing input: {input_data}\n")
    result = pipeline.pipeline(input_data)
    print(f"\nPipeline '{pipeline.name}' completed with result: {result}\n")