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
    
    
    # Export back to config
    export_path = "examples/cortex_nodes/exported_pipeline.yaml"
    pipeline.config_to_yaml(export_path)
    print(f"Exported pipeline config to: {export_path}")

    # Now create a new PipelineConfig from the exported file
    new_cfg = OmegaConf.load(export_path)
    new_pipeline = PipelineConfig.from_config(new_cfg)
    print(f"Re-loaded pipeline from exported config: {new_pipeline}")

    # Verify it works the same
    new_result = new_pipeline.pipeline(input_data)
    print(f"\nRe-loaded pipeline '{new_pipeline.name}' completed with result: {new_result}\n")