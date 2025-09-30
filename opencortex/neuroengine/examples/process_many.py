# Create different pipelines
from opencortex.neuroengine.flux.base.pipe_config import PipelineConfig
from opencortex.neuroengine.flux.base.processor_group import ProcessorGroup
from opencortex.neuroengine.flux.nodes import LogNode, MultiplyNode, AddNode



# Define callbacks for each pipeline
def pipeline1_callback(name: str, result):
    print(f"Pipeline '{name}' completed with result: {result}")

def pipeline2_callback(name: str, result):
    print(f"Pipeline '{name}' completed with result: {result}")

def pipeline3_callback(name: str, result):
    print(f"Pipeline '{name}' completed with result: {result}")


if __name__ == "__main__":
    pipeline1 = MultiplyNode(2, "mul2") >> AddNode(10, "add10") >> LogNode("log1")
    pipeline2 = MultiplyNode(3, "mul3") >> AddNode(5, "add5") >> LogNode("log2")
    pipeline3 = AddNode(100, "add100") >> MultiplyNode(0.5, "mul0.5") >> LogNode("log3")
    # Create pipeline configurations
    configs = [
        PipelineConfig(
            pipeline=pipeline1,
            config={"param1": "value1"},
            callback=pipeline1_callback,
            name="Pipeline_A"
        ),
        PipelineConfig(
            pipeline=pipeline2,
            config={"param2": "value2"},
            callback=pipeline2_callback,
            name="Pipeline_B"
        ),
        PipelineConfig(
            pipeline=pipeline3,
            config={"param3": "value3"},
            callback=pipeline3_callback,
            name="Pipeline_C"
        ),
    ]

    # Create and execute the processor group
    processor_group = ProcessorGroup(
        pipelines=configs,
        name="BCI_ProcessorGroup",
        max_workers=3,
        wait_for_all=True
    )

    # Execute with input data
    input_data = 10
    print(f"\nProcessing input: {input_data}\n")
    results = processor_group(input_data)

    print(f"\nAll pipelines completed!")
    print(f"Results: {results}")

    # You can also compose ProcessorGroup with other nodes
    combined = LogNode("pre-process") >> processor_group >> LogNode("post-process")
    print(f"\n\nComposed pipeline: {combined}")