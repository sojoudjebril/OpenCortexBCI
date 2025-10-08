# OpenCortexBCI Examples

This folder contains example projects demonstrating various functionalities of the OpenCortexBCI. The examples are organized into three subfolders:

## 1. [`brainflow`](./brainflow/)
Basic examples showing how to connect to a Brain-Computer Interface (BCI) and retrieve EEG data using the [BrainFlow](https://brainflow.org/) library. These examples do **not** use OpenCortexBCI-specific features, but are useful for understanding data acquisition.

## 2. [`cortex_utils`](./cortex_utils/)
Examples illustrating the usage of utility functions from `opencortex.utils`. These utilities are helpful for data manipulation tasks such as filtering, normalization, and transformation of EEG data.

## 3. [`cortex_nodes`](./cortex_nodes/)
Advanced examples focused on the `Nodes` class from `opencortex.neuroengine.flux`. 
Each node represents a modular processing unit that can be chained together to create flexible and powerful data processing pipelines.
Here you will learn how to:
- Connect and configure processing nodes.
- Build pipelines for EEG data processing.
- Use nodes for data transformation and manipulation.
- Train neural networks (using PyTorch) or machine learning models (using scikit-learn).

---

Explore each subfolder for detailed, runnable examples to help you get started with OpenCortexBCI!