import opencortex.neuroengine.flux.base.operators
from opencortex.neuroengine.flux.base.node import Node
import time

class MultiplyNode(Node):
    def __init__(self, factor: float, name: str = None):
        super().__init__(name)
        self.factor = factor

    def __call__(self, data):
        time.sleep(0.1)  # Simulate processing
        return data * self.factor


class AddNode(Node):
    def __init__(self, value: float, name: str = None):
        super().__init__(name)
        self.value = value

    def __call__(self, data):
        time.sleep(0.1)  # Simulate processing
        return data + self.value


class LogNode(Node):
    def __call__(self, data):
        print(f"[LogNode] Data: {data}, shape: {getattr(data, 'shape', 'N/A')}, type: {type(data)}")
        return data