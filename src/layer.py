import numpy as np

# Base class for neural network layers
# Each specific layer type will inherit from this class


class Layer:
    def __init__(self):
        self.isVectorizable = False

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Forward method not implemented.")

    def randomize(self, weight_scale, bias_scale) -> None:
        pass
