import numpy as np
from typing import List
from layer import Layer
from linear import Linear
from activations import ActivationReLU, ActivationSigmoid

class Sequential:
    def __init__(self, *layers: Layer, randomize: bool = True):
        self.layers = layers
        self.vectorizable_layers: List[Linear] = [layer for layer in self.layers if layer.isVectorizable]
        self.vector_indexes = []
        index = 0
        for layer in self.vectorizable_layers:
            size_layer_params = layer.weights.size + layer.bias.size
            self.vector_indexes.append((index, index + size_layer_params))
            index += size_layer_params

        if randomize:
            self.randomize()

    def randomize(self, weight_scale=0.1, bias_scale=0.001):
        for layer in self.layers:
            layer.randomize(weight_scale=weight_scale, bias_scale=bias_scale)

    def forward(self, X: np.ndarray) -> np.ndarray:
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def to_vector(self) -> np.ndarray:
        """Concatenate parameters from all layers into a single vector."""
        param_vector = np.array([])
        for layer in self.vectorizable_layers:
            param_vector = np.concatenate((param_vector, layer.to_vector()))
        return param_vector

    def from_vector(self, param_vector: np.ndarray):
        """Set parameters from all layers from a single vector."""
        for i in range(len(self.vectorizable_layers)):
            start_idx, end_idx = self.vector_indexes[i]
            self.vectorizable_layers[i].from_vector(param_vector[start_idx:end_idx])
            
if __name__ == "__main__":
    # Test the Sequential class

    mlp = Sequential(
        Linear(size_input=5, size_hidden=3),
        ActivationReLU(),
        Linear(size_input=3, size_hidden=4),
        ActivationSigmoid(),
        Linear(size_input=4, size_hidden=1)
    )
    X_sample = np.random.randn(1, 5).T
    output = mlp.forward(X_sample)
    vector = mlp.to_vector()
    print(f"""
input:
{X_sample}
output:
{output}

vector:
{vector}
    """)

    mlp.randomize()
    print(f"""
    initial model2 output:
    {mlp.forward(X_sample)}
    """)

    mlp.from_vector(vector)
    print(f"""model2 output after from_vector:
    {mlp.forward(X_sample)}
        """)
