import numpy as np
from layer import Layer

# create a layer class for the MLP

class Linear(Layer):
    def __init__(self, size_input: int, size_hidden: int):
        """Initialize with weights and biases."""
        self.size_input = size_input
        self.size_hidden = size_hidden
        self.weights = self.init_weights()
        self.bias = self.init_biases()
        self.isVectorizable = True

    def init_weights(self, weight_scale=0.1):
        """Initialize weights with values from the standard normal distribution multiplied by weight_scale."""
        return np.random.randn(self.size_hidden, self.size_input) * weight_scale

    def init_biases(self, bias_scale=0.001):
        """Initialize biases at bias_scale."""
        return np.full((self.size_hidden, 1), bias_scale)

    def randomize(self, weight_scale=0.1, bias_scale=0.001):
        """Randomize weights and biases."""
        self.weights = self.init_weights(weight_scale=weight_scale)
        self.bias = self.init_biases(bias_scale=bias_scale)

    def forward(self, X):
        """Perform the forward pass: Y = WX + b."""
        return np.dot(self.weights, X) + self.bias

    def to_vector(self) -> np.ndarray:
        """Flatten weights and biases into a single vector."""
        return np.concatenate((self.weights.flatten(), self.bias.flatten()))

    def from_vector(self, vector: np.ndarray) -> int:
        """Set weights and biases from a single vector."""
        self.weights = vector[: self.weights.size].reshape(self.weights.shape)
        self.bias = vector[self.weights.size :].reshape(self.bias.shape)


if __name__ == "__main__":
    # Test the Linear class
    layer = Linear(size_input=5, size_hidden=3)
    X_sample = np.random.randn(1, 5).T
    output = layer.forward(X_sample)

    vector = layer.to_vector()
    layer2 = Linear(size_input=5, size_hidden=3)
    layer2.from_vector(vector)

    print(f"""
        bias:
        {layer.bias}
        weights:
        {layer.weights}
        input:
        {X_sample}
        output:
        {output}

        vector:
        {vector}
        layer2 bias:
        {layer2.bias}
        layer2 weights:
        {layer2.weights}
    """)
