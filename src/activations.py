import numpy as np
from layer import Layer


class ActivationSigmoid(Layer):
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-input_data))


class ActivationReLU(Layer):
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        return np.maximum(0, input_data)


class ActivationTanh(Layer):
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        return np.tanh(input_data)


if __name__ == "__main__":
    sigmoid = ActivationSigmoid()
    relu = ActivationReLU()
    tanh = ActivationTanh()

    test_input = np.array([-2, -1, 0, 1, 2])

    print("Sigmoid:", sigmoid.forward(test_input))
    print("ReLU:", relu.forward(test_input))
    print("Tanh:", tanh.forward(test_input))
