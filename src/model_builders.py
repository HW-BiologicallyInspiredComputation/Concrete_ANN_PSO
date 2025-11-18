from sequential import Sequential
from linear import Linear
from activations import ActivationReLU

# Example model builder function


def build_base_model(input_size):
    # genome is accepted (in case you use it later)
    return Sequential(
        Linear(size_input=input_size, size_hidden=32),
        ActivationReLU(),
        Linear(size_input=32, size_hidden=16),
        ActivationReLU(),
        Linear(size_input=16, size_hidden=1),
    )
