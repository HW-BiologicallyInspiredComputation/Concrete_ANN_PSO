# W6 - Part 2 - cd..Coding of the Multi-Layer Perceptron (MLP)

Table of Contents
- [1. Class Definition](#1-class-definition)
  - [1.1 LayerMLP Class](#11-layermlp-class)
  - [1.2 MLP Class](#12-mlp-class)
- [2. Initialisation of the PSO](#2-initialisation-of-the-pso)
  - [2.1 AccelerationCoefficients Class](#21-accelerationcoefficients-class)
  - [2.2 ParticleSwarmOptimizer Class](#22-particleswarmoptimizer-class)
- [3. Next Steps](#3-next-steps)


## 1. Class Definition

### 1.1 LayerMLP Class

We first defined a class called `LayerMLP` to represent a layer in the Multi-Layer Perceptron (MLP). This class includes methods for initializing weights and biases, performing the forward pass, and a sigmoid activation function.

We made this class flexible by allowing the user to specify the input_size and the number of neurons in the layer. The weights are initialized randomly, and biases are initialized to 0.01.

```
class LayerMLP:
    def __init__(self, size_input: int, size_hidden: int):
        # Initialize weights and biases
        self.bias = np.full((size_hidden, 1), 0.01)
        self.weights = np.random.randn(size_hidden, size_input)

    def sigmoid(self, z):
        z_clip = np.clip(z, -500, 500)  # Prevent overflow
        return 1/(1 + np.exp(-z_clip))

    def forward(self, X):
        z = np.dot(self.weights, X) + self.bias
        a = self.sigmoid(z)
        return a
    
# Test the LayerMLP class
# layer = LayerMLP(size_input=5, size_hidden=3)
# print(layer.bias)
# print(layer.weights)

# X_sample = np.random.randn(1, 5).T
# print(X_sample)
# output = layer.forward(X_sample)
# print(output)
```

### 1.2 MLP Class

Next, we defined the `MLP` class, which represents the entire Multi-Layer Perceptron. This class creates an MLP with a specified number of layers and neurons per layer. Each Layer is instantiated using the `LayerMLP` class. This makes the MLP modular and easy to extend. It also includes a method for performing the forward pass through all layers of the network.

```
class MLP:
    def __init__(self, size_input: int, size_hidden_layers: List[int]):
        self.layers: List[LayerMLP] = []
        self.layers.append(LayerMLP(size_input=size_input, size_hidden=size_hidden_layers[0])) # first hidden layer

        size_layers = size_hidden_layers + [1]  # add output layer with one neuron
        for i in range(len(size_layers) - 1): # remaining hidden layers and output layer
            layer = LayerMLP(size_input=size_layers[i], size_hidden=size_layers[i+1])
            self.layers.append(layer)

    def forward(self, X):
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a

# Test the MLP class
mlp = MLP(size_input=5, size_hidden_layers=[3, 4])
X_sample = np.random.randn(1, 5).T
output = mlp.forward(X_sample)
print(output)
```

As we have to use a Particle Swarm Optimization (PSO) algorithm to optimize the hyperparameters of the MLP, we ensured that our MLP class is flexible enough to accommodate different architectures by allowing the user to specify the number of hidden layers and neurons per layer. This will enable us to easily experiment with different configurations during the optimization process. The PSO is also the reason why we did not implement backpropagation and training methods in this initial version of the MLP.

## 2. Initialisation of the PSO

As we had some time left after implementing the MLP, we started to set up the initial structure for the Particle Swarm Optimization (PSO) algorithm.

### 2.1 AccelerationCoefficients Class

We defined a data class called `AccelerationCoefficients` to hold the cognitive and social acceleration coefficients used in the PSO algorithm. These coefficients influence how much a particle is attracted to its own best position and the global best position. We also added inertia weight to control the impact of the previous velocity on the current velocity and jump size to limit the maximum change in velocity. Finally, we added global_best_weight to adjust the influence of the global best position on the particle's velocity.

```
@dataclass
class AccelerationCoefficients:
    inertia_weight: float
    cognitive_weight: float
    social_weight: float
    global_best_weight: float
    jump_size: float
```

### 2.2 ParticleSwarmOptimizer Class

We then defined the `ParticleSwarmOptimizer` class, which will manage the PSO process. We simply set up the `__init__` method's parameters for now, including the swarm size, number of iterations, acceleration coefficients, and number of informants.

```
class ParticleSwarmOptimisation:
    def __init__(self, swarm_size: int, epochs: int, accel_coeff: AccelerationCoefficients, num_informants: int):
        pass

pso = ParticleSwarmOptimisation(swarm_size=30,
          epochs=100,
          accel_coeff=AccelerationCoefficients(
              inertia_weight=0.5,
              cognitive_weight=1.5,
              social_weight=1.5,
              global_best_weight=1.0,
              jump_size=0.1,
          ),
          num_informants=5
        )        
```

## 3. Next Steps

With the MLP and initial PSO structure in place, we are now ready to proceed with the next steps of the coursework. Next week we will focus on implementing the PSO algorithm to optimize the hyperparameters of the MLP and integrating the two components to create a complete solution for predicting concrete's compressive strength.