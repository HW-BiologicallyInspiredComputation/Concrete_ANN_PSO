# W7 - Part 1 - Implementation of Particle Swarm Optimization (PSO) for MLP Hyperparameter Tuning

Table of Contents
- [1. Creation of the Particle Class](#1-creation-of-the-particle-class)
- [2. Class ParticleSwarmOptimisation](#2-class-particleswarmoptimisation)
- [3. Updating the MLP Class for PSO](#3-updating-the-mlp-class-for-pso)


## 1. Creation of the Particle Class

While thinking about the implementation of the PSO algorithm, we figured it would be best to create a `Particle` class to represent each particle in the swarm. Each particle has a position (representing a set of hyperparameters for the MLP), a velocity, and a personal best position. The class will also include methods to update the particle's velocity and position based on its own experience and that of the swarm.

```
class Particle:
    def __init__(self, mlp: MLP, accel_coeff: AccelerationCoefficients):
        self.accel_coeff = accel_coeff
        # Initialize other attributes like position, velocity, personal best, etc.
        self.position = mlp
        self.personal_best = None
        self.informants = []
        self.fitness: float = 0.0

    def evaluate_fitness(self, X, target):
        self.prediction = self.mlp.forward(X)
        self.fitness = self.mlp.get_loss(self.prediction, target)

    def update_velocity(self, global_best):
        # Implement velocity update logic
        pass

    def update_position(self):
        pass
```

## 2. Class ParticleSwarmOptimisation

Now that we have the `Particle` class, a population becomes a list of particles. We can then completed the `ParticleSwarmOptimisation` class to manage the swarm, including initializing particles, updating their velocities and positions, intitializing informants, and tracking the global best solution found by the swarm.

```
class ParticleSwarmOptimisation:
    def __init__(
            self,
            X: np.ndarray[tuple[int, int]],
            swarm_size: int,
            epochs: int,
            accel_coeff: AccelerationCoefficients,
            num_informants: int,
            mlp_size_hidden_layers: List[int]
        ):
        self.epochs = epochs
        self.accel_coeff = accel_coeff
        self.num_informants = num_informants

        self.population: List[Particle] = []
        for _ in range(swarm_size):
            mlp = MLP(size_input=X.shape[1], size_hidden_layers=mlp_size_hidden_layers)
            self.population.append(Particle(mlp=mlp, accel_coeff=accel_coeff))

        self.global_best = None
```

In the initialisation we notice the creatuin of a Particle List representing the swarm population. Each particle is initialized with a new MLP instance, allowing for diverse hyperparameter configurations across the swarm.

```
    def update_informants(self):
        for particle in self.population:
            particle.informants = np.random.choice(self.population, size=self.num_informants, replace=False).tolist()
```

Here we have the initial implementation of the `update_informants` method, which randomly selects a specified number of informants for each particle from the swarm.

```
    def update_global_best(self):
        for particle in self.population:
            particle.evaluate_fitness(X=train_features, y=train_targets)
            if self.global_best == None or particle.fitness < self.global_best.fitness:
                self.global_best = deepcopy(particle)
```

This method evaluates the fitness of each particle and updates the global best particle if a better fitness is found.

```
    def update_velocities(self):
        for particle in self.population:
            particle.update_velocity(self.global_best)

    def update_positions(self):
        for particle in self.population:
            particle.update_position()

    def train(self):
        self.update_informants()
        for epoch in range(self.epochs):
            self.update_global_best()
            self.update_velocities()
            self.update_positions()

pso = ParticleSwarmOptimisation(
    X=train_features,
    swarm_size=30,
    epochs=100,
    accel_coeff=AccelerationCoefficients(
        inertia_weight=0.5,
        cognitive_weight=1.5,
        social_weight=1.5,
        global_best_weight=1.0,
        jump_size=0.1,
    ),
    num_informants=5,
    mlp_size_hidden_layers=[5, 5]
)
```

Finally, `def train` uses `update_informants`, `update_global_best`, `update_velocities`, and `update_positions` methods to perform the PSO optimization over a specified number of epochs.

## 3. Updating the MLP Class for PSO

To facilitate the use of the PSO we added some methods to the MLP and LayerMPL classes.

There are two main methods we added to the MLP class:
- `to_vector`: This method converts the MLP's weights and biases into a single vector. This is useful for PSO as it allows us to represent the MLP's parameters in a format that can be easily manipulated by the particles.
- `from_vector`: This method takes a vector of weights and biases and updates the MLP's parameters accordingly. This allows a particle to set the MLP's parameters based on its current position in the search space.
In the LayerMLP class, we added the following methods:
```
    def to_vector(self) -> np.ndarray:
        # Flatten weights and biases into a single vector
        return np.concatenate((self.weights.flatten(), self.bias.flatten()))

    def from_vector(self, param_vector: np.ndarray):
        # Set weights and biases from a single vector
        size_weights = self.weights.size
        self.weights = param_vector[:size_weights].reshape(self.weights.shape)
        self.bias = param_vector[size_weights:].reshape(self.bias.shape)
```
In the MLP class, we implemented the following methods:
```
    def to_vector(self) -> np.ndarray:
        # Concatenate parameters from all layers into a single vector
        param_vector = np.array([])
        for layer in self.layers:
            param_vector = np.concatenate((param_vector, layer.to_vector()))
        return param_vector

    def from_vector(self, param_vector: np.ndarray):
        # Set parameters for all layers from a single vector
        index = 0
        for layer in self.layers:
            size_layer_params = layer.weights.size + layer.bias.size
            layer.from_vector(param_vector[index:index + size_layer_params])
            index += size_layer_params
```
These methods will allow the PSO algorithm to effectively explore the hyperparameter space of the MLP by manipulating the weights and biases as vectors.