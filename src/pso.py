import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from utils import mean_absolute_error, InformantStrategy
from sequential import Sequential
from linear import Linear
from activations import ActivationReLU
from utils import mean_squared_error
from data import load_data
from pso_animation import PsoAnimator, ReducerType
from pso_particle import Particle, AccelerationCoefficients

# Creation of AI class for training the MLP with our PSO


# Define Particle Swarm Optimization class


class ParticleSwarmOptimisation:
    """Class representing the Particle Swarm Optimization algorithm."""

    def __init__(
        self,
        X: np.ndarray[tuple[int, int]],
        Y: np.ndarray[tuple[int]],
        swarm_size: int,
        accel_coeff: AccelerationCoefficients,
        num_informants: int,
        loss_function: callable,
        informants_strategy: InformantStrategy,
        particle_initial_position_scale: Tuple[float, float],
        model: Sequential,
        animator: PsoAnimator = None,
    ):
        """Initialize the PSO with data, swarm parameters, loss function, and model.
        Also initializes the particle population."""
        self.accel_coeff = accel_coeff
        self.swarm_size = swarm_size
        self.num_informants = num_informants

        self.X = X
        self.Y = Y

        self.loss_function = loss_function
        self.informants_strategy = informants_strategy
        self.model = model

        self.losses = []
        self.avg_fitnesses = []

        self.population: List[Particle] = []
        for _ in range(swarm_size):
            self.model.randomize(
                weight_scale=particle_initial_position_scale[0],
                bias_scale=particle_initial_position_scale[1],
            )
            particle_fitness = self.loss_function(self.Y, self.model.forward(self.X))
            self.population.append(
                Particle(
                    position=self.model.to_vector(),
                    accel_coeff=accel_coeff,
                    fitness=particle_fitness,
                )
            )

        self.best_global: np.ndarray = self.population[0].position.copy()
        self.best_global_fitness: float = self.population[0].fittest

        self.animator: PsoAnimator = animator

    def update_informants_random(self):
        """Randomly assign informants to each particle."""
        if self.num_informants >= self.swarm_size:
            raise ValueError("Number of informants must be less than swarm size.")
        for particle in self.population:
            others = [p for p in self.population if p is not particle]
            particle.informants = np.random.choice(
                others, size=self.num_informants, replace=False
            )

    def update_informants_nearest(self):
        """Assign informants based on nearest particles in position space."""
        if self.num_informants >= self.swarm_size:
            raise ValueError("Number of informants must be less than swarm size.")
        for particle in self.population:
            distances = []
            for other in self.population:
                if other is not particle:
                    dist = np.linalg.norm(particle.position - other.position)
                    distances.append((dist, other))
            distances.sort(key=lambda x: x[0])
            particle.informants = [distances[i][1] for i in range(self.num_informants)]

    def update_best_global(self):
        """Update the best global position and fitness based on the current population.
        Also returns the average fitness of the population."""
        loss = 0.0
        fitnesses = []
        for particle in self.population:
            self.model.from_vector(particle.position)
            fitness = self.loss_function(self.Y, self.model.forward(self.X))
            fitnesses.append(fitness)
            loss += fitness
            if fitness < particle.fittest:
                particle.best_personal = particle.position.copy()
                particle.fittest = fitness
                if (
                    self.best_global_fitness is None
                    or fitness < self.best_global_fitness
                ):
                    self.best_global = particle.position.copy()
                    self.best_global_fitness = fitness
        return np.mean(fitnesses)

    def get_accuracy(self, x, y_true) -> float:
        """Evaluate the accuracy in percent of the best global model on given data."""
        self.model.from_vector(self.best_global)
        y_pred = self.model.forward(x)

        mae = mean_absolute_error(y_true, y_pred)
        accuracy = 100 * (1.0 - mae / np.mean(np.abs(y_true)))

        return accuracy

    def update_velocities(self):
        """Update the velocities of all particles in the population."""
        for particle in self.population:
            particle.update_velocity(self.best_global)

    def update_positions(self):
        """Update the positions of all particles in the population."""
        for particle in self.population:
            particle.update_position()

    def train_epoch(self, epoch):
        """Perform a single epoch of training."""
        if self.informants_strategy == InformantStrategy.RANDOM and epoch == 0:
            self.update_informants_random()
        if self.informants_strategy == InformantStrategy.KNEAREST:
            self.update_informants_nearest()
        avg_fitness = self.update_best_global()
        self.update_velocities()
        self.update_positions()

        self.avg_fitnesses.append(avg_fitness)
        self.losses.append(self.best_global_fitness)
        if self.animator is not None:
            self.animator.plot(
                self.losses,
                self.avg_fitnesses,
                self.population,
                self.best_global,
                epoch=epoch,
            )
        return avg_fitness

    def train(self, epochs):
        for epoch in range(epochs):
            avg_fitness = self.train_epoch(epoch)

        return (self.best_global, self.best_global_fitness, self.losses)


if __name__ == "__main__":
    (train_features, train_targets), (test_features, test_targets) = load_data(
        "../data/concrete_data.csv"
    )
    mlp = Sequential(
        Linear(size_input=train_features.shape[1], size_hidden=12),
        ActivationReLU(),
        Linear(size_input=12, size_hidden=12),
        ActivationReLU(),
        Linear(size_input=12, size_hidden=1),
    )

    predictions = mlp.forward(test_features.T)

    # plt.scatter(test_targets, predictions)
    # plt.xlabel("True Values")
    # plt.ylabel("Predictions")
    # plt.title("Predictions vs True Values")
    # plt.plot([test_targets.min(), test_targets.max()], [test_targets.min(), test_targets.max()], 'k--', lw=2)
    # plt.show()

    swarm_size = 50
    epochs = 100
    accel_coeff = AccelerationCoefficients(
        inertia_weight=0.68,
        cognitive_weight=2.80,
        social_weight=0.88,
        global_best_weight=0.96,
        jump_size=0.4,
        max_velocity=0.2,
        max_position=2.87,
    )
    num_informants = 3
    particle_initial_position_scale = (0.0001, 0.087)
    loss_function = mean_squared_error
    informants_strategy = InformantStrategy.KNEAREST

    animator = PsoAnimator(reducer_type=ReducerType.PCA)

    pso = ParticleSwarmOptimisation(
        X=train_features.T,
        Y=train_targets,
        swarm_size=swarm_size,
        accel_coeff=accel_coeff,
        num_informants=num_informants,
        loss_function=loss_function,
        informants_strategy=informants_strategy,
        particle_initial_position_scale=particle_initial_position_scale,
        model=mlp,
        animator=animator
    )

    (final_position, final_score, losses) = pso.train(epochs)

    # After training, create animation of particle movements
    if pso.animator is not None:
        pso.animator.create_animation(pso.population)
    print(f"Final particle fitness: {final_score}")
    print(f"Final particle position sample: {final_position[:5]}")
    mlp.from_vector(final_position)
    predictions = mlp.forward(test_features.T)

    plt.scatter(test_targets, predictions)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("Predictions vs True Values")
    plt.plot(
        [test_targets.min(), test_targets.max()],
        [test_targets.min(), test_targets.max()],
        "k--",
        lw=2,
    )
    plt.show()

    plt.plot(losses)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.show()

    # Accuracy
    train_accuracy = pso.get_accuracy(train_features.T, train_targets)
    print(f"Train Accuracy: {train_accuracy:.2f}%")
    test_accuracy = pso.get_accuracy(test_features.T, test_targets)
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    print(f"""
        Params:
        model: MLP with layers {[type(layer).__name__ for layer in mlp.layers]}
        swarm_size: {swarm_size}
        epochs: {epochs}
        accel_coeff: {accel_coeff}
        num_informants: {num_informants}
        loss_function: {loss_function.__name__}
    """)

    print("Final loss", losses[-1])
