import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from sklearn.decomposition import PCA
import umap
from utils import mean_absolute_error, InformantStrategy
from sequential import Sequential
from linear import Linear
from activations import ActivationReLU
from utils import mean_squared_error
from data import load_data

# Creation of AI class for training the MLP with our PSO

ANIMATION_STEPS = 20
ANIMATION_INTERVAL = 50
USE_CAMERA_FOLLOW = True
CAMERA_ALWAYS_FOCUS = False
SAVE_ANIMATION = True
CAMERA_PERCENTILE = (
    95  # Percentile of particles to include in camera view between 0 and 100
)

# Define acceleration coefficients for PSO

@dataclass
class AccelerationCoefficients:
    inertia_weight: float
    cognitive_weight: float
    social_weight: float
    global_best_weight: float
    jump_size: float
    max_velocity: float
    max_position: float


# Define Particle class for PSO algorithm


class Particle:
    """Class representing a particle in the swarm."""

    def __init__(
        self,
        position: np.ndarray,
        accel_coeff: AccelerationCoefficients,
        fitness: float,
    ):
        """Initialize a particle with position, velocity, and acceleration coefficients.
        But also with personal best position, fitness and informants."""
        self.accel_coeff = accel_coeff
        self.position = position
        self.velocity = np.random.randn(position.shape[0]) * 0.1
        self.fittest = fitness
        self.informants: List[Particle] = []

        self.best_personal: np.ndarray = position.copy()

    def get_best_informant(self):
        """Get the position of the best informant based on fitness."""
        informant_fittest = None
        best_informant = None
        for informant in self.informants:
            if best_informant is None or informant.fittest < informant_fittest:
                informant_fittest = informant.fittest
                best_informant = informant
        return best_informant.position

    def update_velocity(self, best_global):
        """Update the velocity of the particle based on personal best, informants' best, and global best."""
        best_informant = self.get_best_informant()
        for i in range(len(self.position)):
            b = np.random.random() * self.accel_coeff.cognitive_weight
            c = np.random.random() * self.accel_coeff.social_weight
            d = np.random.random() * self.accel_coeff.global_best_weight
            inertia = self.accel_coeff.inertia_weight * self.velocity[i]
            velocity_cognitive = b * (self.best_personal[i] - self.position[i])
            velocity_social = c * (best_informant[i] - self.position[i])
            velocity_global = d * (best_global[i] - self.position[i])
            new_velocity = (
                inertia + velocity_cognitive + velocity_social + velocity_global
            )
            self.velocity[i] = np.clip(
                new_velocity,
                -self.accel_coeff.max_velocity,
                self.accel_coeff.max_velocity,
            )

    def update_position(self):
        """Update the position of the particle based on its velocity.
        Clip the position to be within the allowed range and avoid exploding positions."""
        self.position += self.velocity * self.accel_coeff.jump_size
        self.position = np.clip(
            self.position, -self.accel_coeff.max_position, self.accel_coeff.max_position
        )


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
        loss_function,
        particle_initial_position_scale: Tuple[float, float],
        model: Sequential,
    ):
        """Initialize the PSO with data, swarm parameters, loss function, and model.
        Also initializes the particle population."""
        self.accel_coeff = accel_coeff
        self.swarm_size = swarm_size
        self.num_informants = num_informants

        self.X = X
        self.Y = Y

        self.loss_function = loss_function
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

        self.pca = PCA(n_components=2)
        # self.umap = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)

        self.particle_positions_history = []
        self.particle_fitness_history = []  # Add fitness history
        self.cmap = plt.cm.viridis  # Add colormap for particles

    def get_particle_positions(self):
        positions = np.array([p.position for p in self.population])
        return self.pca.fit_transform(positions)
        # return self.umap.fit_transform(positions)

    def get_camera_focus(self, positions_2d: np.ndarray, percentile: float = 50):
        """Return the centroid and a zoom radius containing a given percentile of particles."""
        # Centroid of particles
        center = np.mean(positions_2d, axis=0)

        # Distance of each particle to the center
        dists = np.linalg.norm(positions_2d - center, axis=1)

        # Radius that includes given percentile of particles
        radius = np.percentile(dists, percentile)

        return center, radius

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

    def get_fitness_colors(self, fitnesses=None):
        # Get fitness values for all particles
        if fitnesses is None:
            fitnesses = np.array([p.fittest for p in self.population])
        # Normalize fitness values between 0 and 1 (reversed so better fitness = darker color)
        normalized = 1 - (fitnesses - fitnesses.min()) / (
            fitnesses.max() - fitnesses.min() + 1e-10
        )
        return self.cmap(normalized)

    def get_best_particle_index(self):
        return min(
            range(len(self.population)), key=lambda i: self.population[i].fittest
        )

    def plot(self, epoch, avg_fitness):
        # if epoch % 2 == 0:
        self.avg_fitnesses.append(avg_fitness)
        self.losses.append(self.best_global_fitness)

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot losses
        ax1.plot(self.losses, label="Loss")
        ax1.plot(self.avg_fitnesses, label="Average Fitness", linestyle="--")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.legend()

        # Get current positions and colors
        positions_2d = self.get_particle_positions()
        current_fitnesses = np.array([p.fittest for p in self.population])
        colors = self.get_fitness_colors(current_fitnesses)

        # Plot particle positions
        if self.particle_positions_history:
            # Get previous positions in PCA space
            prev_positions = self.particle_positions_history[-1]
            prev_fitnesses = self.particle_fitness_history[-1]

            # Interpolate positions and fitnesses between previous and current
            for step in range(1, ANIMATION_STEPS + 1):
                alpha = step / (ANIMATION_STEPS + 1)
                interpolated_positions = prev_positions + alpha * (
                    positions_2d - prev_positions
                )
                interpolated_fitnesses = prev_fitnesses + alpha * (
                    current_fitnesses - prev_fitnesses
                )
                self.particle_positions_history.append(interpolated_positions)
                self.particle_fitness_history.append(interpolated_fitnesses)
        else:
            # First epoch, just append initial positions and fitnesses
            self.particle_positions_history.append(positions_2d)
            self.particle_fitness_history.append(current_fitnesses)

        scatter = ax2.scatter(
            positions_2d[:, 0], positions_2d[:, 1], c=colors, alpha=0.6
        )
        for particle in self.population:
            if not particle.informants:
                continue
            p_idx = self.population.index(particle)
            p_pos = positions_2d[p_idx]
            for informant in particle.informants:
                i_idx = self.population.index(informant)
                i_pos = positions_2d[i_idx]
                ax2.plot(
                    [p_pos[0], i_pos[0]],
                    [p_pos[1], i_pos[1]],
                    color="gray",
                    alpha=0.2,
                    linewidth=0.7,
                    zorder=0,  # ensures lines stay behind points
                )
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.set_title(f"Particle Positions (PCA) - Epoch {epoch // ANIMATION_STEPS}")

        # Add global best position
        best_pos = self.pca.transform(self.best_global.reshape(1, -1))
        # best_pos = self.umap.transform(self.best_global.reshape(1, -1))
        ax2.scatter(
            best_pos[:, 0],
            best_pos[:, 1],
            c="red",
            marker="*",
            s=200,
            label="Global Best",
        )
        # After plotting all particles, highlight best particle
        best_particle_idx = self.get_best_particle_index()
        best_particle_pos = positions_2d[best_particle_idx]
        ax2.scatter(
            best_particle_pos[0],
            best_particle_pos[1],
            c="yellow",
            marker="o",
            s=150,
            edgecolor="black",
            linewidth=1,
            label="Best Particle",
            zorder=3,  # Ensure it's drawn on top
        )
        ax2.legend()

        # Add colorbar
        plt.colorbar(scatter, ax=ax2, label="Fitness (darker is better)")

        clear_output(wait=True)
        display(fig)
        plt.close(fig)

    def train_epoch(self):
        """Perform a single epoch of training."""
        avg_fitness = self.update_best_global()
        self.update_velocities()
        self.update_positions()
        return avg_fitness

    def train(self, epochs, informants_strategy: InformantStrategy):
        if informants_strategy == InformantStrategy.RANDOM:
            self.update_informants_random()
        for epoch in range(epochs):
            if informants_strategy == InformantStrategy.KNEAREST:
                self.update_informants_nearest()
            avg_fitness = self.train_epoch()
            self.plot(epoch, avg_fitness)

        # After training, create animation of particle movements
        self.create_animation()
        return (self.best_global, self.best_global_fitness, self.losses)

    def create_animation(self):
        positions_history = np.array(self.particle_positions_history)
        fitness_history = np.array(self.particle_fitness_history)

        fig, ax = plt.subplots(figsize=(8, 8))
        scatter = ax.scatter(
            positions_history[0, :, 0],
            positions_history[0, :, 1],
            c=self.get_fitness_colors(fitness_history[0]),
        )
        # Add best particle highlight
        best_particle_scatter = ax.scatter(
            [],
            [],
            c="yellow",
            marker="o",
            s=150,
            edgecolor="black",
            linewidth=1,
            label="Best Particle",
            zorder=3,
        )
        lines = []
        positions_2d = positions_history[0]
        for particle in self.population:
            if not particle.informants:
                continue
            p_idx = self.population.index(particle)
            p_pos = positions_2d[p_idx]
            for informant in particle.informants:
                i_idx = self.population.index(informant)
                i_pos = positions_2d[i_idx]
                (line,) = ax.plot(
                    [p_pos[0], i_pos[0]],
                    [p_pos[1], i_pos[1]],
                    color="gray",
                    alpha=0.15,
                    linewidth=0.7,
                    zorder=0,
                )
                lines.append(line)
        plt.colorbar(scatter, ax=ax, label="Fitness (brighter is better)")

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        def update(frame):
            positions_2d = positions_history[frame]
            colors = self.get_fitness_colors(fitness_history[frame])
            scatter.set_offsets(positions_2d)
            scatter.set_array(colors[:, 0])
            ax.set_title(f"Particle Positions - Epoch {frame // ANIMATION_STEPS}")
            # Update best particle position
            best_idx = np.argmin(fitness_history[frame])
            best_particle_scatter.set_offsets([positions_2d[best_idx]])

            if USE_CAMERA_FOLLOW and (
                CAMERA_ALWAYS_FOCUS or frame % ANIMATION_STEPS == 0
            ):
                # Adjust camera to follow majority of particles (50%)
                center, radius = self.get_camera_focus(
                    positions_2d, percentile=CAMERA_PERCENTILE
                )
                center = (0, 0)
                # Smooth pan/zoom by setting axis limits
                zoom_factor = 5  # more = looser framing
                ax.set_xlim(
                    center[0] - radius * zoom_factor, center[0] + radius * zoom_factor
                )
                ax.set_ylim(
                    center[1] - radius * zoom_factor, center[1] + radius * zoom_factor
                )

            # Update informant lines
            for line in lines:
                line.remove()
            lines.clear()
            for particle in self.population:
                if not particle.informants:
                    continue
                p_idx = self.population.index(particle)
                p_pos = positions_2d[p_idx]
                for informant in particle.informants:
                    i_idx = self.population.index(informant)
                    i_pos = positions_2d[i_idx]
                    (line,) = ax.plot(
                        [p_pos[0], i_pos[0]],
                        [p_pos[1], i_pos[1]],
                        color="gray",
                        alpha=0.15,
                        linewidth=0.7,
                        zorder=0,
                    )
                    lines.append(line)

            return (scatter, best_particle_scatter, *lines)

        from matplotlib.animation import FuncAnimation

        anim = FuncAnimation(
            fig, update, frames=len(positions_history), interval=ANIMATION_INTERVAL
        )
        if SAVE_ANIMATION:
            anim.save("particle_swarm_optimization.mp4", dpi=150)
        plt.show()


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

    pso = ParticleSwarmOptimisation(
        X=train_features.T,
        Y=train_targets,
        swarm_size=swarm_size,
        accel_coeff=accel_coeff,
        num_informants=num_informants,
        loss_function=loss_function,
        particle_initial_position_scale=particle_initial_position_scale,
        model=mlp,
    )

    (final_position, final_score, losses) = pso.train(epochs, informants_strategy)
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
