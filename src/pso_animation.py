from enum import Enum
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import numpy as np
from typing import List
from pso_particle import Particle

ANIMATION_STEPS = 20
ANIMATION_INTERVAL = 50
USE_CAMERA_FOLLOW = True
CAMERA_ALWAYS_FOCUS = False
SAVE_ANIMATION = True
CAMERA_PERCENTILE = (
    95  # Percentile of particles to include in camera view between 0 and 100
)


class ReducerType(Enum):
    PCA = 1
    UMAP = 2


class Reducer:
    def fit_transform(self, data):
        pass

    def transform(self, data):
        pass


class PcaReducer(Reducer):
    def __init__(self):
        self.pca = PCA(n_components=2)

    def fit_transform(self, data):
        return self.pca.fit_transform(data)

    def transform(self, data):
        return self.pca.transform(data)


class UmapReducer(Reducer):
    def __init__(self):
        self.umap = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            random_state=42,
        )

    def fit_transform(self, data):
        return self.umap.fit_transform(data)

    def transform(self, data):
        return self.umap.transform(data)


class PsoAnimator:
    def __init__(self, reducer_type: ReducerType):
        if reducer_type == ReducerType.PCA:
            self.reducer = PcaReducer()
        elif reducer_type == ReducerType.UMAP:
            self.reducer = UmapReducer()

        self.particle_positions_history = []
        self.particle_fitness_history = []  # Add fitness history
        self.cmap = plt.cm.viridis  # Add colormap for particles

    def get_fitness_colors(self, population: List[Particle], fitnesses=None):
        # Get fitness values for all particles
        if fitnesses is None:
            fitnesses = np.array([p.fittest for p in population])
        # Normalize fitness values between 0 and 1 (reversed so better fitness = darker color)
        normalized = 1 - (fitnesses - fitnesses.min()) / (
            fitnesses.max() - fitnesses.min() + 1e-10
        )
        return self.cmap(normalized)

    def get_particle_positions(self, population: List[Particle]):
        positions = np.array([p.position for p in population])
        return self.reducer.fit_transform(positions)

    def get_camera_focus(self, positions_2d: np.ndarray, percentile: float = 50):
        """Return the centroid and a zoom radius containing a given percentile of particles."""
        # Centroid of particles
        center = np.mean(positions_2d, axis=0)

        # Distance of each particle to the center
        dists = np.linalg.norm(positions_2d - center, axis=1)

        # Radius that includes given percentile of particles
        radius = np.percentile(dists, percentile)

        return center, radius

    def get_best_particle_index(self, population: List[Particle]):
        return min(range(len(population)), key=lambda i: population[i].fittest)

    def plot(
        self,
        losses: List,
        avg_fitnesses: List,
        population: List[Particle],
        best_global,
        epoch=0,
    ):
        # if epoch % 2 == 0:

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot losses
        ax1.plot(losses, label="Loss")
        ax1.plot(avg_fitnesses, label="Average Fitness", linestyle="--")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.legend()

        # Get current positions and colors
        positions_2d = self.get_particle_positions(population)
        current_fitnesses = np.array([p.fittest for p in population])
        colors = self.get_fitness_colors(population, current_fitnesses)

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
        for particle in population:
            if not particle.informants:
                continue
            p_idx = population.index(particle)
            p_pos = positions_2d[p_idx]
            for informant in particle.informants:
                i_idx = population.index(informant)
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
        best_pos = self.reducer.transform(best_global.reshape(1, -1))
        ax2.scatter(
            best_pos[:, 0],
            best_pos[:, 1],
            c="red",
            marker="*",
            s=200,
            label="Global Best",
        )
        # After plotting all particles, highlight best particle
        best_particle_idx = self.get_best_particle_index(population)
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

    def create_animation(self, population: List[Particle]):
        positions_history = np.array(self.particle_positions_history)
        fitness_history = np.array(self.particle_fitness_history)

        fig, ax = plt.subplots(figsize=(8, 8))
        scatter = ax.scatter(
            positions_history[0, :, 0],
            positions_history[0, :, 1],
            c=self.get_fitness_colors(population, fitness_history[0]),
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
        for particle in population:
            if not particle.informants:
                continue
            p_idx = population.index(particle)
            p_pos = positions_2d[p_idx]
            for informant in particle.informants:
                i_idx = population.index(informant)
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
            colors = self.get_fitness_colors(population, fitness_history[frame])
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
            for particle in population:
                if not particle.informants:
                    continue
                p_idx = population.index(particle)
                p_pos = positions_2d[p_idx]
                for informant in particle.informants:
                    i_idx = population.index(informant)
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
