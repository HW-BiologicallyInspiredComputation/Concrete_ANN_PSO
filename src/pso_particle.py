import numpy as np
from dataclasses import dataclass
from typing import List

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
