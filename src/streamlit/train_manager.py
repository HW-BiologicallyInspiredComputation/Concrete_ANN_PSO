import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading
import time
import random
import pandas as pd
from db import save_progress, load_progress
from ga import GeneticPsoOptimizer, PsoEvaluator, PsoGenome, AccelerationCoefficientsGenome
from data import load_data
from utils import mean_squared_error
from sequential import Sequential
from linear import Linear
from activations import ActivationReLU

class TrainManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._training_thread = None
        self._stop_flag = False
        self._best_genome = None
        self._initialize_ga()

    def _initialize_ga(self):
        (train_features, train_targets), (test_features, test_targets) = load_data(path="../../data/concrete_data.csv")
        self.evaluator = PsoEvaluator(
            X=train_features.T,
            Y=train_targets,
            X_test=test_features.T,
            Y_test=test_targets,
            base_model_builder=lambda genome: Sequential(
                Linear(size_input=train_features.shape[1], size_hidden=32),
                ActivationReLU(),
                Linear(size_input=32, size_hidden=16),
                ActivationReLU(),
                Linear(size_input=16, size_hidden=1)
            ),
            loss_function=mean_squared_error,
            max_train_seconds=15.0,
            num_genome_repeats_per_iteration=5,
            max_repeats_per_genome=50,
            explosion_factor=100,
            accuracy_checks_every=20,
            patience_window=15,
            verbose=False
        )
        
        self.optimizer = GeneticPsoOptimizer(
            evaluator=self.evaluator,
            population_size=22,
            generations=50,
            mutation_rate=0.2,
            crossover_rate=0.7,
            elitism=8,
            tournament_k=3,
            parallel=False
        )

    def _train_loop(self):
        start_time = time.time()
        progress = load_progress()
        
        def seed_genome_factory():
            return PsoGenome(
                swarm_size=random.randint(10, 50),
                accel=AccelerationCoefficientsGenome(
                    inertia_weight=random.uniform(0.4, 0.9),
                    cognitive_weight=random.uniform(1.0, 3.0),
                    social_weight=random.uniform(0.5, 2.0),
                    global_best_weight=random.uniform(0.1, 1.0),
                    jump_size=random.uniform(0.01, 1.0),
                    max_velocity=random.uniform(0.001, 1.0),
                    max_position=random.uniform(0.1, 10.0)
                ),
                num_informants=random.randint(1, 5),
                particle_initial_position_scale=(random.uniform(0.0001, 0.1), random.uniform(0.0001, 0.1)),
                ann_layers=[32, 16]
            )

        self.optimizer.initialize(seed_genome_factory)
        
        for gen in range(self.optimizer.generations):
            if self._stop_flag:
                break
                
            self.optimizer.evaluate_population()
            best_ind = max(self.optimizer.population, key=lambda ind: ind.accuracy)
            self._best_genome = best_ind.genome  # Store best genome
            avg_fitness = sum(ind.accuracy for ind in self.optimizer.population) / len(self.optimizer.population)

            with self._lock:
                progress["epoch"] = gen + 1
                progress["percent"] = int(((gen + 1) / self.optimizer.generations) * 100)
                progress["best_fitness"] = best_ind.accuracy
                progress["avg_fitness"] = avg_fitness
                progress["elapsed_time"] = time.time() - start_time
                progress["best_genome"] = best_ind.genome
                progress["best_genome_timestamp"] = time.time()

                # Create new history DataFrame
                new_history = pd.DataFrame({
                    "best_accuracy": [best_ind.accuracy],
                    "avg_accuracy": [avg_fitness]
                })
                
                if "ga_history" not in progress or progress["ga_history"].empty:
                    progress["ga_history"] = new_history
                else:
                    progress["ga_history"] = pd.concat(
                        [progress["ga_history"], new_history],
                        ignore_index=True
                    ).reset_index(drop=True)
                
                save_progress(progress)
            
            self.optimizer.step()

        self._training_thread = None
        self._stop_flag = False

    def reset_progress(self):
        """Reset training progress to initial state."""
        initial_progress = {
            "epoch": 0,
            "percent": 0,
            "best_fitness": 0,
            "avg_fitness": 0,
            "elapsed_time": 0,
            "ga_history": pd.DataFrame({
                "best_accuracy": [],
                "avg_accuracy": []
            }),
            "loss_history": pd.DataFrame({"loss": []})
        }
        save_progress(initial_progress)

    def start_training(self):
        if not self.is_training():
            self.reset_progress()  # Reset progress before starting new training
            self._stop_flag = False
            self._training_thread = threading.Thread(target=self._train_loop, daemon=True)
            self._training_thread.start()

    def stop_training(self):
        if self.is_training():
            self._stop_flag = True
            self._training_thread.join(timeout=1)  # Wait for thread to finish
            self._training_thread = None
            self.reset_progress()  # Reset progress after stopping

    def is_training(self):
        return self._training_thread is not None and self._training_thread.is_alive()

    def get_progress(self):
        return load_progress()

    def get_best_genome(self):
        """Returns the best genome found during training"""
        progress = load_progress()
        return progress.get("best_genome")
