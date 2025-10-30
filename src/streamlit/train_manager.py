import threading
import time
import random
import pandas as pd
from db import save_progress, load_progress

class TrainManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._training_thread = None
        self._stop_flag = False

    def _train_loop(self):
        start_time = time.time()
        total_epochs = 50
        progress = load_progress()
        loss_history = progress["loss_history"]
        loss = progress["best_fitness"]

        for epoch in range(progress["epoch"], total_epochs):
            if self._stop_flag:
                break
            time.sleep(0.3)  # simulate computation
            loss *= random.uniform(0.9, 0.99)

            with self._lock:
                progress["epoch"] = epoch + 1
                progress["percent"] = int(((epoch + 1) / total_epochs) * 100)
                progress["best_fitness"] = min(progress["best_fitness"], loss)
                progress["elapsed_time"] = time.time() - start_time
                new_row = pd.DataFrame({"loss": [loss]})
                progress["loss_history"] = pd.concat([loss_history, new_row], ignore_index=True)

                save_progress(progress)
                loss_history = progress["loss_history"]

        self._training_thread = None
        self._stop_flag = False

    def reset_progress(self):
        """Reset training progress to initial state."""
        initial_progress = {
            "epoch": 0,
            "percent": 0,
            "best_fitness": float('inf'),
            "elapsed_time": 0,
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
