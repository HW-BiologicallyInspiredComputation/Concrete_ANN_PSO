import numpy as np

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))

if __name__ == "__main__":
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])

    print("MSE:", mean_squared_error(y_true, y_pred))  # Expected: 0.375
    print("MAE:", mean_absolute_error(y_true, y_pred))  # Expected: 0.5
