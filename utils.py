import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    return v - np.mean(v)
