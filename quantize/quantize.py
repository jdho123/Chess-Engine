import numpy as np


def quantize(array: np.ndarray, scaling_factor: int, type_: np.dtype) -> np.ndarray:
    scaled = array * scaling_factor

    rounded = np.round(scaled)

    bounds = np.iinfo(type_)

    return np.clip(rounded, bounds.min, bounds.max).astype(type_)
