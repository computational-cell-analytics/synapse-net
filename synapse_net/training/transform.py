import numpy as np
from torch_em.transform.label import labels_to_binary
from scipy.ndimage import distance_transform_edt


class AZDistanceLabelTransform:
    def __init__(self, max_distance: float = 50.0):
        self.max_distance = max_distance

    def __call__(self, input_):
        binary_target = labels_to_binary(input_).astype("float32")
        if binary_target.sum() == 0:
            distances = np.ones_like(binary_target, dtype="float32")
        else:
            distances = distance_transform_edt(binary_target == 0)
            distances = np.clip(distances, 0.0, self.max_distance)
            distances /= self.max_distance
        return np.stack([binary_target, distances])
