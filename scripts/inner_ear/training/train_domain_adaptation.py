import os
from glob import glob

import numpy as np
from sklearn.model_selection import train_test_split
from synaptic_reconstruction.training.domain_adaptation import mean_teacher_adaptation

from train_structure_segmentation import noop  # noqa

ROOT = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/moser/other_tomograms/"
NAMES = ["vesicle_pools", "tether", "rat"]


def _get_paths(name):
    assert name in NAMES, f"Invalid name {name}"
    if name == "vesicle_pools":
        folder = "01_vesicle_pools"
    elif name == "tether":
        folder = "02_tether"
    else:
        folder = "03_ratten_tomos"
    paths = sorted(glob(os.path.join(ROOT, folder, "*.h5")))
    return paths


# TODO refactor into torch_em
class PseudoLabelSampler:
    def __init__(
        self,
        min_foreground_fraction: float,
        foreground_threshold: float = 0.5,
        p_reject: float = 1.0,
    ):
        self.min_foreground_fraction = min_foreground_fraction
        self.foreground_threshold = foreground_threshold
        self.p_reject = p_reject

    def __call__(self, pseudo_labels, label_filter):
        labels = pseudo_labels[label_filter.bool()]
        n_foreground = (labels > self.foreground_threshold).sum()
        foreground_fraction = n_foreground / label_filter.sum()
        if foreground_fraction > self.min_foreground_fraction:
            return True
        else:
            return np.random.rand() > self.p_reject


def run_structure_domain_adaptation(name):
    paths = _get_paths(name)
    train_paths, val_paths = train_test_split(paths, test_size=0.15, random_state=42)
    model_name = f"structure-model-adapt-{name}-v2"
    patch_shape = (64, 512, 512)

    sampler = PseudoLabelSampler(min_foreground_fraction=0.025, p_reject=0.75)
    mean_teacher_adaptation(
        name=model_name,
        unsupervised_train_paths=train_paths,
        unsupervised_val_paths=val_paths,
        patch_shape=patch_shape,
        save_root=".",
        source_checkpoint="./checkpoints/inner_ear_structure_model",
        confidence_threshold=0.75,
        sampler=sampler,
    )


def main():
    # run_structure_domain_adaptation("rat")
    run_structure_domain_adaptation("vesicle_pools")

    # for name in NAMES:
    #     run_structure_domain_adaptation(name)


if __name__ == "__main__":
    main()
