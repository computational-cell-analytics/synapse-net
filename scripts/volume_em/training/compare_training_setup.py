"""Check that the ported volume EM mitochondria training builds the same trainer as the published model.

This constructs the trainer through `synapse_net.training.mitochondria_vol_em.vol_em_mitochondria_training`,
stops it just before it would start training, and diffs its serialized setup against the setup stored
in the published checkpoint. It is the cheap part of reproducing that model: if this passes, the only
thing a training run can still tell us is whether the optimization lands in the same place.

Three differences are expected and are checked rather than ignored:

- The raw transform. The published run pickled a function that was local to its training script, which
  is why its checkpoint cannot be loaded without a stand-in. The port uses
  `synapse_net.training.transform.RemoveWhitePatchesAndNormalize`. They are compared numerically, on
  real patches drawn from the training data.
- The joint transform. The port wraps the augmentations in `PadIfNecessary`, which the published run
  did not. This is checked to be a no-op for this data, i.e. that no block is smaller than the patch.
- 'name', 'id_', 'save_root', 'device' and 'rank', which differ by construction.

Usage:
    python compare_training_setup.py [-c <checkpoint dir>] [--split_file <json>]
Exits non-zero if anything else differs.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import torch
import torch_em
from skimage.measure import label as connected_components

from synapse_net.training.mitochondria_vol_em import get_vol_em_mitochondria_paths, vol_em_mitochondria_training
from synapse_net.training.transform import RemoveWhitePatchesAndNormalize

REFERENCE_CHECKPOINT = (
    "/mnt/lustre-grete/usr/u15205/volume-em/models/checkpoints/"
    "volume-em-mito-aniso2lvl-lr1e-4-bs4-ps32x512x512-blacked-final"
)
SPLIT_FILE = os.path.join(os.path.dirname(__file__), "split-mito_vol_em_aniso2lvl_final.json")

# These differ by construction and carry no information about the recipe.
IGNORED_FIELDS = ("name", "id_", "save_root", "device", "rank", "logger_kwargs")
# Fields that a newer torch-em adds to the trainer, together with why they cannot change this recipe.
# A field that a newer torch-em adds and that is not listed here is reported as a difference, because
# it may well change what is trained.
TOLERATED_ADDITIONS = {
    "mixed_precision_dtype":
        "only read by the mixed-precision train and validate loops, and this recipe sets "
        "mixed_precision to False",
}
# These hold live objects and are compared structurally further down instead.
STRUCTURAL_FIELDS = ("train_dataset", "val_dataset")


def reference_raw_transform(x):
    """The raw transform of the published run, transcribed from 5aa482e:training/train_mito_generic.py.

    It is reimplemented here rather than imported, because importing it from the other repository pulls
    in napari, mrcfile and z5py. Keeping it standalone is also what makes this an independent check.
    """
    img = x
    if img.dtype != np.uint8:
        if not (np.issubdtype(img.dtype, np.floating) and img.min() >= 0 and img.max() <= 255):
            warnings.warn("img must be uint8, converting to uint8 from " + str(img.dtype))
        img = img.astype(np.uint8)
    if img.ndim not in (2, 3):
        raise ValueError(f"img must be a 2D or 3D array, but got {img.ndim}D")

    white_mask = img == 255
    labeled = connected_components(white_mask)
    if labeled.max() == 0:
        out = img
    else:
        sizes = np.bincount(labeled.ravel())
        large_labels = np.where(sizes >= 20)[0]
        large_labels = large_labels[large_labels != 0]
        if large_labels.size == 0:
            out = img
        else:
            out = img.copy()
            out[np.isin(labeled, large_labels)] = 0
    return torch_em.transform.raw.normalize_percentile(out)


def load_reference_init(checkpoint):
    """Read the setup out of the published checkpoint."""
    # The checkpoint pickled a raw transform that lived in '__main__' of the training script, so
    # unpickling it needs a stand-in under that name.
    sys.modules["__main__"]._raw_transform_with_fix_white_patches = reference_raw_transform
    state = torch.load(os.path.join(checkpoint, "best.pt"), map_location="cpu", weights_only=False)
    return state["init"]


def build_our_init(train_paths, val_paths):
    """Build the trainer through the ported recipe and serialize it, without training."""
    captured = {}
    build_trainer = torch_em.default_segmentation_trainer

    def capture(**kwargs):
        trainer = build_trainer(**kwargs)
        captured["trainer"] = trainer
        trainer.fit = lambda *args, **kwargs: None
        return trainer

    torch_em.default_segmentation_trainer = capture
    try:
        vol_em_mitochondria_training(
            name="compare-training-setup", train_paths=train_paths, val_paths=val_paths,
            save_root="/tmp/compare-training-setup",
        )
    finally:
        torch_em.default_segmentation_trainer = build_trainer

    trainer = captured["trainer"]
    # These two lines are what 'DefaultTrainer._initialize' does before it serializes the setup.
    trainer._model_class = f"{type(trainer.model).__module__}.{type(trainer.model).__name__}"
    trainer._model_kwargs = torch_em.util.get_constructor_arguments(trainer.model)
    return trainer._build_init()


def _describe_dataset(dataset):
    """Reduce a ConcatDataset to the properties that define the training data."""
    subsets = dataset.datasets
    first = subsets[0]
    return {
        "n_datasets": len(subsets),
        "n_samples": len(dataset),
        "lengths": [len(ds) for ds in subsets],
        "files": sorted(os.path.basename(ds.raw_path) for ds in subsets),
        "patch_shape": list(first.patch_shape),
        "raw_key": first.raw_key,
        "label_key": first.label_key,
        "ndim": first.ndim,
        "dtype": str(first.dtype),
        "label_dtype": str(first.label_dtype),
        "sampler": (type(first.sampler).__name__, vars(first.sampler)),
        "label_transform": type(first.label_transform).__name__,
        "label_transform2": type(first.label_transform2).__name__,
        "augmentations": [type(aug).__name__ for aug in _augmentations_of(first.transform)],
    }


def _augmentations_of(transform):
    """Get the kornia augmentations of a joint transform, looking through a Compose."""
    if hasattr(transform, "augmentations"):
        return transform.augmentations
    # The port wraps the augmentations in a Compose together with PadIfNecessary.
    for step in getattr(transform, "transforms", []):
        if hasattr(step, "augmentations"):
            return step.augmentations
    return []


def _pads_of(transform):
    """Get the PadIfNecessary steps of a joint transform, which the published run did not have."""
    return [step for step in getattr(transform, "transforms", []) if type(step).__name__ == "PadIfNecessary"]


def compare_fields(reference, ours):
    """Compare the flat fields of the two setups.

    Returns the unexpected differences, and separately the fields that only the version of torch-em
    running here produces, which are tolerated only if they are known to be inert for this recipe.
    """
    differences, additions = [], []
    fields = sorted(set(reference) | set(ours))
    print(f"{'field':<24} {'match':<7} value")
    print("-" * 100)
    for field in fields:
        if field in IGNORED_FIELDS or field in STRUCTURAL_FIELDS:
            continue

        # A field that the published checkpoint does not have at all was added by a newer torch-em,
        # rather than being a difference in how the recipe is built.
        if field not in reference:
            if field in TOLERATED_ADDITIONS:
                additions.append((field, ours[field], TOLERATED_ADDITIONS[field]))
                print(f"{field:<24} {'added':<7} {ours[field]} (not in the published setup)")
            else:
                differences.append((field, "<not in the published setup>", ours[field]))
                print(f"{field:<24} {'NEW':<7} {ours[field]} (not in the published setup)")
            continue

        ref_value, our_value = reference[field], ours.get(field, "<missing>")
        match = ref_value == our_value
        if not match:
            differences.append((field, ref_value, our_value))
        print(f"{field:<24} {'ok' if match else 'DIFFERS':<7} {ref_value}")
        if not match:
            print(f"{'':<24} {'':<7} ours: {our_value}")
    return differences, additions


def compare_datasets(reference, ours):
    """Compare the training and validation data of the two setups."""
    differences = []
    for field in STRUCTURAL_FIELDS:
        ref_desc = _describe_dataset(reference[field])
        our_desc = _describe_dataset(ours[field])
        print(f"\n{field}")
        print("-" * 100)
        for key in ref_desc:
            match = ref_desc[key] == our_desc[key]
            if not match:
                differences.append((f"{field}.{key}", ref_desc[key], our_desc[key]))
            shown = ref_desc[key] if key != "files" else f"{len(ref_desc[key])} files"
            print(f"  {key:<20} {'ok' if match else 'DIFFERS':<7} {shown}")
            if not match:
                print(f"  {'':<20} {'':<7} ours: {our_desc[key]}")
    return differences


def check_raw_transform(train_paths, n_patches=24, patch_shape=(32, 512, 512), seed=0):
    """Check that the ported raw transform agrees with the published one, on real data."""
    import h5py

    rng = np.random.default_rng(seed)
    ours = RemoveWhitePatchesAndNormalize()
    max_difference, n_with_filler = 0.0, 0

    for i in range(n_patches):
        path = train_paths[i % len(train_paths)]
        with h5py.File(path, "r") as f:
            raw = f["raw"]
            # Draw patches from the volume border half of the time, where the filler lives.
            offsets = [
                0 if (i % 2 == 0 and axis > 0) else int(rng.integers(0, max(size - extent, 1)))
                for axis, (size, extent) in enumerate(zip(raw.shape, patch_shape))
            ]
            bb = tuple(slice(off, off + extent) for off, extent in zip(offsets, patch_shape))
            patch = raw[bb]

        if (patch == 255).sum() >= 20:
            n_with_filler += 1
        difference = np.abs(ours(patch) - reference_raw_transform(patch)).max()
        max_difference = max(max_difference, float(difference))

    print(f"\nraw transform: {n_patches} real patches, {n_with_filler} of them containing filler")
    print(f"  maximal absolute difference to the published transform: {max_difference:g}")
    return max_difference, n_with_filler


def check_padding_is_a_noop(ours, patch_shape=(32, 512, 512)):
    """Check that the PadIfNecessary the port adds never fires, so the joint transforms agree."""
    import h5py

    too_small = []
    for dataset in list(ours["train_dataset"].datasets) + list(ours["val_dataset"].datasets):
        with h5py.File(dataset.raw_path, "r") as f:
            shape = f[dataset.raw_key].shape
        if any(size < extent for size, extent in zip(shape, patch_shape)):
            too_small.append((os.path.basename(dataset.raw_path), shape))
    n_pads = len(_pads_of(ours["train_dataset"].datasets[0].transform))
    print(f"\njoint transform: the port adds {n_pads} PadIfNecessary step(s)")
    print(f"  blocks smaller than {tuple(patch_shape)} in any axis: {len(too_small)}")
    for name, shape in too_small:
        print(f"    {name} {shape}")
    return too_small


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-c", "--checkpoint", default=REFERENCE_CHECKPOINT,
                        help="The checkpoint folder of the published model.")
    parser.add_argument("--split_file", default=SPLIT_FILE, help="The split file of the published run.")
    args = parser.parse_args()

    train_paths, val_paths = get_vol_em_mitochondria_paths(split_file=args.split_file)
    print(f"Comparing against {args.checkpoint}")
    print(f"{len(train_paths)} training and {len(val_paths)} validation blocks from {args.split_file}\n")

    reference = load_reference_init(args.checkpoint)
    ours = build_our_init(train_paths, val_paths)

    differences, additions = compare_fields(reference, ours)
    differences += compare_datasets(reference, ours)

    max_difference, n_with_filler = check_raw_transform(train_paths)
    too_small = check_padding_is_a_noop(ours)

    print("\n" + "=" * 100)
    failed = False
    if differences:
        failed = True
        print(f"{len(differences)} unexpected difference(s) in the training setup:")
        for field, ref_value, our_value in differences:
            print(f"  {field}: published {ref_value!r} vs ours {our_value!r}")
    else:
        print("The training setup matches the published model in every compared field.")

    for field, value, reason in additions:
        print(f"This torch-em adds '{field}' = {value!r}, which the published run did not have. "
              f"It is inert here: {reason}.")

    if max_difference > 1e-6:
        failed = True
        print(f"The raw transform does not agree with the published one ({max_difference:g} > 1e-6).")
    elif n_with_filler == 0:
        failed = True
        print("No patch contained filler, so the raw transform check did not exercise the filler removal.")
    else:
        print("The raw transform agrees with the published one on real data, including on filler.")

    if too_small:
        failed = True
        print(f"{len(too_small)} block(s) are smaller than the patch, so the added padding is not a no-op.")
    else:
        print("The padding the port adds to the joint transform never fires for this data.")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
