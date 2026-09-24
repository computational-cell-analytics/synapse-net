"""Make a volume EM mitochondria checkpoint loadable again.

torch-em pickles the whole training dataset into a checkpoint, including its raw transform. The
published volume EM mitochondria model was trained by a script that defined that transform as a
local function, so the checkpoint references `__main__._raw_transform_with_fix_white_patches` and
cannot be loaded anywhere else:

    AttributeError: Can't get attribute '_raw_transform_with_fix_white_patches' on <module '__main__'>

This rewrites the reference to `synapse_net.training.transform.RemoveWhitePatchesAndNormalize`, which
is the same transform as an importable class, and writes the result to a new folder. The weights, the
optimizer state and every hyperparameter are copied through unchanged; only the dangling reference is
replaced. The original is never modified.

Models trained with `synapse_net.training.mitochondria_vol_em` do not need this: their raw transform
is a module-level class, so their checkpoints load as they are.

Usage:
    python repair_checkpoint.py -i <checkpoint dir> -o <output dir>
"""

import argparse
import os
import shutil
import sys

import torch

from synapse_net.training.transform import RemoveWhitePatchesAndNormalize


def repair_checkpoint(input_dir: str, output_dir: str, min_size: int = 20) -> None:
    """Rewrite the dangling raw transform reference of a checkpoint.

    Args:
        input_dir: The checkpoint folder to repair. It is not modified.
        output_dir: The folder to write the repaired checkpoint to.
        min_size: The minimal size of a filler component, for the transform that replaces the
            dangling reference. The published model used the default of 20.

    Raises:
        FileNotFoundError: If `input_dir` holds no checkpoint.
    """
    checkpoints = [name for name in ("best.pt", "latest.pt") if os.path.exists(os.path.join(input_dir, name))]
    if not checkpoints:
        raise FileNotFoundError(f"Did not find best.pt or latest.pt in {input_dir}.")

    # Give the unpickler something to bind the dangling reference to. The published transform removed
    # the white filler and then normalized with the 1st and 99th percentile, which is exactly what
    # 'RemoveWhitePatchesAndNormalize' does; this was checked numerically in compare_training_setup.py.
    replacement = RemoveWhitePatchesAndNormalize(min_size=min_size)
    sys.modules["__main__"]._raw_transform_with_fix_white_patches = replacement

    os.makedirs(output_dir, exist_ok=True)
    for name in checkpoints:
        state = torch.load(os.path.join(input_dir, name), map_location="cpu", weights_only=False)
        n_patched = 0
        for key in ("train_dataset", "val_dataset"):
            dataset = state["init"].get(key)
            if dataset is None:
                continue
            for sub_dataset in getattr(dataset, "datasets", [dataset]):
                sub_dataset.raw_transform = RemoveWhitePatchesAndNormalize(min_size=min_size)
                n_patched += 1
        torch.save(state, os.path.join(output_dir, name))
        print(f"{name}: iteration {state['iteration']}, best metric {state['best_metric']:.6f}, "
              f"{n_patched} dataset(s) patched")

    for name in os.listdir(input_dir):
        if not name.endswith(".pt"):
            shutil.copy2(os.path.join(input_dir, name), output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-i", "--input", required=True, help="The checkpoint folder to repair.")
    parser.add_argument("-o", "--output", required=True, help="Where to write the repaired checkpoint.")
    parser.add_argument("--white_patch_min_size", type=int, default=20,
                        help="The minimal size of a filler component for the replacement transform.")
    args = parser.parse_args()

    repair_checkpoint(args.input, args.output, min_size=args.white_patch_min_size)

    # Prove that the result is loadable without any stand-in.
    import subprocess
    check = subprocess.run(
        [sys.executable, "-c",
         f"import torch_em; m = torch_em.util.load_model(checkpoint={args.output!r}, device='cpu'); "
         "print('the repaired checkpoint loads:', type(m).__name__)"],
        capture_output=True, text=True,
    )
    print(check.stdout.strip() or check.stderr.strip())
    return check.returncode


if __name__ == "__main__":
    sys.exit(main())
