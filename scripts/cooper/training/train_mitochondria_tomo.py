"""Train the mitochondria model for electron tomography.

This reproduces the training of SynapseNet's 'mitochondria2' model. The training data consists of
the fidi (downscaled by a factor of 4) and wichmann (downscaled by a factor of 2) tomograms, which
are both at a resolution of roughly 2.87 nm, with refined mitochondria annotations.

The split file next to this script holds the exact train / val split of the published run.
Delete the '--split_file' argument below to split the data randomly instead.
"""

import argparse
import os

from synapse_net.training.mitochondria import (
    _resolve_resume_checkpoint, get_mitochondria_paths, mitochondria_training
)

TRAIN_ROOT = "/mnt/lustre-grete/usr/u12103/mitochondria/mito-tomo-all"
OUTPUT_ROOT = "/mnt/lustre-grete/usr/u12103/mitochondria/tomo"
SPLIT_FILE = os.path.join(os.path.dirname(__file__), "split-mito_tomo_s4_refined.json")

PATCH_SHAPE = (32, 256, 256)
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
N_ITERATIONS = 150000
EARLY_STOPPING = 20


def main():
    parser = argparse.ArgumentParser(description="Train the mitochondria model for electron tomography.")
    parser.add_argument("-n", "--name", default="mitotomo-net32-lr1e-4-bs8-ps32x256x256-s4-refined-final",
                        help="The name of the model to be trained.")
    parser.add_argument("-i", "--train_root", default=TRAIN_ROOT, help="The folder with the training data.")
    parser.add_argument("-o", "--output_root", default=OUTPUT_ROOT, help="The folder for the checkpoint and logs.")
    parser.add_argument("--random_split", action="store_true",
                        help="Split the data randomly instead of using the split of the published run.")
    parser.add_argument("--resume", action="store_true",
                        help="Initialize the model from the best checkpoint of a previous run with the same name.")
    parser.add_argument("--check", action="store_true", help="Check the dataloaders instead of running training.")
    args = parser.parse_args()

    train_paths, val_paths = get_mitochondria_paths(
        args.train_root, split_file=None if args.random_split else SPLIT_FILE,
    )
    print("Training on", len(train_paths), "tomograms and validating on", len(val_paths), "tomograms.")

    checkpoint_path = None
    if args.resume:
        checkpoint_path = _resolve_resume_checkpoint(args.output_root, args.name, None)

    mitochondria_training(
        name=args.name,
        train_paths=train_paths,
        val_paths=val_paths,
        save_root=args.output_root,
        patch_shape=PATCH_SHAPE,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        n_iterations=N_ITERATIONS,
        early_stopping=EARLY_STOPPING,
        checkpoint_path=checkpoint_path,
        check=args.check,
    )


if __name__ == "__main__":
    main()
