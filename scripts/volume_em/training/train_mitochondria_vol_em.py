"""Train SynapseNet's volume EM mitochondria model.

This reproduces the published model, which was trained on the FIB-SEM datasets 4007 and 4009 at a
voxel size of 25 nm in z and 5 nm in xy. The split is pinned in 'split-mito_vol_em_aniso2lvl_final.json',
which also carries the data roots, so the script runs unchanged after the data has been moved.

The published run stopped at iteration 14,500 of 50,000 (epoch 115, best metric 0.2927) because the
job ran out of wall time, not because it converged. Pass '--resume' to continue it.

To deviate from the recipe, use the CLI instead, which exposes all hyperparameters:
synapse_net.run_vol_em_mitochondria_training -h
"""

import argparse
import os

from synapse_net.training.mitochondria_vol_em import get_vol_em_mitochondria_paths, vol_em_mitochondria_training

TRAIN_ROOTS = {
    "4007": "/mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/4007_hdf5/all_cutouts_s0",
    "4009": "/mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/4009_hdf5/cutouts_segmented",
}
OUTPUT_ROOT = "/mnt/lustre-grete/usr/u15205/volume-em/models"
SPLIT_FILE = os.path.join(os.path.dirname(__file__), "split-mito_vol_em_aniso2lvl_final.json")

PATCH_SHAPE = (32, 512, 512)
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
N_ITERATIONS = 50000
EARLY_STOPPING = 20


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-n", "--name", default="volume-em-mito-aniso2lvl-lr1e-4-bs4-ps32x512x512-blacked-final",
                        help="The name of the model to be trained.")
    parser.add_argument("-i", "--train_root", nargs="+", help="Override the data roots of the split file.")
    parser.add_argument("-o", "--output_root", default=OUTPUT_ROOT, help="Where to save the checkpoint and log dir.")
    parser.add_argument("--random_split", action="store_true",
                        help="Split the data randomly instead of using the pinned split.")
    parser.add_argument("--resume", action="store_true", help="Continue a previous run with the same name.")
    parser.add_argument("--check", action="store_true",
                        help="Visualize samples from the data loaders instead of running training.")
    args = parser.parse_args()

    train_roots = TRAIN_ROOTS if args.train_root is None else args.train_root
    train_paths, val_paths = get_vol_em_mitochondria_paths(
        train_roots, split_file=None if args.random_split else SPLIT_FILE,
    )
    print("Training on", len(train_paths), "blocks and validating on", len(val_paths), "blocks.")

    vol_em_mitochondria_training(
        name=args.name,
        train_paths=train_paths,
        val_paths=val_paths,
        save_root=args.output_root,
        patch_shape=PATCH_SHAPE,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        n_iterations=N_ITERATIONS,
        early_stopping=EARLY_STOPPING,
        resume=args.resume,
        check=args.check,
    )


if __name__ == "__main__":
    main()
