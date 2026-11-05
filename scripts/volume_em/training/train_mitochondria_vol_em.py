"""Train SynapseNet's volume EM mitochondria model.

This reproduces the published model, which was trained on the FIB-SEM datasets 4007 and 4009 at a
voxel size of 25 nm in z and 5 nm in xy. The split is pinned in 'split-mito_vol_em_aniso2lvl_final.json',
which also carries the data roots, so the script runs unchanged after the data has been moved.

How the published checkpoint came about is worth knowing, because its counters do not mean what they
look like. Two jobs for it were submitted 24 minutes apart and ran concurrently for about 23 hours,
both writing the same checkpoint folder. The second one picked up the first one's 'best.pt' as it
stood after 15 epochs and warm-started from it, weights only, so its optimizer, scheduler and
counters all restarted. Both then stopped on early stopping, neither on the 50,000 iterations or on
the wall clock. The surviving 'best.pt' is the second job's: 14,500 of its own iterations on top of
the 2,000 it inherited, about 16,500 gradient steps in total, with an optimizer reset in the middle.
The first job's remaining 14,125 steps, including its own best of 0.296378, were overwritten.

That is an accident, not a recipe, and it gained little: 0.292714 against the 0.296378 the first job
had reached on its own, and 0.298842 for the sibling 'finalv2' run. This script therefore trains a
single clean run, which is what the recipe describes. Note that running it twice with the same name
and save_root is now refused outright rather than silently racing.

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
