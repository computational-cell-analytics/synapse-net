"""Train the mitochondria model for electron tomography, reproducing 'mitochondria2'.

The training data are the fidi (downscaled by 4) and wichmann (downscaled by 2) tomograms with refined
annotations, at about 2.87 nm. The split file next to this script holds the split of the published run.
The training needs an 80 GB GPU. Evaluate the model with 'evaluate_test_set.py -t mito'.
"""

import argparse
import os

from synapse_net.training.mitochondria import get_mitochondria_paths, mitochondria_training

TRAIN_ROOT = "/mnt/lustre-grete/usr/u12103/mitochondria/mito-tomo-all"
SPLIT_FILE = os.path.join(os.path.dirname(__file__), "split-mito_tomo_s4_refined.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output_root", required=True, help="The folder for the checkpoint and the logs.")
    parser.add_argument("-n", "--name", default="mitotomo-net32-lr1e-4-bs8-ps32x256x256-s4-refined-final")
    parser.add_argument("--random_split", action="store_true", help="Split randomly instead of the published split.")
    parser.add_argument("--resume", action="store_true", help="Continue the previous run with this name.")
    parser.add_argument("--check", action="store_true", help="Check the dataloaders instead of training.")
    args = parser.parse_args()

    train_paths, val_paths = get_mitochondria_paths(TRAIN_ROOT, split_file=None if args.random_split else SPLIT_FILE)
    print("Training on", len(train_paths), "tomograms and validating on", len(val_paths))
    mitochondria_training(
        args.name, train_paths, val_paths, save_root=args.output_root, resume=args.resume, check=args.check
    )


if __name__ == "__main__":
    main()
