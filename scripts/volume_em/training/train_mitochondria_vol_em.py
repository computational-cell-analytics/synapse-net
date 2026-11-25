"""Train the mitochondria model for FIB-SEM volume EM, reproducing 'mitochondria_fibsem'.

The training data are blocks of the FIB-SEM datasets 4007 and 4009 at 25 nm in z and 5 nm in xy. The split file
next to this script holds the split of the published run and the data roots. The training needs an 80 GB GPU.
Evaluate the model with 'scripts/cooper/training/evaluate_test_set.py -t fibsem'.
"""

import argparse
import os

from synapse_net.training.mitochondria_vol_em import get_vol_em_mitochondria_paths, vol_em_mitochondria_training

SPLIT_FILE = os.path.join(os.path.dirname(__file__), "split-mito_vol_em_aniso2lvl_final.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output_root", required=True, help="The folder for the checkpoint and the logs.")
    parser.add_argument("-n", "--name", default="volume-em-mito-aniso2lvl-lr1e-4-bs4-ps32x512x512-blacked-final")
    parser.add_argument("--resume", action="store_true", help="Continue the previous run with this name.")
    parser.add_argument("--check", action="store_true", help="Check the dataloaders instead of training.")
    args = parser.parse_args()

    train_paths, val_paths = get_vol_em_mitochondria_paths(split_file=SPLIT_FILE)
    print("Training on", len(train_paths), "blocks and validating on", len(val_paths))
    vol_em_mitochondria_training(
        args.name, train_paths, val_paths, save_root=args.output_root, resume=args.resume, check=args.check
    )


if __name__ == "__main__":
    main()
