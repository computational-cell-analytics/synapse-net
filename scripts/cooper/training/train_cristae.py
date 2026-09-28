"""Train the cristae model for electron tomography, reproducing 'cristae5'.

The training data are the cooper and wichmann tomograms at about 1.74 nm, with cristae annotations and the
mitochondria state. The split file next to this script holds the split of the published run and its data roots.
The membrane weighting raised the fraction of mitochondria with a detected cristae junction from 4.2% to 54.2%.
The training needs an 80 GB GPU. The published run stopped at iteration 39,406 of 100,000 after running into the
job time limit twice; '--resume' continues a run. Evaluate the model with 'evaluate_test_set.py -t cristae'.
"""

import argparse
import json
import os

from synapse_net.training.cristae import cristae_training, get_cristae_paths, get_cristae_test_paths

NAME = "cristae-net32-lr1.7e-4-bs24-ps32x256x256-membw-pos3-neg2-band12-2026-07-31"
SPLIT_FILE = os.path.join(os.path.dirname(__file__), "split-cristae_membw_pos3neg2_2026-07-31.json")
# Left out of a random split: the raw data of the first looks wrong, the others have poor cristae annotations.
EXCLUDE = ["Otof_AVCN03_429C_WT_M.Stim_G3_1_model_combined", "WT20_eb8_AZ1_model_combined", "WT22_eb8_model_combined"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output_root", required=True, help="The folder for the checkpoint and the logs.")
    parser.add_argument("-n", "--name", default=NAME)
    parser.add_argument("--random_split", action="store_true", help="Split randomly instead of the published split.")
    parser.add_argument("--resume", action="store_true", help="Continue the previous run with this name.")
    parser.add_argument("--check", action="store_true", help="Check the dataloaders instead of training.")
    args = parser.parse_args()

    if args.random_split:  # Also leave out the test data of the published split.
        with open(SPLIT_FILE) as f:
            roots = json.load(f)["roots"]
        exclude = EXCLUDE + get_cristae_test_paths(SPLIT_FILE)
        train_paths, val_paths = get_cristae_paths(roots, exclude=exclude)
    else:
        train_paths, val_paths = get_cristae_paths(split_file=SPLIT_FILE)
    print("Training on", len(train_paths), "tomograms and validating on", len(val_paths))
    cristae_training(
        args.name, train_paths, val_paths, save_root=args.output_root, resume=args.resume, check=args.check
    )


if __name__ == "__main__":
    main()
