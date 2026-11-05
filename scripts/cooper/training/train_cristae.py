"""Train the cristae model for electron tomography.

This reproduces the training of SynapseNet's 'cristae5' model. The training data consists of the
cooper and wichmann tomograms at a resolution of roughly 1.74 nm, with cristae annotations and a
semantic mitochondria state that marks which mitochondria carry those annotations.

The split file next to this script holds the exact train / val split of the published run. Pass
'--random_split' to split the data randomly instead; the three volumes in EXCLUDE are then left out,
because their raw data or their annotations are unreliable, and so are the volumes that the split
file holds out for testing, which the run would otherwise train on.

Note that the published checkpoint did not converge: it stopped at iteration 39,406 of 100,000 after
running into the job time limit twice. Rerun this script with '--resume' to continue it; N_ITERATIONS
is the total number of iterations, so it will train the remaining ones.
"""

import argparse
import os

from synapse_net.training.cristae import cristae_training, get_cristae_paths, get_cristae_test_paths

TRAIN_ROOTS = {
    "cooper_s2": "/scratch-grete/projects/nim00007/data/mitochondria/cooper/raw_mito_combined_s2",
    "cooper_cristae": "/mnt/lustre-grete/usr/u12103/mitochondria/cooper/cristae",
    "wichmann": "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann",
    "wichmann_needs_corrections": "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann_needs_corrections",
}
OUTPUT_ROOT = "/mnt/lustre-grete/usr/u12103/cristae"
SPLIT_FILE = os.path.join(os.path.dirname(__file__), "split-cristae_membw_pos3neg2_2026-07-31.json")

EXCLUDE = [
    "Otof_AVCN03_429C_WT_M.Stim_G3_1_model_combined",  # The raw data of this volume looks wrong.
    "WT20_eb8_AZ1_model_combined",  # Poor cristae annotations.
    "WT22_eb8_model_combined",  # Poor cristae annotations.
]

PATCH_SHAPE = (32, 256, 256)
BATCH_SIZE = 24
LEARNING_RATE = 1.7e-4
N_ITERATIONS = 100000
EARLY_STOPPING = 25

# The membrane proximity weighting. This was validated both on the average precision and directly on
# the detection of cristae junctions, where it raised the fraction of mitochondria with a detected
# junction from 4.2% to 54.2% within 8 nm of the membrane.
MEMBRANE_W_POS = 3.0
MEMBRANE_W_NEG = 2.0
MEMBRANE_BAND_NM = 12.0


def main():
    parser = argparse.ArgumentParser(description="Train the cristae model for electron tomography.")
    parser.add_argument("-n", "--name", default="cristae-net32-lr1.7e-4-bs24-ps32x256x256-membw-pos3-neg2-band12-2026-07-31",  # noqa
                        help="The name of the model to be trained.")
    parser.add_argument("-o", "--output_root", default=OUTPUT_ROOT, help="The folder for the checkpoint and logs.")
    parser.add_argument("--random_split", action="store_true",
                        help="Split the data randomly instead of using the split of the published run.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue the previous run with the same name, restoring its optimizer and "
                             "iteration count.")
    parser.add_argument("--check", action="store_true", help="Check the dataloaders instead of running training.")
    args = parser.parse_args()

    exclude = EXCLUDE if not args.random_split else EXCLUDE + get_cristae_test_paths(SPLIT_FILE, TRAIN_ROOTS)
    train_paths, val_paths = get_cristae_paths(
        TRAIN_ROOTS, split_file=None if args.random_split else SPLIT_FILE, exclude=exclude,
    )
    print("Training on", len(train_paths), "tomograms and validating on", len(val_paths), "tomograms.")

    cristae_training(
        name=args.name,
        train_paths=train_paths,
        val_paths=val_paths,
        save_root=args.output_root,
        patch_shape=PATCH_SHAPE,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        n_iterations=N_ITERATIONS,
        early_stopping=EARLY_STOPPING,
        membrane_w_pos=MEMBRANE_W_POS,
        membrane_w_neg=MEMBRANE_W_NEG,
        membrane_band_nm=MEMBRANE_BAND_NM,
        resume=args.resume,
        check=args.check,
    )


if __name__ == "__main__":
    main()
