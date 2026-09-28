"""Evaluate the mitochondria or cristae model for electron tomography on its held-out test set.

    python evaluate_test_set.py -t mito -m mitochondria2 -o mito.csv
    python evaluate_test_set.py -t cristae -m cristae5 -o cristae.csv

The model is the name of a pretrained model, a checkpoint folder or a serialized model.
Mitochondria are scored by instance matching (F1, precision, recall) and the symmetric best dice.
Cristae are scored voxel-wise in the mitochondria with cristae annotations (dice, precision, recall), and by the
average precision and ROC-AUC of the foreground prediction there, also in bands at the membrane and in the rest.

The segmentation changes with the tiling, whose default depends on the GPU memory. The published evaluations
used '--tile 32 256 256 --halo 4 32 32' for mitochondria2 and '--tile 32 256 256 --halo 8 64 64' for cristae5.
"""

import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch_em
from elf.evaluation import matching, symmetric_best_dice_score
from elf.io import open_file
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import average_precision_score, roc_auc_score

from synapse_net.inference.cristae import segment_cristae
from synapse_net.inference.inference import get_model
from synapse_net.inference.mitochondria import segment_mitochondria
from synapse_net.training.cristae import get_cristae_test_paths
from synapse_net.training.transform import CRISTAE_VOXEL_SIZE

MITO_TEST_ROOT = "/mnt/lustre-grete/usr/u12103/mitochondria/synapse-net-eval-data/eval_data_h5_s4"
CRISTAE_SPLIT_FILE = os.path.join(os.path.dirname(__file__), "split-cristae_membw_pos3neg2_2026-07-31.json")
BAND_NM = (8.0, 12.0)  # The thicknesses of the membrane bands for the average precision.
MAX_POINTS_AP = 3_000_000  # The average precision is computed on a random subset of at most this many voxels.
EPS = 1e-8


def score_mito(path, seg):
    with open_file(path, "r") as f:
        labels = f["labels/mitochondria"][:]
    stats = matching(seg, labels)
    return {"f1": stats["f1"], "precision": stats["precision"], "recall": stats["recall"],
            "sbd": symmetric_best_dice_score(seg, labels)}


def _average_precision(labels, pred, region):
    labels, pred = labels[region] > 0, pred[region]
    if labels.all() or not labels.any():
        return np.nan, np.nan
    if labels.size > MAX_POINTS_AP:
        index = np.random.default_rng(0).choice(labels.size, size=MAX_POINTS_AP, replace=False)
        labels, pred = labels[index], pred[index]
    return float(average_precision_score(labels, pred)), float(roc_auc_score(labels, pred))


def score_cristae(path, seg, pred):
    with open_file(path, "r") as f:
        labels, state = f["labels/cristae"][:], f["raw_mitos_combined"][1]
    mito = state == 1  # Only the mitochondria with cristae annotations are scored.
    gt, seg = labels.astype(bool) & mito, seg.astype(bool) & mito
    tp, fp, fn = (int(np.logical_and(a, b).sum()) for a, b in ((seg, gt), (seg, ~gt), (~seg, gt)))
    scores = {"dice": 2 * tp / (2 * tp + fp + fn + EPS),
              "precision": tp / (tp + fp + EPS), "recall": tp / (tp + fn + EPS)}

    # The membrane is approximated by the surface of the annotated mitochondria.
    distance = distance_transform_edt(mito, sampling=CRISTAE_VOXEL_SIZE).astype(np.float32)
    regions = {"all": mito}
    for thickness in BAND_NM:
        band = mito & (distance <= thickness)
        regions[f"band{thickness:g}"], regions[f"core{thickness:g}"] = band, mito & ~band
    for name, region in regions.items():
        scores[f"ap_{name}"], scores[f"auc_{name}"] = _average_precision(labels, pred, region)
    return scores


def evaluate(task, path, model, tiling):
    with open_file(path, "r") as f:
        if task == "mito":
            # The percentile normalization of the training recipe is part of the model.
            seg = segment_mitochondria(f["raw"][:], model=model, tiling=tiling, verbose=False,
                                       preprocess=torch_em.transform.raw.normalize_percentile)
            return score_mito(path, seg)
        # The tomogram and the mitochondria state, which segment_cristae turns into the mitochondria mask.
        seg, pred = segment_cristae(f["raw_mitos_combined"][:], voxel_size=np.mean(CRISTAE_VOXEL_SIZE), model=model,
                                    tiling=tiling, verbose=False, return_predictions=True)
    return score_cristae(path, seg, np.asarray(pred[0], dtype=np.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-t", "--task", choices=["mito", "cristae"], required=True)
    parser.add_argument("-m", "--model", required=True, help="A pretrained model, a checkpoint folder or a .pt file.")
    parser.add_argument("-o", "--output_path", required=True, help="The csv file for the scores.")
    parser.add_argument("--tile", type=int, nargs=3, help="The tile shape (z, y, x), including the halo.")
    parser.add_argument("--halo", type=int, nargs=3, help="The halo (z, y, x).")
    args = parser.parse_args()
    if (args.tile is None) != (args.halo is None):
        parser.error("Pass both '--tile' and '--halo', or neither.")
    tiling = None if args.tile is None else {"tile": dict(zip("zyx", args.tile)), "halo": dict(zip("zyx", args.halo))}

    if os.path.isdir(args.model):
        model = torch_em.util.load_model(checkpoint=args.model)
    elif os.path.isfile(args.model):
        model = torch.load(args.model, map_location="cpu", weights_only=False)
    else:
        model = get_model(args.model)
    model.eval()

    paths = sorted(glob(os.path.join(MITO_TEST_ROOT, "*.h5"))) if args.task == "mito" else \
        get_cristae_test_paths(CRISTAE_SPLIT_FILE)
    results = []
    for path in paths:
        results.append({"tomogram": os.path.basename(path), **evaluate(args.task, path, model, tiling)})
        print(results[-1], flush=True)
    results = pd.DataFrame(results)
    results.loc[len(results)] = ["mean", *results.iloc[:, 1:].mean()]
    results.to_csv(args.output_path, index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
