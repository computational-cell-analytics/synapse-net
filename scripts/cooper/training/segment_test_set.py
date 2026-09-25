"""Segment the held-out test set of the mitochondria or cristae model for electron tomography.

The segmentations are written to one hdf5 file per test volume, with the same name as the volume,
so that they can be scored with 'evaluate_mitochondria.py' or 'evaluate_cristae.py':

    python segment_test_set.py -t mito -m mitochondria2 -o seg_mito
    python evaluate_mitochondria.py -sp seg_mito -n mitochondria2 -o eval

    python segment_test_set.py -t cristae -m cristae5 -o seg_cristae --save_predictions
    python evaluate_cristae.py -sp seg_cristae -n cristae5 -o eval

The model is either the name of a pretrained model, the checkpoint folder of a training run, or a
serialized model. The test volumes are already at the training resolution, so they are not rescaled.

The default tiling depends on the memory of the GPU, and the segmentation changes with the tiling.
Pass '--tile' and '--halo' to get the same results on every GPU. The published evaluations used
'--tile 32 256 256 --halo 4 32 32' for mitochondria2 and '--tile 32 256 256 --halo 8 64 64' for cristae5.
"""

import argparse
import os
from glob import glob

import h5py
import numpy as np
import torch
import torch_em
from elf.io import open_file

from synapse_net.inference.cristae import segment_cristae
from synapse_net.inference.inference import get_model
from synapse_net.inference.mitochondria import segment_mitochondria
from synapse_net.training.cristae import get_cristae_test_paths
from synapse_net.training.transform import CRISTAE_VOXEL_SIZE

from evaluate_mitochondria import TEST_ROOT as MITO_TEST_ROOT
from train_cristae import SPLIT_FILE, TRAIN_ROOTS


def load_model(model):
    if os.path.isdir(model):
        return torch_em.util.load_model(checkpoint=model)
    if os.path.isfile(model):
        return torch.load(model, map_location="cpu", weights_only=False)
    return get_model(model)


def segment_volume(task, path, model, tiling=None, return_predictions=False):
    kwargs = dict(model=model, tiling=tiling, verbose=False, return_predictions=return_predictions)
    with open_file(path, "r") as f:
        if task == "mito":
            raw = f["raw"][:]
            # The percentile normalization of the training recipe is part of the model.
            return segment_mitochondria(raw, preprocess=torch_em.transform.raw.normalize_percentile, **kwargs)
        # The tomogram and the mitochondria state, which segment_cristae turns into the mitochondria mask.
        raw = f["raw_mitos_combined"][:]
        return segment_cristae(raw, voxel_size=np.mean(CRISTAE_VOXEL_SIZE), **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Segment the test set of the mitochondria or cristae model.")
    parser.add_argument("-t", "--task", choices=["mito", "cristae"], required=True)
    parser.add_argument("-m", "--model", required=True,
                        help="The name of a pretrained model, a checkpoint folder or a serialized model.")
    parser.add_argument("-o", "--output_folder", required=True)
    parser.add_argument("-k", "--key", default="seg", help="The key for the segmentation in the output files.")
    parser.add_argument("--tile", type=int, nargs=3, default=None,
                        help="The tile shape (z, y, x), including the halo. By default it depends on the GPU memory, "
                             "so pass it together with '--halo' to get the same results on every GPU.")
    parser.add_argument("--halo", type=int, nargs=3, default=None, help="The halo (z, y, x) of the tiles.")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Also save the foreground and boundary predictions, in 'pred/foreground' and "
                             "'pred/boundary'. evaluate_cristae.py needs them for the average precision.")
    args = parser.parse_args()
    if (args.tile is None) != (args.halo is None):
        parser.error("Pass both '--tile' and '--halo', or neither.")
    tiling = None if args.tile is None else {
        "tile": dict(zip("zyx", args.tile)), "halo": dict(zip("zyx", args.halo))
    }

    if args.task == "mito":
        paths = sorted(glob(os.path.join(MITO_TEST_ROOT, "*.h5")))
    else:
        paths = get_cristae_test_paths(SPLIT_FILE, TRAIN_ROOTS)
    print("Segmenting", len(paths), "test volumes with", args.model)

    model = load_model(args.model)
    model.eval()
    os.makedirs(args.output_folder, exist_ok=True)
    for path in paths:
        output_path = os.path.join(args.output_folder, os.path.basename(path))
        if os.path.exists(output_path):
            print("Skipping", output_path, "because it exists already")
            continue
        seg = segment_volume(args.task, path, model, tiling=tiling, return_predictions=args.save_predictions)
        if args.save_predictions:
            seg, pred = seg
        with h5py.File(output_path, "w") as f:
            f.create_dataset(args.key, data=seg.astype("uint32"), compression="gzip")
            if args.save_predictions:
                f.create_dataset("pred/foreground", data=pred[0].astype("float32"), compression="gzip")
                f.create_dataset("pred/boundary", data=pred[1].astype("float32"), compression="gzip")
        print(os.path.basename(path), ":", len(np.unique(seg)) - 1, "objects")


if __name__ == "__main__":
    main()
