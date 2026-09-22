"""Compare volume EM mitochondria models on the held-out test blocks.

Segments every test block with every model, using identical preprocessing and identical
post-processing, and reports instance and semantic metrics per block. Holding the post-processing
fixed matters: tuning it per model would fold a post-processing difference into what is meant to be
a comparison of the trained networks.

The segmentations are cached, so the script can be re-run to add a model without redoing the others.

Metrics:
  f1 / precision / recall  instance matching at an IoU of 0.5, the metric the volume EM pipeline
                           has always reported (`elf.evaluation.matching`)
  msa                      mean segmentation accuracy, averaged over IoU thresholds 0.5 to 0.95
  sbd                      symmetric best dice
  dice                     semantic dice of the foreground, independent of the instance splitting
  n_pred / n_true          instance counts, which show over- or under-segmentation directly
  val_metric               the best validation DiceLoss of the run, read from its checkpoint

Usage:
    python compare_models.py -m name=/path/to/checkpoint [-m ...] -o RESULTS.md
"""

import argparse
import importlib.util
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch
import torch_em
from elf.evaluation import dice_score, matching, mean_segmentation_accuracy, symmetric_best_dice_score

from synapse_net.inference.util import parse_tiling

TEST_SPLIT = "/mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/test_split"
SEGMENTATION_ROOT = "/mnt/lustre-grete/usr/u15205/volume-em/repro-comparison"

_INFERENCE_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "inference", "run_mitochondria_vol_em_segmentation.py",
)


def _load_inference_module():
    """Load the segmentation script, so the comparison uses exactly the shipped inference path."""
    spec = importlib.util.spec_from_file_location("run_mitochondria_vol_em_segmentation", _INFERENCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_test_blocks(test_split: str):
    """Find the held-out test blocks, as a mapping of dataset name to filepath."""
    blocks = {}
    for dataset in sorted(os.listdir(test_split)):
        folder = os.path.join(test_split, dataset)
        if not os.path.isdir(folder):
            continue
        files = sorted(name for name in os.listdir(folder) if name.endswith(".h5"))
        if len(files) != 1:
            raise ValueError(f"Expect exactly one test block in {folder}, found {len(files)}.")
        blocks[dataset] = os.path.join(folder, files[0])
    if not blocks:
        raise ValueError(f"Did not find any test blocks in {test_split}.")
    return blocks


def segment_block(inference, model_path, block_path, output_path, tiling, force=False):
    """Segment one test block, reusing a cached result unless `force` is set."""
    if os.path.exists(output_path) and not force:
        with h5py.File(output_path, "r") as f:
            if "seg" in f:
                return
    with h5py.File(block_path, "r") as f:
        raw = f["raw"][:]

    segmentation = inference.segment_mitochondria_vol_em(
        raw,
        white_patch_min_size=20,
        model_path=model_path,
        tiling=tiling,
        preprocess=torch_em.transform.raw.normalize_percentile,
        min_size=inference.MIN_SIZE,
        seed_distance=inference.SEED_DISTANCE,
        boundary_threshold=inference.BOUNDARY_THRESHOLD,
        area_threshold=inference.AREA_THRESHOLD,
        verbose=False,
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, "a") as f:
        if "seg" in f:
            del f["seg"]
        f.create_dataset("seg", data=segmentation, compression="gzip")


def evaluate_block(segmentation_path, block_path):
    """Compute the instance and semantic metrics for one segmentation."""
    with h5py.File(segmentation_path, "r") as f:
        seg = f["seg"][:]
    with h5py.File(block_path, "r") as f:
        gt = f["labels/mitochondria"][:]

    stats = matching(segmentation=seg, groundtruth=gt, threshold=0.5, criterion="iou", ignore_label=0)
    return {
        "f1": stats["f1"],
        "precision": stats["precision"],
        "recall": stats["recall"],
        "msa": mean_segmentation_accuracy(seg, gt),
        "sbd": symmetric_best_dice_score(seg, gt),
        "dice": dice_score(seg, gt),
        "n_pred": int(len(np.unique(seg)) - (1 if 0 in seg else 0)),
        "n_true": int(len(np.unique(gt)) - (1 if 0 in gt else 0)),
    }


def read_val_metric(model_path):
    """Read the best validation metric of a run out of its checkpoint."""
    checkpoint = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
    best = os.path.join(checkpoint, "best.pt")
    if not os.path.exists(best):
        return None, None
    try:
        state = torch.load(best, map_location="cpu", weights_only=False)
    except AttributeError:
        # An unrepaired checkpoint of the published model, see repair_checkpoint.py.
        return None, None
    return state.get("best_metric"), state.get("iteration")


def _markdown_table(frame: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    """Render a dataframe as a markdown table, without depending on tabulate."""
    def render(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "-"
        return float_format.format(value) if isinstance(value, float) else str(value)

    header = list(frame.columns)
    rows = [[render(value) for value in row] for row in frame.itertuples(index=False)]
    widths = [max(len(header[i]), *(len(row[i]) for row in rows)) if rows else len(header[i])
              for i in range(len(header))]
    lines = ["| " + " | ".join(name.ljust(width) for name, width in zip(header, widths)) + " |",
             "|" + "|".join("-" * (width + 2) for width in widths) + "|"]
    lines += ["| " + " | ".join(value.ljust(width) for value, width in zip(row, widths)) + " |" for row in rows]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-m", "--model", action="append", required=True, metavar="NAME=PATH",
                        help="A model to compare, as 'name=checkpoint folder'. Repeat for each model.")
    parser.add_argument("--test_split", default=TEST_SPLIT, help="The folder with the held-out test blocks.")
    parser.add_argument("--segmentation_root", default=SEGMENTATION_ROOT,
                        help="Where the segmentations are cached.")
    parser.add_argument("-o", "--output", help="Write the report to this markdown file.")
    parser.add_argument("--tile_shape", type=int, nargs=3, help="The tile shape for prediction, in ZYX.")
    parser.add_argument("--halo", type=int, nargs=3, help="The halo for prediction, in ZYX.")
    parser.add_argument("--force", action="store_true", help="Recompute segmentations that are already cached.")
    args = parser.parse_args()

    models = {}
    for entry in args.model:
        if "=" not in entry:
            parser.error(f"Expect '--model name=path', got {entry!r}.")
        name, path = entry.split("=", 1)
        if not os.path.exists(path):
            parser.error(f"The checkpoint of {name!r} does not exist: {path}")
        models[name] = path

    inference = _load_inference_module()
    tiling = parse_tiling(args.tile_shape, args.halo)
    blocks = find_test_blocks(args.test_split)
    print(f"Comparing {len(models)} model(s) on {len(blocks)} test block(s): {', '.join(blocks)}\n")

    rows = []
    for name, model_path in models.items():
        val_metric, iteration = read_val_metric(model_path)
        for dataset, block_path in blocks.items():
            output_path = os.path.join(args.segmentation_root, name, f"{dataset}.h5")
            print(f"  {name} / {dataset} ...", flush=True)
            segment_block(inference, model_path, block_path, output_path, tiling, force=args.force)
            metrics = evaluate_block(output_path, block_path)
            metrics.update(model=name, dataset=dataset, val_metric=val_metric, iteration=iteration)
            rows.append(metrics)

    results = pd.DataFrame(rows)
    columns = ["model", "dataset", "f1", "precision", "recall", "msa", "sbd", "dice", "n_pred", "n_true"]
    per_block = results[columns].sort_values(["dataset", "model"])

    mean_columns = ["f1", "precision", "recall", "msa", "sbd", "dice"]
    averaged = results.groupby("model")[mean_columns].mean().reset_index()
    val = results.groupby("model")[["val_metric", "iteration"]].first().reset_index()
    averaged = averaged.merge(val, on="model")

    report = [
        "# Volume EM mitochondria: reproduction vs. the published model",
        "",
        "Instance metrics are matching at an IoU of 0.5; `msa` averages over the thresholds 0.5 to 0.95.",
        "`dice` is the semantic foreground dice. `val_metric` is the best validation DiceLoss of the run,",
        "read from its checkpoint, where lower is better.",
        "",
        "All models were segmented with identical preprocessing and identical post-processing",
        f"(seed_distance {inference.SEED_DISTANCE}, boundary_threshold {inference.BOUNDARY_THRESHOLD}, "
        f"area_threshold {inference.AREA_THRESHOLD}, min_size {inference.MIN_SIZE}).",
        "",
        "## Averaged over the test blocks",
        "",
        _markdown_table(averaged),
        "",
        "## Per test block",
        "",
        _markdown_table(per_block),
        "",
    ]
    report = "\n".join(report)
    print("\n" + report)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
