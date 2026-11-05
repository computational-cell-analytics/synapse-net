"""Score a voxel-size sweep of a baseline method and select the best setting per approach.

The sweep is made by segment_mitonet.py or segment_microsam.py on the validation blocks of the pinned
split, one folder per setting and mode; a sweep root holds the folders of one method. The settings
fall into three groups:
  A  z stays at the native 25 nm and only xy is resampled; the xy stack only, since MitoNet's
     ortho-plane consensus needs isotropic voxels and micro-sam has no ortho-plane mode
  B  isotropic resampling of all axes; for MitoNet the ortho-plane consensus and the xy stack
  C  any other single setting, such as 30 nm in z and 8 nm in xy, the voxel size MitoNet_v1 was
     probably trained at
25 nm isotropic belongs to both A and B. A setting may be run on the raw and on the inverted contrast.

Every folder is scored with compare_models.evaluate_block against the native ground truth, so the
numbers are the ones compare_models.py reports. The metrics are cached in a 'metrics.json' per
folder, so extending the grid only scores the new settings.

The best setting of an approach is the one with the highest F1 averaged over the validation blocks,
ties broken by msa. It is selected here, on the validation blocks, and then run once on the test
split; nothing is selected on the test blocks.

Usage:
    python score_voxel_size_sweep.py --sweep_root ROOT [-o RESULTS.md] [--workers 8]
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_models  # noqa: E402

SPLIT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "training", "split-mito_vol_em_aniso2lvl_final.json",
)
NATIVE_Z = 25.0

# Below this, no setting is presented as working: the best of a set of failures is not a result.
F1_FLOOR = 0.05

METRICS = ["f1", "precision", "recall", "msa", "sbd", "dice"]


def read_setting(folder, datasets):
    """Read the setting of a sweep folder from the attributes of its segmentations."""
    paths = {dataset: os.path.join(folder, f"{dataset}.h5") for dataset in datasets}
    missing = [path for path in paths.values() if not os.path.exists(path)]
    if missing:
        raise ValueError(f"Missing segmentations in {folder}: {', '.join(missing)}")
    with h5py.File(next(iter(paths.values())), "r") as f:
        attrs = f["seg"].attrs
        z, y, x = (float(v) for v in attrs["target_voxel_size_nm"])
        setting = {
            "folder": os.path.basename(folder),
            "model": str(attrs["model"]),
            "voxel_size": (z, y, x),
            "contrast": "inverted" if bool(attrs["invert"]) else "raw",
            "mode": "ortho" if str(attrs["mode"]).startswith("ortho") else "xy",
        }
    approaches = []
    if z == NATIVE_Z and y == x and setting["mode"] == "xy":
        approaches.append("A")
    if z == y == x:
        approaches.append("B")
    setting["approaches"] = approaches or ["C"]
    return setting, paths


def voxel_label(approach, voxel_size):
    """The voxel size a table row is keyed by: xy for A, the isotropic size for B, all axes for C."""
    z, y, x = voxel_size
    if approach == "A":
        return y
    if approach == "B":
        return z
    return "/".join(f"{v:g}" for v in voxel_size)


def _score(task):
    folder, dataset, segmentation_path, block_path = task
    metrics = compare_models.evaluate_block(segmentation_path, block_path)
    # Plain Python numbers, so the metrics can be cached as json.
    metrics = {key: int(value) if key.startswith("n_") else float(value) for key, value in metrics.items()}
    return folder, dataset, metrics


def score_sweep(folders, blocks, workers):
    """Score every folder on every block, reusing the cached metrics of unchanged segmentations."""
    cached, tasks = {}, []
    for folder, paths in folders.items():
        cache_path = os.path.join(folder, "metrics.json")
        cache = {}
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cache = json.load(f)
        cached[folder] = {}
        for dataset, path in paths.items():
            entry = cache.get(dataset)
            if entry is not None and entry.get("seg_mtime") == os.path.getmtime(path):
                cached[folder][dataset] = entry
            else:
                tasks.append((folder, dataset, path, blocks[dataset]))

    print(f"Scoring {len(tasks)} segmentation(s), {sum(map(len, cached.values()))} cached", flush=True)
    with Pool(workers) as pool:
        for folder, dataset, metrics in pool.imap_unordered(_score, tasks):
            metrics["seg_mtime"] = os.path.getmtime(folders[folder][dataset])
            cached[folder][dataset] = metrics
            print(f"  {os.path.basename(folder)} / {dataset}: f1 {metrics['f1']:.4f}", flush=True)
            with open(os.path.join(folder, "metrics.json"), "w") as f:
                json.dump(cached[folder], f, indent=2)
    return cached


def select_best(frame):
    """The best row of an approach: highest mean F1, ties broken by msa."""
    return frame.sort_values(["f1", "msa"], ascending=False).iloc[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--sweep_root", required=True,
                        help="The folder with one folder per setting and mode, of one method.")
    parser.add_argument("--split_file", default=SPLIT_FILE, help="The split file the sweep was run on.")
    parser.add_argument("--split_key", default="val", help="The key of the split file the sweep was run on.")
    parser.add_argument("-o", "--output", help="Write the report to this markdown file.")
    parser.add_argument("--best", help="Write the selected settings to this json file. "
                                       "Default: best.json in the sweep root.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel scorings; each takes ~6-8 GB at the native resolution.")
    args = parser.parse_args()

    blocks = compare_models.find_split_blocks(args.split_file, args.split_key)
    settings, folders = [], {}
    for name in sorted(os.listdir(args.sweep_root)):
        folder = os.path.join(args.sweep_root, name)
        if not os.path.isdir(folder):
            continue
        setting, paths = read_setting(folder, blocks)
        settings.append(setting)
        folders[folder] = paths
    if not settings:
        raise ValueError(f"Did not find any sweep folders in {args.sweep_root}.")
    models = sorted({setting["model"] for setting in settings})
    if len(models) != 1:
        raise ValueError(f"A sweep root must hold one model, found {', '.join(models)} in {args.sweep_root}.")
    model = models[0]

    metrics = score_sweep(folders, blocks, args.workers)

    per_block, averaged = [], []
    for setting in settings:
        folder_metrics = metrics[os.path.join(args.sweep_root, setting["folder"])]
        rows = [{"setting": setting["folder"], "dataset": dataset,
                 **{key: folder_metrics[dataset][key] for key in METRICS + ["n_pred", "n_true"]}}
                for dataset in blocks]
        per_block += rows
        mean = {key: float(np.mean([row[key] for row in rows])) for key in METRICS}
        counts = {key: int(sum(row[key] for row in rows)) for key in ("n_pred", "n_true")}
        for approach in setting["approaches"]:
            averaged.append({"approach": approach, "voxel_nm": voxel_label(approach, setting["voxel_size"]),
                             "contrast": setting["contrast"], "mode": setting["mode"],
                             "setting": setting["folder"], "voxel_size": setting["voxel_size"],
                             **mean, **counts})
    averaged = pd.DataFrame(averaged)
    per_block = pd.DataFrame(per_block).sort_values(["setting", "dataset"])

    table_columns = ["voxel_nm", "contrast", "mode", *METRICS, "n_pred", "n_true"]
    report = [
        f"# Volume EM mitochondria: {model} voxel-size sweep on the validation blocks",
        "",
        f"{model} on the `{args.split_key}` blocks of `{os.path.basename(args.split_file)}` "
        f"({', '.join(blocks)}), which it never saw.",
        "Each block is resampled to the voxel size, segmented, and resized back to the native",
        "25/5/5 nm grid, where it is scored against the untouched ground truth. `min_size` is scaled",
        "to the same physical volume as synapse-net's 1,000 native voxels at every voxel size.",
        "",
        "Instance metrics are matching at an IoU of 0.5; `msa` averages over the thresholds 0.5 to 0.95.",
        "Metrics are averaged over the blocks, `n_pred` and `n_true` are summed.",
        "",
        "The best setting of each approach is selected here, by F1 with ties broken by msa, and then run",
        "once on the test split. Nothing is selected on the test blocks.",
        "",
    ]
    best = {}
    for approach, title in (("A", "A: z at 25 nm, xy resampled (xy stack)"),
                            ("B", "B: isotropic"),
                            ("C", "C: single settings (voxel size z/y/x in nm)")):
        frame = averaged[averaged["approach"] == approach].sort_values(["contrast", "mode", "voxel_nm"])
        if frame.empty:
            continue
        row = select_best(frame)
        works = row["f1"] >= F1_FLOOR
        # C is a set of single settings, not a grid, so it has no edge to extend.
        grid = sorted(frame["voxel_nm"].unique()) if approach != "C" else []
        on_edge = works and len(grid) > 1 and row["voxel_nm"] in (grid[0], grid[-1])
        best[approach] = {
            "setting": row["setting"], "voxel_size": list(row["voxel_size"]), "contrast": row["contrast"],
            "mode": row["mode"], "f1": row["f1"], "msa": row["msa"], "works": bool(works),
            "on_grid_edge": bool(on_edge),
            "segment_args": " ".join(
                ["--voxel_size", *(f"{v:g}" for v in row["voxel_size"]), "--modes", row["mode"]]
                + (["--invert"] if row["contrast"] == "inverted" else [])
            ),
        }
        if works:
            verdict = (f"**Selected: `{row['setting']}`** — {row['contrast']} contrast, {row['mode']} mode, "
                       f"F1 {row['f1']:.4f}, msa {row['msa']:.4f}.")
            if on_edge:
                verdict += (f" It is on the edge of the grid ({grid[0]:g}–{grid[-1]:g} nm): extend the grid "
                            "by a step before relying on it.")
        else:
            verdict = (f"**No setting works:** the best is `{row['setting']}` at F1 {row['f1']:.4f}, "
                       f"below {F1_FLOOR}, so none is selected.")
        table = frame[table_columns].assign(voxel_nm=frame["voxel_nm"].map(
            lambda v: v if isinstance(v, str) else f"{v:g}"))
        report += [f"## {title}", "", verdict, "", compare_models._markdown_table(table), ""]
    report += ["## Per block", "", compare_models._markdown_table(per_block), ""]

    report = "\n".join(report)
    print("\n" + report)
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Wrote {args.output}")
    best_path = args.best or os.path.join(args.sweep_root, "best.json")
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"Wrote {best_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
