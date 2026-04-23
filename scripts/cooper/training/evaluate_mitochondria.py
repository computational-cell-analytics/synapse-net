import argparse
from glob import glob
import os

# import h5py
# from elf.io import open_file
from tifffile import imread
import pandas as pd

from elf.evaluation import matching, symmetric_best_dice_score

from elf.io.files import open_file


def evaluate(labels, vesicles):
    assert labels.shape == vesicles.shape
    stats = matching(vesicles, labels)
    sbd = symmetric_best_dice_score(vesicles, labels)
    return [stats["f1"], stats["precision"], stats["recall"], sbd]


def summarize_eval(results):
    summary = (
        results[["dataset", "f1-score", "precision", "recall", "SBD score"]]
        .groupby("dataset")
        .mean()
        .reset_index("dataset")
    )
    total = results[["f1-score", "precision", "recall", "SBD score"]].mean().values.tolist()
    summary.iloc[-1] = ["all"] + total
    table = summary.to_markdown(index=False)
    print(table)


def evaluate_file(labels_path, seg_path, model_name, segment_key, anno_key, mask_key, output_folder):
    print(f"Evaluate labels \n{labels_path} and segmentations \n{seg_path}")
    labels, seg = None, None
    if ".tif" in labels_path:
        labels = imread(labels_path)
    elif anno_key is not None:
        labels = open_file(labels_path, "r")[anno_key][:]
    if ".tif" in seg_path:
        seg = imread(seg_path)
    elif segment_key is not None:
        seg = open_file(seg_path, "r")[segment_key][:]
    if labels is None or seg is None:
        print("Could not find label file for", seg_path)
        print("Skipping...")

    # evaluate the match of ground truth and vesicles
    scores = evaluate(labels, seg)

    # store results
    result_folder = output_folder
    os.makedirs(result_folder, exist_ok=True)
    result_path = os.path.join(result_folder, f"evaluation_{model_name}.csv")
    print("Evaluation results are saved to:", result_path)

    if os.path.exists(result_path):
        results = pd.read_csv(result_path)
    else:
        results = None
    ds_name = os.path.basename(os.path.dirname(labels_path))
    tomo = os.path.basename(labels_path)
    res = pd.DataFrame(
        [[ds_name, tomo] + scores], columns=["dataset", "tomogram", "f1-score", "precision", "recall", "SBD score"]
    )
    if results is None:
        results = res
    else:
        results = pd.concat([results, res])
    results.to_csv(result_path, index=False)

    # print results
    summarize_eval(results)


def evaluate_folder(labels_path, segmentation_path, model_name, segment_key, anno_key,
                    anno_ext, segment_ext, mask_key, output_folder):
    print(f"Evaluating folder {segmentation_path}")
    print(f"Using labels stored in {labels_path}")
    if labels_path is not None:
        label_paths = get_file_paths(labels_path, ext=anno_ext)
    else:
        label_paths = _get_default_label_paths()
    seg_paths = get_file_paths(segmentation_path, ext=segment_ext)
    if label_paths is None or seg_paths is None:
        print("Could not find label file or segmentation file")
        return

    for seg_path in seg_paths:
        label_path = find_label_file(seg_path, label_paths)
        if label_path is not None:
            evaluate_file(label_path, seg_path, model_name, segment_key, anno_key, mask_key, output_folder)
        else:
            print("Could not find label file for", seg_path)
            print("Skipping...")


def get_file_paths(path, ext=".h5", reverse=False):
    if ext in path:
        return [path]
    else:
        paths = sorted(glob(os.path.join(path, "**", f"*{ext}"), recursive=True), reverse=reverse)
        return paths


def find_label_file(given_path: str, label_paths: list) -> str:
    """
    Find the corresponding label file for a given raw file.

    Args:
        given_path (str): The path we want to find label file to.
        label_paths (list): A list of label file paths.

    Returns:
        str: The path to the matching label file, or None if no match is found.
    """
    raw_base = os.path.splitext(os.path.basename(given_path))[0]  # Remove extension
    raw_base = raw_base.replace("prediction", "").replace("pred", "")
    raw_base = raw_base.replace("segmentation", "").replace("seg", "")
    raw_base = raw_base.replace("mito-v3_sd18_bt015_with__", "")
    raw_base = raw_base.rstrip("_")
    print("raw_base", raw_base)
    for label_path in label_paths:
        label_base = os.path.splitext(os.path.basename(label_path))[0]  # Remove extension
        if raw_base.strip().lower() in label_base.strip().lower():  # Ensure raw name is contained in label name
            return label_path

    return None  # No match found


def _get_default_label_paths():
    return ['/scratch-grete/projects/nim00007/data/mitochondria/wichmann/refined_mitos/M2_eb10_model.h5',
            '/scratch-grete/projects/nim00007/data/mitochondria/wichmann/refined_mitos/WT21_eb3_model2.h5',
            '/scratch-grete/projects/nim00007/data/mitochondria/wichmann/refined_mitos/M10_eb9_model.h5',
            '/scratch-grete/projects/nim00007/data/mitochondria/wichmann/refined_mitos/KO9_eb4_model.h5',
            '/scratch-grete/projects/nim00007/data/mitochondria/wichmann/refined_mitos/M7_eb11_model.h5',
            '/scratch-grete/projects/nim00007/data/mitochondria/cooper/fidi_down_s2/36859_J1_66K_TS_CA3_PS_25_rec_2Kb1dawbp_crop_downscaled.h5'
            ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-sp", "--segmentation_path", required=True)
    parser.add_argument("-gp", "--groundtruth_path", default=None)
    parser.add_argument("-n", "--model_name", required=True)
    parser.add_argument("-sk", "--segmentation_key", default=None)
    parser.add_argument("-gk", "--groundtruth_key", default=None)
    parser.add_argument("-ae", "--annotation_extension", default=None)
    parser.add_argument("-se", "--segmentation_extension", default=None)
    parser.add_argument("-m", "--mask_key", default=None)
    parser.add_argument("-o", "--output_folder", required=True)
    args = parser.parse_args()

    if os.path.isdir(args.segmentation_path):
        evaluate_folder(args.groundtruth_path, args.segmentation_path, args.model_name, args.segmentation_key,
                        args.groundtruth_key, args.annotation_extension, args.segmentation_extension,
                        args.mask_key, args.output_folder)
    else:
        evaluate_file(args.groundtruth_path, args.segmentation_path, args.model_name, args.segmentation_key,
                      args.groundtruth_key,
                      args.mask_key, args.output_folder)


if __name__ == "__main__":
    main()
