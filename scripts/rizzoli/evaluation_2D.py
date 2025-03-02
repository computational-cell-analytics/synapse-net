import argparse
import os

import h5py
import pandas as pd
import numpy as np

from elf.evaluation import matching, symmetric_best_dice_score
from skimage.transform import rescale

def transpose_tomo(tomogram):
        data0 = np.swapaxes(tomogram, 0, -1)
        data1 = np.fliplr(data0)
        transposed_data = np.swapaxes(data1, 0, -1)
        return transposed_data

def evaluate(labels, vesicles):
    assert labels.shape == vesicles.shape
    stats = matching(vesicles, labels)
    sbd = symmetric_best_dice_score(vesicles, labels)
    return [stats["f1"], stats["precision"], stats["recall"], sbd]

def evaluate_slices(gt, vesicles):
    """Evaluate 2D model performance for each z-slice of the 3D volume."""
    f1_scores = []
    precision_scores = []
    recall_scores = []
    sbd_scores = []
    
    # Iterate through each slice along the z-axis
    for z in range(gt.shape[0]):
        # Extract the 2D slice from the 3D volume
        gt_slice = gt[z, :, :]
        vesicles_slice = vesicles[z, :, :]
        
        # Evaluate the performance for the current slice
        f1, precision, recall, sbd = evaluate(gt_slice, vesicles_slice)
        
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
        sbd_scores.append(sbd)
    
    print(f"f1 scores to be averaged {f1_scores}")
    print(f"sbd scores to be averaged {sbd_scores}")

    # Calculate the mean for each metric
    mean_f1 = np.mean(f1_scores)
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mead_sbd = np.mean(sbd_scores)
    
    return [mean_f1, mean_precision, mean_recall, mead_sbd]

def summarize_eval(results):
    summary = results[["dataset", "f1-score", "precision", "recall", "SBD score"]].groupby("dataset").mean().reset_index("dataset")
    total = results[["f1-score", "precision", "recall", "SBD score"]].mean().values.tolist()
    summary.iloc[-1] = ["all"] + total
    table = summary.to_markdown(index=False)
    print(table)

def evaluate_file(labels_path, vesicles_path, model_name, segment_key, anno_key):
    print(f"Evaluate labels {labels_path} and vesicles {vesicles_path}")

    ds_name = os.path.basename(os.path.dirname(labels_path))
    tomo = os.path.basename(labels_path)

    use_mask=True #set to true for eg maus data

    #get the labels and vesicles
    with h5py.File(labels_path) as label_file:
        labels = label_file["labels"]
        #vesicles = labels["vesicles"]
        gt = labels[anno_key][:]
        gt = rescale(gt, scale=0.5, order=0, anti_aliasing=False, preserve_range=True).astype(gt.dtype)
        gt = transpose_tomo(gt)

        if use_mask:
            mask = labels["mask"][:]
            mask = rescale(mask, scale=0.5, order=0, anti_aliasing=False, preserve_range=True).astype(mask.dtype)
            mask = transpose_tomo(mask)
        
    with h5py.File(vesicles_path) as seg_file:
        segmentation = seg_file["vesicles"]
        vesicles = segmentation[segment_key][:] 
    
    if use_mask:
        gt[mask == 0] = 0
        vesicles[mask == 0] = 0
    
    #evaluate the match of ground truth and vesicles
    if len(vesicles.shape) == 3:
        scores = evaluate_slices(gt, vesicles)
    else:
        scores = evaluate(gt,vesicles)
    
    #store results
    result_folder ="/user/muth9/u12095/synapse-net/scripts/cooper/evaluation_results"
    os.makedirs(result_folder, exist_ok=True)
    result_path=os.path.join(result_folder, f"2Devaluation_{model_name}.csv")
    print("Evaluation results are saved to:", result_path)
    
    if os.path.exists(result_path):
        results = pd.read_csv(result_path)
    else:
        results = None
        
    res = pd.DataFrame(
        [[ds_name, tomo] + scores], columns=["dataset", "tomogram", "f1-score", "precision", "recall", "SBD score"]
    )
    if results is None:
        results = res
    else:
        results = pd.concat([results, res])
    results.to_csv(result_path, index=False)

    #print results
    summarize_eval(results)


def evaluate_folder(labels_path, vesicles_path, model_name, segment_key, anno_key):
    print(f"Evaluating folder {vesicles_path}")
    print(f"Using labels stored in {labels_path}")

    label_files = os.listdir(labels_path)
    vesicles_files = os.listdir(vesicles_path)
    
    for vesicle_file in vesicles_files:
        if vesicle_file in label_files:

            evaluate_file(os.path.join(labels_path, vesicle_file), os.path.join(vesicles_path, vesicle_file), model_name, segment_key, anno_key)



def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--labels_path", required=True)
    parser.add_argument("-v", "--vesicles_path", required=True)
    parser.add_argument("-n", "--model_name", required=True)
    parser.add_argument("-sk", "--segment_key", required=True)
    parser.add_argument("-ak", "--anno_key", required=True)
    args = parser.parse_args()

    vesicles_path = args.vesicles_path
    if os.path.isdir(vesicles_path):
        evaluate_folder(args.labels_path, vesicles_path, args.model_name, args.segment_key, args.anno_key)
    else:
        evaluate_file(args.labels_path, vesicles_path, args.model_name, args.segment_key, args.anno_key)
    
    

if __name__ == "__main__":
    main()
