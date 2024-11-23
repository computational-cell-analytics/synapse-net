import argparse
import os

import h5py
import pandas as pd
import numpy as np

from elf.evaluation.dice import dice_score

def extract_gt_bounding_box(segmentation, gt, halo=[20, 320, 320]):
    # Find the bounding box for the ground truth
    bb = np.where(gt > 0)
    bb = tuple(slice(
        max(int(b.min() - ha), 0),  # Ensure indices are not below 0
        min(int(b.max() + ha), sh) # Ensure indices do not exceed shape dimensions
    ) for b, sh, ha in zip(bb, gt.shape, halo))
    
    # Apply the bounding box to both segmentations
    segmentation_cropped = segmentation[bb]
    gt_cropped = gt[bb]
    
    return segmentation_cropped, gt_cropped

def evaluate(labels, segmentation):
    assert labels.shape == segmentation.shape
    score = dice_score(segmentation, labels)
    return score

def evaluate_file(labels_path, segmentation_path, model_name, crop= False):
    print(f"Evaluate labels {labels_path} and vesicles {segmentation_path}")

    ds_name = os.path.basename(os.path.dirname(labels_path))
    tomo = os.path.basename(labels_path)

    #get the labels and segmentation
    with h5py.File(labels_path) as label_file:
        gt = label_file["/labels/AZ"][:]
        
    with h5py.File(segmentation_path) as seg_file:
        segmentation = seg_file["/AZ/segment_from_AZmodel_v3"][:]

    if crop:
        print("cropping the annotation and segmentation")
        segmentation, gt = extract_gt_bounding_box(segmentation, gt)

    # Evaluate the match of ground truth and segmentation
    dice_score = evaluate(gt, segmentation)

    # Store results
    result_folder = "/user/muth9/u12095/synaptic-reconstruction/scripts/cooper/evaluation_results"
    os.makedirs(result_folder, exist_ok=True)
    result_path = os.path.join(result_folder, f"evaluation_{model_name}.csv")
    print("Evaluation results are saved to:", result_path)

    # Load existing results if the file exists
    if os.path.exists(result_path):
        results = pd.read_csv(result_path)
    else:
        results = None

    # Create a new DataFrame for the current evaluation
    res = pd.DataFrame(
        [[ds_name, tomo, dice_score]], columns=["dataset", "tomogram", "dice_score"]
    )

    # Combine with existing results or initialize with the new results
    if results is None:
        results = res
    else:
        results = pd.concat([results, res])

    # Save the results to the CSV file
    results.to_csv(result_path, index=False)

def evaluate_folder(labels_path, segmentation_path, model_name, crop = False):
    print(f"Evaluating folder {segmentation_path}")
    print(f"Using labels stored in {labels_path}")

    label_files = os.listdir(labels_path)
    vesicles_files = os.listdir(segmentation_path)
    
    for vesicle_file in vesicles_files:
        if vesicle_file in label_files:

            evaluate_file(os.path.join(labels_path, vesicle_file), os.path.join(segmentation_path, vesicle_file), model_name, crop)



def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--labels_path", required=True)
    parser.add_argument("-v", "--segmentation_path", required=True)
    parser.add_argument("-n", "--model_name", required=True)
    parser.add_argument("--crop", action="store_true", help="Crop around the annotation.")
    args = parser.parse_args()

    segmentation_path = args.segmentation_path
    if os.path.isdir(segmentation_path):
        evaluate_folder(args.labels_path, segmentation_path, args.model_name, args.crop)
    else:
        evaluate_file(args.labels_path, segmentation_path, args.model_name, args.crop)
    
    

if __name__ == "__main__":
    main()
