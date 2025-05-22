import os
import glob
import h5py
import argparse
import sys
sys.path.append('/user/muth9/u12095/synapse-net')
from synapse_net.ground_truth.shape_refinement import edge_filter

def postprocess_file(file_path):
    """Processes a single .h5 file, applying an edge filter and saving the membrane mask."""#
    print(f"processing file {file_path}")
    with h5py.File(file_path, "a") as f:
        raw = f["raw"][:]
        print("applying the edge filter ...")
        hmap = edge_filter(raw, sigma=1.0, method="sato", per_slice=True, n_threads=8)
        membrane_mask = hmap > 0.5
        print("saving results ....")
        try:
            f.create_dataset("labels/membrane_mask", data=membrane_mask, compression="gzip")
        except:
            print(f"membrane mask aleady saved for {file_path}")    
    print("Done!")

def postprocess_folder(folder_path):
    """Processes all .h5 files in a given folder recursively."""
    files = sorted(glob.glob(os.path.join(folder_path, '**', '*.h5'), recursive=True))
    print("Processing files:", files)
    
    for file_path in files:
        postprocess_file(file_path)

def main():
    #/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/exported_imod_objects/01_hoi_maus_2020_incomplete 
    #/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/exported_imod_objects/04_hoi_stem_examples 
    #/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/exported_imod_objects/06_hoi_wt_stem750_fm 
    #/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/exported_imod_objects/12_chemical_fix_cryopreparation 
    
    parser = argparse.ArgumentParser(description="Postprocess .h5 files by applying edge filtering.")
    parser.add_argument("-p", "--data_path", required=True, help="Path to the .h5 file or folder.")
    args = parser.parse_args()
    
    if os.path.isdir(args.data_path):
        postprocess_folder(args.data_path)
    else:
        postprocess_file(args.data_path)

if __name__ == "__main__":
    main()
