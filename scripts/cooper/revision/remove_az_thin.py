'''import h5py

files = [
    "/mnt/ceph-hdd/cold/nim00007/new_AZ_train_data/stem_cropped2_rescaled/36859_H2_SP_02_rec_2Kb1dawbp_crop_crop1.h5",
    "/mnt/ceph-hdd/cold/nim00007/new_AZ_train_data/stem_cropped2_rescaled/36859_H2_SP_07_rec_2Kb1dawbp_crop_crop1.h5"
]

for file in files:
    with h5py.File(file, "r+") as f:
        # Load the replacement data
        gt = f["labels/az"][:]

        # Delete the existing dataset if it exists
        if "labels/az_thin" in f:
            del f["labels/az_thin"]

        # Recreate the dataset with the new data
        f.create_dataset("labels/az_thin", data=gt)
'''
'''import os
import h5py
from glob import glob
import numpy as np

# Collect all file paths
file_paths1 = glob(os.path.join("/mnt/ceph-hdd/cold/nim00007/new_AZ_train_data/chemical_fixation", "*.h5"))
file_paths2 = glob(os.path.join("/mnt/ceph-hdd/cold/nim00007/new_AZ_train_data/stem", "*.h5"))
file_paths3 = glob(os.path.join("/mnt/ceph-hdd/cold/nim00007/new_AZ_train_data/stem_cropped", "*.h5"))
file_paths4 = glob(os.path.join("/mnt/ceph-hdd/cold/nim00007/new_AZ_train_data/tem", "*.h5"))

all_file_paths = file_paths1 + file_paths2 + file_paths3 + file_paths4

for fname in all_file_paths:
    with h5py.File(fname, "a") as f:
        if "/labels/az_merged_v6" in f:
            az_merged = f["/labels/az_merged_v6"][:]  # shape (1, 46, 446, 446)
            az_merged = np.squeeze(az_merged)         # shape (46, 446, 446)

            del f["/labels/az_merged_v6"]             # delete old dataset

            f.create_dataset("/labels/az_merged_v6", data=az_merged, compression="lzf")
            print(f"Updated file: {fname}")
        else:
            print(f"Dataset not found in: {fname}")'''

import os
import h5py

# List of target folders
base_path = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/ground_truth/04Dataset_for_vesicle_eval"
folders = [
    "20241019_Tomo-eval_MF_Synapse",
    "20241019_Tomo-eval_PS_Synapse",
    "20241019_Tomo-eval_SC_Synapse"
]

# Keys to delete
keys_to_delete = ["/predictions/az/seg_v7", "/predictions/az/v7", "/predictions/az", "/predictions"]

for folder in folders:
    folder_path = os.path.join(base_path, folder)
    for filename in os.listdir(folder_path):
        if filename.endswith(".h5"):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing: {file_path}")
            with h5py.File(file_path, 'a') as h5file:
                for key in keys_to_delete:
                    if key in h5file:
                        print(f"  Deleting key: {key}")
                        del h5file[key]
                    else:
                        print(f"  Key not found: {key}")
