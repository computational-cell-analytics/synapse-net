import numpy as np

from synapse_net.file_utils import read_mrc
import mrcfile
import h5py
import tifffile
from pathlib import Path
import re

def apply_ignore_label(h5_path, mask_path, ignore_label: int=-1):
    """For supervised training: set masked voxels to -1 (ignore_label)."""
    with h5py.File(h5_path, "r") as f:
        raw = f["raw"][:]
        labels = f["labels/actin"][:]

    with mrcfile.open(mask_path, permissive=True) as mrc:
        ignore_mask = mrc.data.astype(bool)
        #ignore_mask = np.flip(ignore_mask, axis=1)
    
    labels_masked = labels.astype(np.int32) # ensure signed int type 
    labels_masked[(labels == 0) & ignore_mask] = ignore_label 

    out_dir = Path(h5_path).parent / "ignore_label"
    out_dir.mkdir(parents=True, exist_ok=True)
    fstem = Path(h5_path).stem
    out_path = out_dir / f"{fstem}.h5"

    print(f"Writing out h5 file with masked labels to {out_path}.")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("/labels/actin", data=labels_masked, compression="gzip")
    
def convert_tiff2mrc(in_dir, pixel_size, out_dir=None): 
    """Batch convert tiff files to mrc."""
    in_dir = Path(in_dir)

    if out_dir == None:
        out_dir = in_dir
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    path_list = [str(p) for p in in_dir.glob("*.tif")]
    
    for path in path_list:
        data = tifffile.imread(path)
        data = np.flip(data, axis=1)
        filename = Path(path).stem
        out_path = out_dir / f"{filename}.mrc"
        
        print(f"Writing out mrc file to {out_path}.")
        with mrcfile.new(out_path, overwrite=True) as mrc:
            mrc.set_data(data.astype(np.uint8))
            mrc.voxel_size = (pixel_size, pixel_size, pixel_size)

def h5_split_tomograms(h5_path, z_range):
    """
    Split paired raw and label data (z,y,x) into 8 non-overlapping subvolumes
    by cutting it in half along each axis.
    """
    with h5py.File(h5_path, "r") as f:
        z0, z1 = z_range
        raw = f["raw"][z0:z1, :, :]
        labels = f["labels/actin"][z0:z1, :, :]

    z, y, x = raw.shape

    # Compute midpoints
    z_mid, y_mid, x_mid = z // 2, y // 2, x // 2

    # Define ranges for each half
    z_ranges = [(0, z_mid), (z_mid, z)]
    y_ranges = [(0, y_mid), (y_mid, y)]
    x_ranges = [(0, x_mid), (x_mid, x)]

    raw_subvols, label_subvols = [], []

    for zi, (z0, z1) in enumerate(z_ranges):
        for yi, (y0, y1) in enumerate(y_ranges):
            for xi, (x0, x1) in enumerate(x_ranges):
                raw_subvol = raw[z0:z1, y0:y1, x0:x1]
                label_subvol = labels[z0:z1, y0:y1, x0:x1]
                raw_subvols.append(raw_subvol)
                label_subvols.append(label_subvol)

    return raw_subvols, label_subvols

def write_h5(raw_path, label_path, out_path):
    """Write the raw and labels to an HDF5 file."""
    if out_path.exists():
        print(f"File {out_path} already exists, skipping.")
        return
    
    raw = read_mrc(raw_path)[0]
    labels = read_mrc(label_path)[0]

    print(f"Writing file to {out_path}.")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("/labels/actin", data=labels, compression="gzip")

def write_h5_deepict():
    PARENT_DIR = Path("/mnt/data1/sage/actin-segmentation/data/deepict/deepict_actin/ignore_label")
    TRAIN_DIR = PARENT_DIR / "train"
    VAL_DIR = PARENT_DIR / "val"
    TEST_DIR = PARENT_DIR / "test"

    TRAIN_DIR.mkdir(exist_ok=True)
    VAL_DIR.mkdir(exist_ok=True)
    TEST_DIR.mkdir(exist_ok=True)

    raw_subvols1, label_subvols1 = h5_split_tomograms(
        Path(PARENT_DIR / "00004_cleaned.h5"), z_range = (326, 464)
        )
    raw_subvols2, label_subvols2 = h5_split_tomograms(
        Path(PARENT_DIR / "00012_cleaned.h5"), z_range = (147, 349)
        )

    raw_subvols = raw_subvols1 + raw_subvols2
    label_subvols = label_subvols1 + label_subvols2

    # predefined indices for train, val, test (10:2:4)
    train_idx = [0, 3, 4, 7, 8, 9, 10, 11, 12, 15]
    val_idx = [6, 14]
    test_idx = [1, 2, 5, 13]

    def write_split(idx_list, folder, prefix):
        for idx in idx_list:
            raw = raw_subvols[idx]
            labels = label_subvols[idx]

            # tomogram 00004: indices 0-7 -> A
            # tomogram 00012: indices 8-15 -> B
            if idx < 8:
                tag = f"A{idx}"
            else:
                tag = f"B{idx - 8}" 
            out_path = folder / f"{prefix}_{tag}.h5"

            print(f"Writing file to {out_path}.")
            with h5py.File(out_path, "w") as f:
                f.create_dataset("raw", data=raw, compression="gzip")
                f.create_dataset("/labels/actin", data=labels, compression="gzip")

    write_split(train_idx, TRAIN_DIR, "train")
    write_split(val_idx, VAL_DIR, "val")
    write_split(test_idx, TEST_DIR, "test")
    print("\n Finished writing all subvolumes.")

def write_h5_optogenetics():
    RAW_DIR = Path("/mnt/data1/sage/actin-segmentation/data/EMPIAR-12292/tomos/")
    LABEL_DIR = Path("/mnt/data1/sage/actin-segmentation/data/EMPIAR-12292/labels/")
    OUT_DIR = Path("/mnt/data1/sage/actin-segmentation/data/EMPIAR-12292/h5/")

    raw_paths = {re.sub('_rec', '', f.stem): f for f in RAW_DIR.glob("*_rec.mrc")}
    label_paths = {re.sub('_mask', '', f.stem): f for f in LABEL_DIR.glob("*_mask.mrc")}

    stems = raw_paths.keys() | label_paths.keys()

    for stem in stems:
        if stem not in raw_paths:
            print(f"Warning: Missing tomo file for {stem}.")
            continue

        if stem not in label_paths:
            print(f"Warning: Missing label file for {stem}.")
            continue

        raw_path = raw_paths[stem]
        label_path = label_paths[stem]
        out_path = OUT_DIR / f"{stem}.h5"
        write_h5(raw_path, label_path, out_path)

def main():
    #write_h5_optogenetics()
    #write_h5_deepict()
    #convert_tiff2mrc(
    #    input_dir = "/mnt/data1/sage/actin-segmentation/data/deepict/deepict_actin/background_masks",
    #    pixel_size = 13.48
    #)


    # apply ignore label for masking background during supervised training 
    PARENT_DIR = Path("/mnt/data1/sage/actin-segmentation/data/deepict/deepict_actin/")
    MASK_DIR = PARENT_DIR / "background_masks"
    h5_paths = [PARENT_DIR / "00004_cleaned.h5", PARENT_DIR / "00012_cleaned.h5"]
    mask_paths = [MASK_DIR / "00004.mrc", MASK_DIR / "00012.mrc"]

    for i, (path1, path2) in enumerate(zip(h5_paths, mask_paths)): 
        apply_ignore_label(path1, path2)
    
    write_h5_deepict()

if __name__ == "__main__":
    main()