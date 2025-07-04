import os
from glob import glob

import h5py
import tifffile
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
from skimage.measure import label
from scipy.ndimage import binary_closing

ROOT = "/mnt/ceph-hdd/cold/nim00007/AZ_data/training_data"
INTER_ROOT = "/mnt/ceph-hdd/cold/nim00007/AZ_predictions"
OUTPUT_ROOT = "/mnt/ceph-hdd/cold/nim00007/new_AZ_train_data"
STEM_INPUT="/mnt/lustre-emmy-hdd/usr/u12095/synaptic_reconstruction/for_revison/postprocessed_AZ"
TIF_INPUT = "/mnt/ceph-hdd/cold/nim00007/new_AZ_train_data/stem/"


def _check_data(files, label_folder, check_thinned):
    for ff in files:
        with h5py.File(ff, "r") as f:
            shape = f["raw"].shape
            az = f["labels/az"][:]
        n_az = az.max()

        if check_thinned:
            label_file = os.path.join(label_folder, os.path.basename(ff))
            with h5py.File(label_file, "r") as f:
                az_thin = f["labels/az_thin2"][:]
            n_az_thin = az_thin.max()
        else:
            n_az_thin = None

        print(os.path.basename(ff), ":", shape, ":", n_az, ":", n_az_thin)


def assort_tem():
    old_name = "01data_withoutInvertedFiles_minusSVseg_corrected"
    new_name = "tem"

    raw_folder = os.path.join(ROOT, old_name)
    label_folder = os.path.join(INTER_ROOT, old_name)
    output_folder = os.path.join(OUTPUT_ROOT, new_name)
    os.makedirs(output_folder, exist_ok=True)

    files = glob(os.path.join(raw_folder, "*.h5"))
    for ff in tqdm(files):
        with h5py.File(ff, "r") as f:
            raw = f["raw"][:]
            az = f["labels/az"][:]

        label_path = os.path.join(label_folder, os.path.basename(ff))
        with h5py.File(label_path, "r") as f:
            az_thin = f["labels/az_thin2"][:]

        z_range1 = np.where(az != 0)[0]
        z_range2 = np.where(az != 0)[0]
        z_range = slice(
            np.min(np.concatenate([z_range1, z_range2])),
            np.max(np.concatenate([z_range1, z_range2])) + 1,
        )
        raw, az, az_thin = raw[z_range], az[z_range], az_thin[z_range]

        out_path = os.path.join(output_folder, os.path.basename(ff))
        with h5py.File(out_path, "a") as f:
            f.create_dataset("raw", data=raw, compression="lzf")
            f.create_dataset("labels/az_thin", data=az_thin, compression="lzf")
            f.create_dataset("labels/az", data=az, compression="lzf")


def assort_chemical_fixation():
    old_name = "12_chemical_fix_cryopreparation_minusSVseg_corrected"
    new_name = "chemical_fixation"

    raw_folder = os.path.join(ROOT, old_name)
    label_folder = os.path.join(INTER_ROOT, old_name)
    output_folder = os.path.join(OUTPUT_ROOT, new_name)
    os.makedirs(output_folder, exist_ok=True)

    label_key = "labels/az_thin2"

    files = glob(os.path.join(raw_folder, "*.h5"))
    for ff in tqdm(files):
        with h5py.File(ff, "r") as f:
            raw = f["raw"][:]
            az = f["labels/az"][:]

        label_path = os.path.join(label_folder, os.path.basename(ff))
        with h5py.File(label_path, "r") as f:
            az_thin = f[label_key][:]

        z_range1 = np.where(az != 0)[0]
        z_range2 = np.where(az != 0)[0]
        z_range = slice(
            np.min(np.concatenate([z_range1, z_range2])),
            np.max(np.concatenate([z_range1, z_range2])) + 1,
        )
        raw, az, az_thin = raw[z_range], az[z_range], az_thin[z_range]

        out_path = os.path.join(output_folder, os.path.basename(ff))
        with h5py.File(out_path, "a") as f:
            f.create_dataset("raw", data=raw, compression="lzf")
            f.create_dataset("labels/az_thin", data=az_thin, compression="lzf")
            f.create_dataset("labels/az", data=az, compression="lzf")


def assort_stem():
    old_names = [
        "04_hoi_stem_examples_fidi_and_sarah_corrected",
        "04_hoi_stem_examples_minusSVseg_cropped_corrected",
        "06_hoi_wt_stem750_fm_minusSVseg_cropped_corrected",
    ]
    new_names = ["stem", "stem_cropped", "stem_cropped"]
    for old_name, new_name in zip(old_names, new_names):
        print(old_name)
        raw_folder = os.path.join(ROOT, f"{old_name}_rescaled_tomograms")
        label_folder = os.path.join(INTER_ROOT, old_name)
        files = glob(os.path.join(raw_folder, "*.h5"))

        # _check_data(files, label_folder, check_thinned=True)
        # continue

        output_folder = os.path.join(OUTPUT_ROOT, new_name)
        os.makedirs(output_folder, exist_ok=True)
        for ff in tqdm(files):
            with h5py.File(ff, "r") as f:
                raw = f["raw"][:]
                az = f["labels/az"][:]

            label_path = os.path.join(label_folder, os.path.basename(ff))
            with h5py.File(label_path, "r") as f:
                az_thin = f["labels/az_thin2"][:]
            az_thin = resize(az_thin, az.shape, order=0, anti_aliasing=False, preserve_range=True).astype(az_thin.dtype)
            assert az_thin.shape == az.shape

            out_path = os.path.join(output_folder, os.path.basename(ff))
            with h5py.File(out_path, "a") as f:
                f.create_dataset("raw", data=raw, compression="lzf")
                f.create_dataset("labels/az_thin", data=az_thin, compression="lzf")
                f.create_dataset("labels/az", data=az, compression="lzf")


def assort_wichmann():
    old_name = "wichmann_withAZ_rescaled_tomograms"
    new_name = "endbulb_of_held"

    raw_folder = os.path.join(ROOT, old_name)
    output_folder = os.path.join(OUTPUT_ROOT, new_name)
    os.makedirs(output_folder, exist_ok=True)

    files = glob(os.path.join(raw_folder, "*.h5"))

    output_folder = os.path.join(OUTPUT_ROOT, new_name)
    os.makedirs(output_folder, exist_ok=True)
    for ff in tqdm(files):
        with h5py.File(ff, "r") as f:
            raw = f["raw"][:]
            az = f["labels/az"][:]

        output_file = os.path.join(output_folder, os.path.basename(ff))
        with h5py.File(output_file, "a") as f:
            f.create_dataset("raw", data=raw, compression="lzf")
            f.create_dataset("labels/az", data=az, compression="lzf")
            f.create_dataset("labels/az_thin", data=az, compression="lzf")


def crop_wichmann():
    input_name = "endbulb_of_held"
    output_name = "endbulb_of_held_cropped"

    input_folder = os.path.join(OUTPUT_ROOT, input_name)
    output_folder = os.path.join(OUTPUT_ROOT, output_name)
    os.makedirs(output_folder, exist_ok=True)
    files = glob(os.path.join(input_folder, "*.h5"))

    min_shape = (32, 512, 512)

    for ff in tqdm(files):
        with h5py.File(ff, "r") as f:
            az = f["labels/az"][:]
            bb = np.where(az != 0)
            bb = tuple(slice(int(b.min()), int(b.max()) + 1) for b in bb)
            pad_width = [max(sh - (b.stop - b.start), 0) // 2 for b, sh in zip(bb, min_shape)]
            bb = tuple(
                slice(max(b.start - pw, 0), min(b.stop + pw, sh)) for b, pw, sh in zip(bb, pad_width, az.shape)
            )
            az = az[bb]
            raw = f["raw"][bb]

        # import napari
        # v = napari.Viewer()
        # v.add_image(raw)
        # v.add_labels(az)
        # v.add_labels(az_thin)
        # napari.run()

        output_path = os.path.join(output_folder, os.path.basename(ff).replace(".h5", "_cropped.h5"))
        with h5py.File(output_path, "a") as f:
            f.create_dataset("raw", data=raw, compression="lzf")
            f.create_dataset("labels/az", data=az, compression="lzf")
            f.create_dataset("labels/az_thin", data=az, compression="lzf")

def crop_stem():
    #forgot about 06, added later
    input_name = "06_hoi_wt_stem750_fm_minusSVseg"#"04_hoi_stem_examples_minusSVseg"
    output_name = "stem_cropped2"

    input_folder = os.path.join(STEM_INPUT, input_name)
    output_folder = os.path.join(OUTPUT_ROOT, output_name)
    os.makedirs(output_folder, exist_ok=True)
    files = glob(os.path.join(input_folder, "*.h5"))

    min_shape = (32, 512, 512)

    for ff in tqdm(files):
        with h5py.File(ff, "r") as f:
            az = f["labels/az"][:]
            raw_full = f["raw"][:]

        # Label connected components in the az volume 
        labeled = label(az)
        num, sizes = np.unique(labeled, return_counts=True)
        #print(f"num {num}, sizes {sizes}")
        num, sizes = num[1:], sizes[1:]

        #exclude artifacts and background
        keep_labels = num[(sizes > 2000) & (num != 0)]
        #print(f"keep_labels {keep_labels}")

        #Clean up az annotations
        az = np.isin(labeled, keep_labels).astype("uint8")
        # Apply binary closing.
        az = np.logical_or(az, binary_closing(az, iterations=4)).astype("uint8")

        crop_id = 1
        for l in keep_labels:

            output_path = os.path.join(output_folder, os.path.basename(ff).replace(".h5", f"_crop{crop_id}.h5"))
            if os.path.exists(output_path):
                print(f"Skipping existing file: {output_path}")
                crop_id += 1
                continue


            mask = labeled == l
            bb = np.where(mask)
            if not bb[0].size:
                continue
            bb = tuple(slice(int(b.min()), int(b.max()) + 1) for b in bb)
            pad_width = [max(sh - (b.stop - b.start), 0) // 2 for b, sh in zip(bb, min_shape)]
            bb = tuple(
                slice(max(b.start - pw, 0), min(b.stop + pw, sh)) for b, pw, sh in zip(bb, pad_width, az.shape)
            )
            az_crop = az[bb]
            raw_crop = raw_full[bb]


            import napari
            v = napari.Viewer()
            v.add_image(raw_crop)
            v.add_labels(az_crop)
            napari.run()

            with h5py.File(output_path, "a") as f:
                f.create_dataset("raw", data=raw_crop, compression="lzf")
                f.create_dataset("labels/az", data=az_crop, compression="lzf")
            crop_id += 1

def get_bounding_box_3d(file_path, raw_volume):
    volume = tifffile.imread(file_path)
    filename = os.path.basename(file_path)
    print(f"filename {filename}")
    
    # Find the z index where the 2D rectangle is located (non-zero slice)
    z_indices = np.where(np.any(volume, axis=(1, 2)))[0]
    
    if len(z_indices) == 0:
        raise ValueError("No non-zero 2D rectangle found in the volume.")
    
    z_rect = z_indices[0]
    
    # Get the 2D mask from that slice
    mask_2d = volume[z_rect]
    y_indices, x_indices = np.where(mask_2d)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        raise ValueError("Found slice has no non-zero pixels.")
    
    x_min, x_max = x_indices.min(), x_indices.max() + 1
    y_min, y_max = y_indices.min(), y_indices.max() + 1
    
    # Determine z_start and z_end based on filename
    if filename.endswith("_toend.tif"):
        z_start, z_end = z_rect, raw_volume.shape[0]
    elif filename.endswith("_tostart.tif"):
        z_start, z_end = 0, z_rect + 1
    else:
        print("here?")
        z_start, z_end = z_rect, z_rect + 1

    # Return bounding box as slices, usable directly for numpy indexing
    return (
        slice(z_start, z_end),
        slice(y_min, y_max),
        slice(x_min, x_max)
    )

def neg_crop_stem():
    input_name = "mask_for_neg_example"#"04_hoi_stem_examples_minusSVseg"
    output_name = "stem_cropped2"

    input_folder = TIF_INPUT
    tif_input_folder = os.path.join(TIF_INPUT, input_name)
    output_folder = os.path.join(OUTPUT_ROOT, output_name)
    os.makedirs(output_folder, exist_ok=True)
    tif_files = glob(os.path.join(tif_input_folder, "*.tif"))
    print(f"tif_files {tif_files}")

    for ff in tqdm(tif_files):
        input_path = os.path.join(input_folder, os.path.basename(ff).replace('_tostart.tif', '.h5').replace('_toend.tif', '.h5'))
        with h5py.File(input_path, "r") as f:
            raw_full = f["raw"][:]


        output_path = os.path.join(output_folder, os.path.basename(ff).replace('_tostart.tif', '_cropped_noAZ.h5').replace('_toend.tif', '_cropped_noAZ.h5'))
        if os.path.exists(output_path):
            print(f"Skipping existing file: {output_path}")
            continue


        bb = get_bounding_box_3d(ff, raw_full)
        print(f"bb {bb}")

        raw_crop = raw_full[bb]


        import napari
        v = napari.Viewer()
        v.add_image(raw_crop)
        napari.run()

        with h5py.File(output_path, "a") as f:
            f.create_dataset("raw", data=raw_crop, compression="lzf")

def main():
    # assort_tem()
    # assort_chemical_fixation()

    # assort_stem()

    # assort_wichmann()
    #crop_wichmann()

    #crop_stem()
    neg_crop_stem()


if __name__ == "__main__":
    main()
