import os
from glob import glob
from shutil import copyfile

import h5py
from tqdm import tqdm

OUTPUT_ROOT = "./data_summary/for_zenodo"


def _copy_vesicles(tomos, out_folder):
    label_key = "labels/vesicles/combined_vesicles"
    os.makedirs(out_folder, exist_ok=True)
    for tomo in tqdm(tomos, desc="Export tomos"):
        out_path = os.path.join(out_folder, os.path.basename(tomo))
        if os.path.exists(out_path):
            continue

        with h5py.File(tomo, "r") as f:
            raw = f["raw"][:]
            labels = f[label_key][:]
            try:
                fname = f.attrs["filename"]
            except KeyError:
                fname = None

        with h5py.File(out_path, "a") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels/vesicles", data=labels, compression="gzip")
            if fname is not None:
                f.attrs["filename"] = fname


def _export_vesicles(train_root, test_root, name):
    train_tomograms = sorted(glob(os.path.join(train_root, "*.h5")))
    test_tomograms = sorted(glob(os.path.join(test_root, "*.h5")))
    print(f"Vesicle data for {name}:")
    print(len(train_tomograms), len(test_tomograms), len(train_tomograms) + len(test_tomograms))

    train_out = os.path.join(OUTPUT_ROOT, "synapse-net", "vesicles", "train", name)
    _copy_vesicles(train_tomograms, train_out)

    test_out = os.path.join(OUTPUT_ROOT, "synapse-net", "vesicles", "test", name)
    _copy_vesicles(test_tomograms, test_out)


def _export_az(train_root, test_tomos, name):
    tomograms = sorted(glob(os.path.join(train_root, "*.h5")))
    print(f"AZ data for {name}:")

    train_out = os.path.join(OUTPUT_ROOT, "synapse-net", "active_zones", "train", name)
    test_out = os.path.join(OUTPUT_ROOT, "synapse-net", "active_zones", "test", name)

    os.makedirs(train_out, exist_ok=True)
    os.makedirs(test_out, exist_ok=True)

    for tomo in tqdm(tomograms):
        fname = os.path.basename(tomo)
        if fname in test_tomos:
            out_path = os.path.join(test_out, fname)
        else:
            out_path = os.path.join(train_out, fname)
        if os.path.exists(out_path):
            continue

        with h5py.File(tomo, "r") as f:
            raw = f["raw"][:]
            az = f["labels/AZ"][:]

        with h5py.File(out_path, "a") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels/AZ", data=az, compression="gzip")


# NOTE: we have very few mito annotations from 01, so we don't include them in here.
def prepare_single_ax_stem_chemical_fix():
    # single-axis-tem: vesicles
    train_root = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/vesicles_processed_v2/01_hoi_maus_2020_incomplete"  # noqa
    test_root = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/vesicles_processed_v2/testsets/01_hoi_maus_2020_incomplete"  # noqa
    _export_vesicles(train_root, test_root, name="single_axis_tem")

    # single-axis-tem: active zones
    train_root = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/exported_imod_objects/01_hoi_maus_2020_incomplete"  # noqa
    test_tomos = [
        "WT_MF_DIV28_01_MS_09204_F1.h5", "WT_MF_DIV14_01_MS_B2_09175_CA3.h5", "M13_CTRL_22723_O2_05_DIV29_5.2.h5", "WT_Unt_SC_09175_D4_05_DIV14_mtk_05.h5",  # noqa
        "20190805_09002_B4_SC_11_SP.h5", "20190807_23032_D4_SC_01_SP.h5", "M13_DKO_22723_A1_03_DIV29_03_MS.h5", "WT_MF_DIV28_05_MS_09204_F1.h5", "M13_CTRL_09201_S2_06_DIV31_06_MS.h5", # noqa
        "WT_MF_DIV28_1.2_MS_09002_B1.h5", "WT_Unt_SC_09175_C4_04_DIV15_mtk_04.h5",   "M13_DKO_22723_A4_10_DIV29_10_MS.h5",  "WT_MF_DIV14_3.2_MS_D2_09175_CA3.h5",  # noqa
           "20190805_09002_B4_SC_10_SP.h5", "M13_CTRL_09201_S2_02_DIV31_02_MS.h5", "WT_MF_DIV14_04_MS_E1_09175_CA3.h5", "WT_MF_DIV28_10_MS_09002_B3.h5",   "WT_Unt_SC_05646_D4_02_DIV16_mtk_02.h5",   "M13_DKO_22723_A4_08_DIV29_08_MS.h5",  "WT_MF_DIV28_04_MS_09204_M1.h5",   "WT_MF_DIV28_03_MS_09204_F1.h5",   "M13_DKO_22723_A1_05_DIV29_05_MS.h5",  # noqa
        "WT_Unt_SC_09175_C4_06_DIV15_mtk_06.h5",  "WT_MF_DIV28_09_MS_09002_B3.h5", "20190524_09204_F4_SC_07_SP.h5",
           "WT_MF_DIV14_02_MS_C2_09175_CA3.h5",    "M13_DKO_23037_K1_01_DIV29_01_MS.h5",  "WT_Unt_SC_09175_E2_01_DIV14_mtk_01.h5", "20190807_23032_D4_SC_05_SP.h5",   "WT_MF_DIV14_01_MS_E2_09175_CA3.h5",   "WT_MF_DIV14_03_MS_B2_09175_CA3.h5",   "M13_DKO_09201_O1_01_DIV31_01_MS.h5",  "M13_DKO_09201_U1_04_DIV31_04_MS.h5",  # noqa
        "WT_MF_DIV14_04_MS_E2_09175_CA3_2.h5",   "WT_Unt_SC_09175_D5_01_DIV14_mtk_01.h5",
        "M13_CTRL_22723_O2_05_DIV29_05_MS_.h5",  "WT_MF_DIV14_02_MS_B2_09175_CA3.h5", "WT_MF_DIV14_01.2_MS_D1_09175_CA3.h5",  # noqa
    ]
    _export_az(train_root, test_tomos, name="single_axis_tem")

    # chemical_fixation: vesicles
    train_root = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/vesicles_processed_v2/12_chemical_fix_cryopreparation"  # noqa
    test_root = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/vesicles_processed_v2/testsets/12_chemical_fix_cryopreparation"  # noqa
    _export_vesicles(train_root, test_root, name="chemical_fixation")

    # chemical-fixation: active zones
    train_root = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/exported_imod_objects/12_chemical_fix_cryopreparation"  # noqa
    test_tomos = ["20180305_09_MS.h5", "20180305_04_MS.h5", "20180305_08_MS.h5",
                  "20171113_04_MS.h5", "20171006_05_MS.h5", "20180305_01_MS.h5"]
    _export_az(train_root, test_tomos, name="chemical_fixation")


def prepare_ier():
    root = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/moser/other_tomograms"
    sets = {
        "01_vesicle_pools": "vesicle_pools",
        "02_tether": "tether",
        "03_ratten_tomos": "rat",
    }

    output_folder = os.path.join(OUTPUT_ROOT, "IER")
    label_names = {
        "ribbons": "ribbon",
        "membrane": "membrane",
        "presynapse": "PD",
        "postsynapse": "PSD",
        "vesicles": "vesicles",
    }

    for name, output_name in sets.items():
        out_set = os.path.join(output_folder, output_name)
        os.makedirs(out_set, exist_ok=True)
        tomos = sorted(glob(os.path.join(root, name, "*.h5")))

        print("Export", output_name)
        for tomo in tqdm(tomos):
            with h5py.File(tomo, "r") as f:
                try:
                    fname = os.path.split(f.attrs["filename"])[1][:-4]
                except KeyError:
                    fname = f.attrs["path"][1]
                    fname = "_".join(fname.split("/")[-2:])

                out_path = os.path.join(out_set, os.path.basename(tomo))
                if os.path.exists(out_path):
                    continue

                raw = f["raw"][:]
                labels = {}
                for label_name, out_name in label_names.items():
                    key = f"labels/{label_name}"
                    if key not in f:
                        continue
                    labels[out_name] = f[key][:]

            with h5py.File(out_path, "a") as f:
                f.attrs["filename"] = fname
                f.create_dataset("raw", data=raw, compression="gzip")
                for label_name, seg in labels.items():
                    f.create_dataset(f"labels/{label_name}", data=seg, compression="gzip")


def prepare_frog():
    root = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/rizzoli/extracted"
    train_tomograms = [
        "block10U3A_three.h5", "block30UB_one_two.h5", "block30UB_two.h5", "block10U3A_one.h5",
        "block184B_one.h5", "block30UB_three.h5", "block10U3A_two.h5", "block30UB_four.h5",
        "block30UB_one.h5", "block10U3A_five.h5",
    ]
    test_tomograms = ["block10U3A_four.h5", "block30UB_five.h5"]

    output_folder = os.path.join(OUTPUT_ROOT, "frog")
    output_train = os.path.join(output_folder, "train_unlabeled")
    os.makedirs(output_train, exist_ok=True)

    for name in train_tomograms:
        path = os.path.join(root, name)
        out_path = os.path.join(output_train, name)
        if os.path.exists(out_path):
            continue
        copyfile(path, out_path)

    output_test = os.path.join(output_folder, "test")
    os.makedirs(output_test, exist_ok=True)
    for name in test_tomograms:
        path = os.path.join(root, name)
        out_path = os.path.join(output_test, name)
        if os.path.exists(out_path):
            continue
        copyfile(path, out_path)


def prepare_2d_tem():
    train_root = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/2D_data/maus_2020_tem2d_wt_unt_div14_exported_scaled/good_for_DAtraining/maus_2020_tem2d_wt_unt_div14_exported_scaled"  # noqa
    test_root = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/vesicle_gt_2d/maus_2020_tem2d"  # noqa
    train_images = [
        "MF_05649_P-09175-E_06.h5", "MF_05646_C-09175-B_001B.h5", "MF_05649_P-09175-E_07.h5",
        "MF_05649_G-09175-C_001.h5", "MF_05646_C-09175-B_002.h5", "MF_05649_G-09175-C_04.h5",
        "MF_05649_P-09175-E_05.h5", "MF_05646_C-09175-B_000.h5", "MF_05646_C-09175-B_001.h5"
    ]
    test_images = [
        "MF_05649_G-09175-C_04B.h5", "MF_05646_C-09175-B_000B.h5",
        "MF_05649_G-09175-C_03.h5", "MF_05649_G-09175-C_02.h5"
    ]
    print(len(train_images) + len(test_images))

    output_folder = os.path.join(OUTPUT_ROOT, "2d_tem")

    output_train = os.path.join(output_folder, "train_unlabeled")
    os.makedirs(output_train, exist_ok=True)
    for name in tqdm(train_images, desc="Export train images"):
        out_path = os.path.join(output_train, name)
        if os.path.exists(out_path):
            continue
        in_path = os.path.join(train_root, name)
        with h5py.File(in_path, "r") as f:
            raw = f["raw"][:]
        with h5py.File(out_path, "a") as f:
            f.create_dataset("raw", data=raw, compression="gzip")

    output_test = os.path.join(output_folder, "test")
    os.makedirs(output_test, exist_ok=True)
    for name in tqdm(test_images, desc="Export test images"):
        out_path = os.path.join(output_test, name)
        if os.path.exists(out_path):
            continue
        in_path = os.path.join(test_root, name)
        with h5py.File(in_path, "r") as f:
            raw = f["data"][:]
            labels = f["labels/vesicles"][:]
            mask = f["labels/mask"][:]
        with h5py.File(out_path, "a") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels/vesicles", data=labels, compression="gzip")
            f.create_dataset("labels/mask", data=mask, compression="gzip")


def prepare_munc_snap():
    pass


def main():
    prepare_single_ax_stem_chemical_fix()
    # prepare_2d_tem()
    # prepare_frog()
    # prepare_ier()
    # prepare_munc_snap()


if __name__ == "__main__":
    main()
