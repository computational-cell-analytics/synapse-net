import os
from glob import glob

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

chemical_fixation = "Chemical Fixation"
single_ax_tem = "Single-Axis TEM"
dual_ax_tem = "Dual-Axis TEM"
stem = "STEM"


def aggregate_vesicle_train_data(roots, conditions, resolutions):
    tomo_names = []
    tomo_vesicles_all, tomo_vesicles_imod = [], []
    tomo_condition = []
    tomo_resolution = []
    tomo_train = []

    def aggregate_split(ds, split_root, split):
        if ds.startswith("04"):
            tomograms = sorted(glob(os.path.join(split_root, "2024**", "*.h5"), recursive=True))
        else:
            tomograms = sorted(glob(os.path.join(split_root, "*.h5")))

        assert len(tomograms) > 0, ds
        this_condition = conditions[ds]
        this_resolution = resolutions[ds][0]

        for tomo_path in tqdm(tomograms, desc=f"Aggregate {split}"):
            fname = os.path.basename(tomo_path)
            with h5py.File(tomo_path, "r") as f:
                try:
                    tomo_name = f.attrs["filename"]
                except KeyError:
                    tomo_name = fname

                if "labels/vesicles/combined_vesicles" in f:
                    all_vesicles = f["labels/vesicles/combined_vesicles"][:]
                    imod_vesicles = f["labels/vesicles/masked_vesicles"][:]
                    n_vesicles_all = len(np.unique(all_vesicles)) - 1
                    n_vesicles_imod = len(np.unique(imod_vesicles)) - 2
                else:
                    vesicles = f["labels/vesicles"][:]
                    n_vesicles_all = len(np.unique(vesicles)) - 1
                    n_vesicles_imod = n_vesicles_all

            tomo_names.append(tomo_name)
            tomo_vesicles_all.append(n_vesicles_all)
            tomo_vesicles_imod.append(n_vesicles_imod)
            tomo_condition.append(this_condition)
            tomo_resolution.append(this_resolution)
            tomo_train.append(split)

    for ds, root in roots.items():
        print("Aggregate data for", ds)
        train_root = root["train"]
        if train_root != "":
            aggregate_split(ds, train_root, "train/val")
        test_root = root["test"]
        if test_root != "":
            aggregate_split(ds, test_root, "test")

    df = pd.DataFrame({
        "tomogram": tomo_names,
        "condition": tomo_condition,
        "resolution": tomo_resolution,
        "used_for": tomo_train,
        "vesicle_count_all": tomo_vesicles_all,
        "vesicle_count_imod": tomo_vesicles_imod,
    })

    os.makedirs("data_summary", exist_ok=True)
    df.to_excel("./data_summary/vesicle_training_data.xlsx", index=False)


def vesicle_train_data():
    roots = {
        "01": {
            "train": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/vesicles_processed_v2/01_hoi_maus_2020_incomplete",  # noqa
            "test": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/vesicles_processed_v2/testsets/01_hoi_maus_2020_incomplete",  # noqa
        },
        "02": {
            "train": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/vesicles_processed_v2/02_hcc_nanogold",  # noqa
            "test": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/vesicles_processed_v2/testsets/02_hcc_nanogold",  # noqa
        },
        "03": {
            "train": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/vesicles_processed_v2/03_hog_cs1sy7",  # noqa
            "test": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/vesicles_processed_v2/testsets/03_hog_cs1sy7",  # noqa
        },
        "04": {
            "train": "",
            "test": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/ground_truth/04Dataset_for_vesicle_eval/",  # noqa
        },
        "05": {
            "train": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/vesicles_processed_v2/05_stem750_sv_training",  # noqa
            "test": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/vesicles_processed_v2/testsets/05_stem750_sv_training",  # noqa
        },
        "07": {
            "train": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/vesicles_processed_v2/07_hoi_s1sy7_tem250_ihgp",  # noqa
            "test": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/vesicles_processed_v2/testsets/07_hoi_s1sy7_tem250_ihgp",  # noqa
        },
        "09": {
            "train": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/vesicles_processed_v2/09_stem750_66k",  # noqa
            "test": "",
        },
        "10": {
            "train": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/vesicles_processed_v2/10_tem_single_release",  # noqa
            "test": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/vesicles_processed_v2/testsets/10_tem_single_release",  # noqa
        },
        "11": {
            "train": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/vesicles_processed_v2/11_tem_multiple_release",  # noqa
            "test": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/vesicles_processed_v2/testsets/11_tem_multiple_release",  # noqa
        },
        "12": {
            "train": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/vesicles_processed_v2/12_chemical_fix_cryopreparation",  # noqa
            "test": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/vesicles_processed_v2/testsets/12_chemical_fix_cryopreparation",  # noqa
        },
    }

    conditions = {
        "01": single_ax_tem,
        "02": dual_ax_tem,
        "03": dual_ax_tem,
        "04": stem,
        "05": stem,
        "07": dual_ax_tem,
        "09": stem,
        "10": dual_ax_tem,
        "11": dual_ax_tem,
        "12": chemical_fixation,
    }

    resolutions = {
        "01": (1.554, 1.554, 1.554),
        "02": (1.2, 1.2, 1.2),
        "03": (1.24, 1.24, 1.24),
        "04": (0.868, 0.868, 0.868),
        "05": (0.868, 0.868, 0.868),
        "07": (1.24, 1.24, 1.24),
        "09": (0.868, 0.868, 0.868),
        "10": (1.24, 1.24, 1.24),
        "11": (1.24, 1.24, 1.24),
        "12": (1.554, 1.554, 1.554)
    }

    aggregate_vesicle_train_data(roots, conditions, resolutions)


def aggregate_az_train_data(roots, test_tomograms, conditions, resolutions):
    tomo_names = []
    tomo_azs = []
    tomo_condition = []
    tomo_resolution = []
    tomo_train = []

    for ds, root in roots.items():
        print("Aggregate data for", ds)
        tomograms = sorted(glob(os.path.join(root, "*.h5")))
        this_test_tomograms = test_tomograms[ds]

        assert len(tomograms) > 0, ds
        this_condition = conditions[ds]
        this_resolution = resolutions[ds][0]

        for tomo_path in tqdm(tomograms):
            fname = os.path.basename(tomo_path)
            with h5py.File(tomo_path, "r") as f:
                if "labels" not in f:
                    print("Can't find AZ labels in", tomo_path)
                    continue
                n_label_sets = len(f["labels"])
                if n_label_sets > 1:
                    print(tomo_path, "contains the following labels:", list(f["labels"].keys()))
                seg = f["labels/AZ"][:]
                n_az = len(np.unique(seg)) - 1

            tomo_names.append(fname)
            tomo_azs.append(n_az)
            tomo_condition.append(this_condition)
            tomo_resolution.append(this_resolution)
            tomo_train.append("test" if fname in this_test_tomograms else "train/val")

    df = pd.DataFrame({
        "tomogram": tomo_names,
        "condition": tomo_condition,
        "resolution": tomo_resolution,
        "used_for": tomo_train,
        "az_count": tomo_azs,
    })

    os.makedirs("data_summary", exist_ok=True)
    df.to_excel("./data_summary/active_zone_training_data.xlsx", index=False)


def active_zone_train_data():
    roots = {
        "01": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/exported_imod_objects/01_hoi_maus_2020_incomplete",  # noqa
        "04": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/exported_imod_objects/04_hoi_stem_examples",  # noqa
        "06": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/exported_imod_objects/06_hoi_wt_stem750_fm",  # noqa
        "12": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/2D_data/20241021_imig_2014_data_transfer_exported_grouped",  # noqa
    }

    test_tomograms = {
        "01": [
            "WT_MF_DIV28_01_MS_09204_F1.h5", "WT_MF_DIV14_01_MS_B2_09175_CA3.h5", "M13_CTRL_22723_O2_05_DIV29_5.2.h5", "WT_Unt_SC_09175_D4_05_DIV14_mtk_05.h5",  # noqa
            "20190805_09002_B4_SC_11_SP.h5", "20190807_23032_D4_SC_01_SP.h5", "M13_DKO_22723_A1_03_DIV29_03_MS.h5", "WT_MF_DIV28_05_MS_09204_F1.h5", "M13_CTRL_09201_S2_06_DIV31_06_MS.h5", # noqa
            "WT_MF_DIV28_1.2_MS_09002_B1.h5", "WT_Unt_SC_09175_C4_04_DIV15_mtk_04.h5",   "M13_DKO_22723_A4_10_DIV29_10_MS.h5",  "WT_MF_DIV14_3.2_MS_D2_09175_CA3.h5",  # noqa
               "20190805_09002_B4_SC_10_SP.h5", "M13_CTRL_09201_S2_02_DIV31_02_MS.h5", "WT_MF_DIV14_04_MS_E1_09175_CA3.h5", "WT_MF_DIV28_10_MS_09002_B3.h5",   "WT_Unt_SC_05646_D4_02_DIV16_mtk_02.h5",   "M13_DKO_22723_A4_08_DIV29_08_MS.h5",  "WT_MF_DIV28_04_MS_09204_M1.h5",   "WT_MF_DIV28_03_MS_09204_F1.h5",   "M13_DKO_22723_A1_05_DIV29_05_MS.h5",  # noqa
            "WT_Unt_SC_09175_C4_06_DIV15_mtk_06.h5",  "WT_MF_DIV28_09_MS_09002_B3.h5", "20190524_09204_F4_SC_07_SP.h5",
               "WT_MF_DIV14_02_MS_C2_09175_CA3.h5",    "M13_DKO_23037_K1_01_DIV29_01_MS.h5",  "WT_Unt_SC_09175_E2_01_DIV14_mtk_01.h5", "20190807_23032_D4_SC_05_SP.h5",   "WT_MF_DIV14_01_MS_E2_09175_CA3.h5",   "WT_MF_DIV14_03_MS_B2_09175_CA3.h5",   "M13_DKO_09201_O1_01_DIV31_01_MS.h5",  "M13_DKO_09201_U1_04_DIV31_04_MS.h5",  # noqa
            "WT_MF_DIV14_04_MS_E2_09175_CA3_2.h5",   "WT_Unt_SC_09175_D5_01_DIV14_mtk_01.h5",
            "M13_CTRL_22723_O2_05_DIV29_05_MS_.h5",  "WT_MF_DIV14_02_MS_B2_09175_CA3.h5", "WT_MF_DIV14_01.2_MS_D1_09175_CA3.h5",  # noqa
        ],
        "04": [
            "36859_H3_SP_10_rec_2kb1dawbp_crop.h5", "36859_H3_SP_09_rec_2kb1dawbp_crop.h5",
            "36859_H2_SP_04_rec_2Kb1dawbp_crop.h5", "36859_J1_STEM750_66K_SP_15_rec_2kb1dawbp_crop.h5",
            "36859_H2_SP_10_rec_crop.h5", "36859_H2_SP_02_rec_2Kb1dawbp_crop.h5",
            "36859_H3_SP_01_rec_2kb1dawbp_crop.h5"
        ],
        "06": ["36859_J1_66K_TS_CA3_PS_43_rec_2Kb1dawbp_crop.h5"],
        "12": ["20180305_09_MS.h5", "20180305_04_MS.h5", "20180305_08_MS.h5",
               "20171113_04_MS.h5", "20171006_05_MS.h5", "20180305_01_MS.h5"],
    }

    conditions = {
        "01": single_ax_tem,
        "04": stem,
        "06": stem,
        "12": chemical_fixation,
    }

    resolutions = {
        "01": (1.554, 1.554, 1.554),
        "04": (0.868, 0.868, 0.868),
        "06": (0.868, 0.868, 0.868),
        "12": (1.554, 1.554, 1.554)
    }

    aggregate_az_train_data(roots, test_tomograms, conditions, resolutions)


def aggregate_compartment_train_data(roots, condition, resolution):
    tomo_names = []
    tomo_compartments = []

    for ds, root in roots.items():
        print("Aggregate data for", ds)
        if ds == "04":
            tomograms = sorted(glob(os.path.join(root, "**", "*.h5"), recursive=True))
        else:
            tomograms = sorted(glob(os.path.join(root, "*.h5")))

        for tomo_path in tqdm(tomograms):
            fname = os.path.basename(tomo_path)
            with h5py.File(tomo_path, "r") as f:
                seg = f["labels/compartments"][:]
                n_comp = len(np.unique(seg)) - 1
            tomo_names.append(fname)
            tomo_compartments.append(n_comp)

    n_tomos = len(tomo_names)
    df = pd.DataFrame({
        "tomogram": tomo_names,
        "condition": [condition] * n_tomos,
        "resolution": [resolution] * n_tomos,
        "used_for": ["train/val"] * n_tomos,
        "compartment_count": tomo_compartments,
    })

    os.makedirs("data_summary", exist_ok=True)
    df.to_excel("./data_summary/compartment_training_data.xlsx", index=False)


def compartment_train_data():
    roots = {
        "04": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/ground_truth/compartments/v3",
        "05": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/ground_truth/compartments/v2/05_stem750_sv_training",  # noqa
        "06": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/ground_truth/compartments/v2/06_hoi_wt_stem750_fm",  # noqa
    }

    condition = "stem"
    resolution = 4 * 0.868

    aggregate_compartment_train_data(roots, condition, resolution)


def aggregate_da(roots, train_tomograms, test_tomograms, resolutions):
    out_path = "./data_summary/vesicle_domain_adaptation_data.xlsx"

    for ds, root in roots.items():
        print("Extract data for", ds)
        root_train = root["train"]

        # Nested folder structure
        if ds in ("inner_ear", "endbulb"):
            tomo_paths = sorted(glob(os.path.join(root_train, "**", "*.h5"), recursive=True))
            # Exclude auto seg folder for endbulb.
            tomo_paths = [path for path in tomo_paths if "Automatische_Segmentierung_Dataset_Validierung" not in path]
            is_nested = True

        # Simple folder structure
        else:
            tomo_paths = sorted(glob(os.path.join(root_train, "*.h5")))
            tomo_paths += sorted(glob(os.path.join(root_train, "*.mrc")))
            is_nested = False
        assert len(tomo_paths) > 0

        this_resolution = resolutions[ds]
        this_train = train_tomograms.get(ds, None)
        this_test = test_tomograms.get(ds, None)

        tomo_names, tomo_vesicles, tomo_train = [], [], []
        for tomo in tqdm(tomo_paths, desc="Extract train tomograms."):
            fname = os.path.relpath(tomo, root_train) if is_nested else os.path.basename(tomo)
            if this_train and fname not in this_train:
                continue
            if this_test and fname in this_test:
                continue
            tomo_names.append(fname)
            tomo_vesicles.append(0)
            tomo_train.append("train/val")

        if "test" in root:
            root_test = root["test"]
            test_tomos = sorted(glob(os.path.join(root_test, "*.h5")))
            assert len(test_tomos) > 0
            for tomo in tqdm(test_tomos, desc="Extract test tomos"):
                with h5py.File(tomo) as f:
                    seg = f["labels/vesicles"][:]
                    n_ves = len(np.unique(seg)) - 1
                fname = os.path.basename(tomo)
                tomo_names.append(fname)
                tomo_vesicles.append(n_ves)
                tomo_train.append("test")

        n_tomos = len(tomo_names)
        df = pd.DataFrame({
            "tomogram": tomo_names,
            "resolution": [this_resolution] * n_tomos,
            "used_for": tomo_train,
            "vesicle_count": tomo_vesicles,
        })

        if ds.startswith("cryo"):
            da_name = "cryo"
            if os.path.exists(out_path):
                prev = pd.read_excel(out_path)
                df = pd.concat([prev, df])
                os.remove(out_path)
                df.to_excel(out_path, sheet_name=da_name, index=False)
                continue
        else:
            da_name = ds

        if os.path.exists(out_path):
            with pd.ExcelWriter(out_path, engine="openpyxl", mode="a") as writer:
                df.to_excel(writer, sheet_name=da_name, index=False)
        else:
            df.to_excel(out_path, sheet_name=da_name, index=False)


def vesicle_domain_adaptation_data():
    roots = {
        "cryo_old": {
            "train": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/fernandez-busnadiego/from_arsen/old_data"  # noqa
        },
        "cryo_deconv": {
            "train": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/fernandez-busnadiego/from_arsen/tomos_deconv_18924",   # noqa
            "test": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/fernandez-busnadiego/vesicle_gt/v3"  # noqa
        },
        "inner_ear": {
            "train": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/moser/inner_ear_data",  # noqa
            "test": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/moser/vesicle_gt/",  # noqa
        },
        "endbulb": {
            "train": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/wichmann/extracted/endbulb_of_held",   # noqa
            "test": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/wichmann/extracted/endbulb_of_held/Automatische_Segmentierung_Dataset_Validierung"   # noqa
        },
        "maus_2d": {
            "train": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/2D_data/maus_2020_tem2d_wt_unt_div14_exported_scaled/good_for_DAtraining/maus_2020_tem2d_wt_unt_div14_exported_scaled",  # noqa
            "test": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/vesicle_gt_2d/maus_2020_tem2d"  # noqa
        },
        "frog": {
            "train": "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/rizzoli/extracted/upsampled_by2"  # noqa
        }
    }

    train_tomograms = {
        "maus_2d": [
            "MF_05649_P-09175-E_06.h5", "MF_05646_C-09175-B_001B.h5", "MF_05649_P-09175-E_07.h5",
            "MF_05649_G-09175-C_001.h5", "MF_05646_C-09175-B_002.h5", "MF_05649_G-09175-C_04.h5",
            "MF_05649_P-09175-E_05.h5", "MF_05646_C-09175-B_000.h5", "MF_05646_C-09175-B_001.h5"
        ],
        "frog": [
            "block10U3A_three.h5", "block30UB_one_two.h5", "block30UB_two.h5", "block10U3A_one.h5",
            "block184B_one.h5", "block30UB_three.h5", "block10U3A_two.h5", "block30UB_four.h5",
            "block30UB_one.h5", "block10U3A_five.h5",
        ]
    }

    test_tomograms = {
        "inner_ear": [
            "WT-control_Mouse-1.0_modiolar/1.h5",
            "WT-control_Mouse-1.0_modiolar/2.h5",
            "WT-control_Mouse-1.0_modiolar/3.h5",
            "WT-control_Mouse-1.0_pillar/1.h5",
            "WT-control_Mouse-1.0_pillar/2.h5",
            "WT-control_Mouse-1.0_pillar/4.h5",
            "WT-strong-stim_Mouse-1.0_modiolar/2.h5",
            "WT-strong-stim_Mouse-1.0_modiolar/3.h5",
            "WT-strong-stim_Mouse-1.0_modiolar/9.h5",
            "WT-strong-stim_Mouse-1.0_pillar/1.h5",
            "WT-strong-stim_Mouse-1.0_pillar/2.h5",
            "WT-strong-stim_Mouse-1.0_pillar/4.h5",
        ],
        "endbulb": [
            "Adult_WT_Rest/1Otof_AVCN07_451A_WT_Rest_B3_10_35932.h5",
            "Young_WT_MStim/1Otof_AVCN03_429A_WT_M.Stim_D3_2_35461.h5",
            "Young_WT_Rest/1Otof_AVCN03_429D_WT_Rest_H5_4_35461.h5",
        ],
        "maus_2d": ["MF_05649_G-09175-C_04B.h5", "MF_05646_C-09175-B_000B.h5", "MF_05649_G-09175-C_03.h5", "MF_05649_G-09175-C_02.h5"],  # noqa
        "frog": ["block10U3A_four.h5", "block30UB_five.h5"],
    }

    # Cryo Test tomograms:
    # 33K: 1.46
    # 64K: 0.76
    resolutions = {
        "cryo_old": 0.917,
        "cryo_deconv": 0.756,
        "inner_ear": 1.47,
        "endbulb": 1.75,
        "maus_2d": 0.592,
        "frog": np.nan,
    }

    aggregate_da(roots, train_tomograms, test_tomograms, resolutions)


def get_n_images_frog():
    root = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/rizzoli/extracted/upsampled_by2"
    tomos = ["block10U3A_three.h5", "block30UB_one_two.h5", "block30UB_two.h5", "block10U3A_one.h5",
             "block184B_one.h5", "block30UB_three.h5", "block10U3A_two.h5", "block30UB_four.h5",
             "block30UB_one.h5", "block10U3A_five.h5"]

    n_images = 0
    for tomo in tomos:
        path = os.path.join(root, tomo)
        with h5py.File(path, "r") as f:
            n_images += f["raw"].shape[0]
    print(n_images)


def get_image_sizes_tem_2d():
    root = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/2D_data/maus_2020_tem2d_wt_unt_div14_exported_scaled/good_for_DAtraining/maus_2020_tem2d_wt_unt_div14_exported_scaled"  # noqa
    tomos = [
        "MF_05649_P-09175-E_06.h5", "MF_05646_C-09175-B_001B.h5", "MF_05649_P-09175-E_07.h5",
        "MF_05649_G-09175-C_001.h5", "MF_05646_C-09175-B_002.h5", "MF_05649_G-09175-C_04.h5",
        "MF_05649_P-09175-E_05.h5", "MF_05646_C-09175-B_000.h5", "MF_05646_C-09175-B_001.h5"
    ]
    for tomo in tomos:
        path = os.path.join(root, tomo)
        with h5py.File(path, "r") as f:
            print(f["raw"].shape)


def main():
    # active_zone_train_data()
    # compartment_train_data()
    # mito_train_data()
    vesicle_train_data()

    # vesicle_domain_adaptation_data()
    # get_n_images_frog()
    # get_image_sizes_tem_2d()


main()
