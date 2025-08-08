import os
from glob import glob
from pathlib import Path

from synapse_net.ground_truth import extract_vesicle_training_data

ROOT = "/mnt/ceph-hdd/cold/nim00007/LDCVs_matea/20250714_EEC_Gut_2D_segmentation"  # noqa
OUT_ROOT = "/mnt/ceph-hdd/cold/nim00007/LDCVs_matea/extracted/res_3"  # noqa


def extract_01():
    name = "01_hoi_maus_2020_incomplete"
    input_folder = os.path.join(ROOT, name)
    output_folder = os.path.join(OUT_ROOT, name)

    def to_label_path(gt_folder, relative_file_path):
        rel_path = Path(relative_file_path)
        folder, fname = rel_path.parent, rel_path.stem

        # this filepattern matches the files in A_WT_SC_DIV_14
        imod_name = fname.rstrip("_SP").split("_")
        imod_name = "_".join(imod_name[:-1] + ["mtk"] + imod_name[-1:])
        imod_path = os.path.join(gt_folder, folder, f"{imod_name}.mod")

        # try matching the closest filename
        if not os.path.exists(imod_path):

            if str(folder).startswith("E"):
                match_name = "_".join(fname.replace("_MS", "").split("_")[:6])
            elif fname.startswith("WT_Unt_05646_D1_3.1_DIV16") and str(folder).startswith("B"):
                match_name = "WT_Unt_05646_D1_3.1_DIV16"
            else:
                match_name = fname

            imod_file_names = [os.path.basename(xx) for xx in sorted(glob(os.path.join(gt_folder, folder, "*.mod")))]
            matching_name = [name for name in imod_file_names if (name.startswith(match_name) and "AZ" not in name)]
            # matching_name = [name for name in imod_file_names if name.startswith(match_name)]
            if len(matching_name) != 1:
                if len(matching_name) == 2:
                    if fname.endswith("05_MS"):
                        assert matching_name[0] == "M13_CTRL_22723_O2_05_DIV29_mtk_05.mod"
                        imod_path = os.path.join(gt_folder, folder, matching_name[0])
                        return imod_path
                    elif fname.endswith("5.2_MS"):
                        assert matching_name[1] == "M13_CTRL_22723_O2_05_DIV29_mtk_5.2.mod"
                        imod_path = os.path.join(gt_folder, folder, matching_name[1])
                        return imod_path
                    elif fname.startswith("WT_MF_DIV28_06_MS_09204_F1"):  # use just one of the AZs
                        imod_path = os.path.join(gt_folder, folder, matching_name[0])
                        return imod_path
                raise RuntimeError(
                    f"Did not find a matching annotation for {relative_file_path}, matching with {match_name}. "
                    f"Found: {matching_name}"
                )
            imod_path = os.path.join(gt_folder, folder, matching_name[0])

        assert os.path.exists(imod_path), imod_path
        return imod_path

    extract_vesicle_training_data(
        input_folder, input_folder, output_folder, to_label_path, skip_no_labels=False,
        exclude_label_patterns=["wimp", "contact_sites", "Exo/endo", "ExoEndo"]
    )


def extract_02():
    name = "02_hcc_nanogold"
    input_folder = os.path.join(ROOT, name)
    output_folder = os.path.join(OUT_ROOT, name)

    def to_label_path(gt_folder, relative_file_path):
        rel_path = Path(relative_file_path)
        folder, fname = rel_path.parent, rel_path.stem
        imod_path = os.path.join(gt_folder, folder, f"{fname}.mod")
        return imod_path

    # Note: the resolution information in the mrc header is wrong.
    # the correct resolution is 1.2 nm.
    extract_vesicle_training_data(
        input_folder, input_folder, output_folder, to_label_path, skip_no_labels=False,
        visualize=False, exclude_label_patterns=["plasma membrane"],
        resolution=(1.2, 1.2, 1.2)
    )
    return output_folder


def extract_03():
    name = "03_hog_cs1sy7"
    input_folder = os.path.join(ROOT, name)
    output_folder = os.path.join(OUT_ROOT, name)

    def to_label_path(gt_folder, relative_file_path):
        rel_path = Path(relative_file_path)
        folder, fname = rel_path.parent, rel_path.stem
        imod_path = os.path.join(gt_folder, folder, f"{fname}_AZ01.mod")
        if not os.path.exists(imod_path):
            imod_path = os.path.join(gt_folder, folder, f"{fname}_AZ1.mod")
        return imod_path

    extract_vesicle_training_data(
        input_folder, input_folder, output_folder, to_label_path, skip_no_labels=False,
        visualize=False, exclude_label_patterns=["plasma membrane"],
    )
    return output_folder


def extract_04():
    name = "04_hoi_stem_examples"
    input_folder = os.path.join(ROOT, name)
    output_folder = os.path.join(OUT_ROOT, name)

    def to_label_path(gt_folder, relative_file_path):
        rel_path = Path(relative_file_path)
        folder, fname = rel_path.parent, rel_path.stem
        imod_path = os.path.join(gt_folder, folder, f"{fname}.mod")
        return imod_path

    extract_vesicle_training_data(
        input_folder, input_folder, output_folder, to_label_path, skip_no_labels=True,
        visualize=False,
    )
    return output_folder


def extract_05():
    name = "05_stem750_sv_training"
    input_folder = os.path.join(ROOT, name)
    output_folder = os.path.join(OUT_ROOT, name)

    def to_label_path(gt_folder, relative_file_path):
        rel_path = Path(relative_file_path)
        folder, fname = rel_path.parent, rel_path.stem
        imod_path = os.path.join(gt_folder, folder, f"{fname}_SVs.mod")
        return imod_path

    extract_vesicle_training_data(
        input_folder, input_folder, output_folder, to_label_path, visualize=False,
    )
    return output_folder


# NOTE: This contains vesicles automatically segmented with some other method.
# We should not use it for now and will wait for Ben to copy over the original manual annotations.
def extract_06():
    name = "06_hoi_wt_stem750_fm"
    input_folder = os.path.join(ROOT, name)
    output_folder = os.path.join(OUT_ROOT, name)

    def to_label_path(gt_folder, relative_file_path):
        folder = Path(relative_file_path).parent
        fname = Path(relative_file_path).stem
        if fname == "36859_J1_66K_TS_CA3_MF_21_rec_2Kb1dawbp_crop":
            fname = "36859_J1_66K_TS_CA3_MF_21_rec_2Kb1dawbp"
        imod_path = os.path.join(gt_folder, folder, f"{fname}.mod")
        return imod_path

    extract_vesicle_training_data(
        input_folder, input_folder, output_folder, to_label_path, visualize=False,
        exclude_labels=[7, 10, 12, 30, 36, 40, 48, 49, 50]
    )
    return output_folder


def extract_07():
    name = "07_hoi_s1sy7_tem250_ihgp"
    input_folder = os.path.join(ROOT, name)
    output_folder = os.path.join(OUT_ROOT, name)

    def to_label_path(gt_folder, relative_file_path):
        rel_path = Path(relative_file_path)
        folder, fname = rel_path.parent, rel_path.stem
        imod_path = os.path.join(gt_folder, folder, f"{fname}.mod")
        return imod_path

    extract_vesicle_training_data(
        input_folder, input_folder, output_folder, to_label_path, visualize=False,
        exclude_label_patterns=["plasma membrane"]
    )
    return output_folder


# NOTE: Ben claims this is a complete duplicate of 02.
# we should check that, and if it is we can exclude it.
def extract_08():
    name = "08_hcc_wt_lumsyt1nb_tem250_kw"
    input_folder = os.path.join(ROOT, name)
    output_folder = os.path.join(OUT_ROOT, name)

    def to_label_path(gt_folder, relative_file_path):
        rel_path = Path(relative_file_path)
        folder, fname = rel_path.parent, rel_path.stem
        imod_path = os.path.join(gt_folder, folder, f"{fname}.mod")
        return imod_path

    extract_vesicle_training_data(
        input_folder, input_folder, output_folder, to_label_path, visualize=False,
        exclude_label_patterns=["plasma membrane"],
        resolution=(1.2, 1.2, 1.2)
    )


def extract_09():
    name = "09_stem750_66k"
    input_folder = os.path.join(ROOT, name)
    output_folder = os.path.join(OUT_ROOT, name)

    def to_label_path(gt_folder, relative_file_path):
        folder = Path(relative_file_path).parent
        fname = Path(relative_file_path).stem
        imod_path = os.path.join(gt_folder, folder, f"{fname}.mod")
        return imod_path

    extract_vesicle_training_data(
        input_folder, input_folder, output_folder, to_label_path, visualize=False,
        exclude_label_patterns=["presyn", "postsyn", "ribo"]
    )
    return output_folder


def extract_10():
    name = "10_tem_single_release"
    input_folder = os.path.join(ROOT, name)
    output_folder = os.path.join(OUT_ROOT, name)

    def to_label_path(gt_folder, relative_file_path):
        rel_path = Path(relative_file_path)
        folder, fname = rel_path.parent, rel_path.stem
        if fname == "36894_D2_36K_TS_R01A_MF03_rec_4kb3dang_WBP_AZ1":
            fname = "36894_D2_36K_TS_R01A_MF03_rec_4Kb3dang_WBP_AZ1"
        if fname == "36894_D2_36K_TS_R01A_MF06_rec_4kb3_dang_WBP_AZ1":
            fname = "36894_D2_36K_TS_R01A_MF06_rec_4Kb3dang_WBP_AZ1"
        if fname == "36894_D2_36K_TS_R01A_MF07_rec_4kb3dang_WBP_AZ1":
            fname = "36894_D2_36K_TS_R01A_MF07_rec_4Kb3dang_WBP_AZ1"
        imod_path = os.path.join(gt_folder, folder, f"{fname}.mod")
        return imod_path

    extract_vesicle_training_data(
        input_folder, input_folder, output_folder, to_label_path, visualize=False,
        exclude_label_patterns=[
            " Plasma Membrane", " Active Zone Plasma Membrane",
            " Mitochondrial Position", " Endocytic Structure Position"]
    )


def extract_11():
    name = "11_tem_multiple_release"
    input_folder = os.path.join(ROOT, name)
    output_folder = os.path.join(OUT_ROOT, name)

    def to_label_path(gt_folder, relative_file_path):
        rel_path = Path(relative_file_path)
        folder, fname = rel_path.parent, rel_path.stem

        if fname == "36894_D2_36K_TS_R01A_MF05_rec_4kb3dang_WBP_AZ1":
            fname = "36894_D2_36K_TS_R01A_MF05_rec_4Kb3dang_WBP"
        else:
            fname = fname.replace("_AZ1", "")

        imod_path = os.path.join(gt_folder, folder, f"{fname}_combinedAZs.mod")
        if not os.path.exists(imod_path):
            fname = fname.replace("dang", "_dang")
            imod_path = os.path.join(gt_folder, folder, f"{fname}_combinedAZs.mod")
        return imod_path

    extract_vesicle_training_data(
        input_folder, input_folder, output_folder, to_label_path, visualize=False,
        exclude_label_patterns=[" Plasma Membrane", " Active Zone Plasma Membrane"]
    )
    return output_folder


def extract_12():
    name = "12_chemical_fix_cryopreparation"
    input_folder = os.path.join(ROOT, name)
    output_folder = os.path.join(OUT_ROOT, name)

    def to_label_path(gt_folder, relative_file_path):
        rel_path = Path(relative_file_path)
        folder, fname = rel_path.parent, rel_path.stem
        fname = fname.split("_")[0]
        imod_path = os.path.join(gt_folder, folder, f"{fname}.mod")
        return imod_path

    extract_vesicle_training_data(
        input_folder, input_folder, output_folder, to_label_path, visualize=False,
        skip_no_labels=True,
        exclude_label_patterns=[" Exo/Endo"]
    )
    return output_folder

def extract_LDCV():
    input_folder = os.path.join(ROOT)
    output_folder = os.path.join(OUT_ROOT)

    extract_vesicle_training_data(
        input_folder, input_folder, output_folder, visualize=False,
        skip_no_labels=True,
        exclude_labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], resolution=(3, 3, 3)
    )
    
def main():
    '''extract_01()
    extract_02()
    extract_03()
    extract_04()
    extract_05()
    extract_06()
    extract_07()
    extract_08()
    extract_09()
    extract_10()
    extract_11()
    extract_12()'''
    extract_LDCV()


if __name__ == "__main__":
    main()
