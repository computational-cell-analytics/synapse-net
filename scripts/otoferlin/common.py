import os
from glob import glob


# These are the files just for the test data.
# INPUT_ROOT = "/home/ag-wichmann/data/test-data/tomograms"
# OUTPUT_ROOT = "/home/ag-wichmann/data/test-data/segmentation"

# These are the otoferlin tomograms.
INPUT_ROOT = "/home/ag-wichmann/data/otoferlin/tomograms"
OUTPUT_ROOT = "/home/ag-wichmann/data/otoferlin/segmentation"

STRUCTURE_NAMES = ("ribbon", "PD", "membrane")


def get_folders():
    if os.path.exists(INPUT_ROOT):
        return INPUT_ROOT, OUTPUT_ROOT
    root_in = "./data/tomograms"
    assert os.path.exists(root_in)
    root_out = "./data/segmentation"
    return root_in, root_out


def get_all_tomograms():
    root, _ = get_folders()
    tomograms = glob(os.path.join(root, "**", "*.mrc"), recursive=True)
    tomograms += glob(os.path.join(root, "**", "*.rec"), recursive=True)
    tomograms = sorted(tomograms)
    return tomograms


def get_seg_path(mrc_path, version=1):
    input_root, output_root = get_folders()
    rel_path = os.path.relpath(mrc_path, input_root)
    rel_folder, fname = os.path.split(rel_path)
    fname = os.path.splitext(fname)[0]
    seg_path = os.path.join(output_root, f"v{version}", rel_folder, f"{fname}.h5")
    return seg_path


def get_colormaps():
    pool_map = {
        "RA-V": (0, 0.33, 0),
        "MP-V": (1.0, 0.549, 0.0),
        "Docked-V": (1, 1, 0),
    }
    return {"pools": pool_map}


# TODO: sync the ukon folder with the tomograms.
# UKON Path:
# /run/user/1000/gvfs/smb-share:server=wfs-medizin.top.gwdg.de,share=ukon-all$/UKON100/archiv/EM/For Segmentation
def sync_tomograms():
    pass


if __name__ == "__main__":
    tomos = get_all_tomograms()
    print("We have", len(tomos), "tomograms")
