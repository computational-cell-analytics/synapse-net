import argparse
import os
from shutil import rmtree, copyfile

from elf.io import open_file
from torch_em.util.modelzoo import export_bioimageio_model, get_default_citations


def _load_data(input_):
    with open_file(input_, "r") as f:
        ds = f["raw"]
        shape = ds.shape
        halo = [16, 128, 128]
        bb = tuple(slice(sh // 2 - ha, sh // 2 + ha) for sh, ha in zip(shape, halo))
        raw = ds[bb]
    return raw

def _load_2D_data(input_):
    with open_file(input_, "r") as f:
        ds = f["raw"]
        shape = ds.shape
        halo = [128, 128]
        bb = tuple(slice(sh // 2 - ha, sh // 2 + ha) for sh, ha in zip(shape, halo))
        raw = ds[bb]
    return raw


def export_to_bioimageio(checkpoint, output_path, name):

    # Check if '2D' is in name
    if "2D" in name:
        print("It's a 2D model.")
        input_path = "/scratch-emmy/usr/nimsmuth/synapse_seg/data/vesicles_processed/01_hoi_maus_2020_incomplete/2D_example/tomogram-010_slice_75.h5"
        input_data = _load_2D_data(input_path).astype('float32')
        description = "Segment vesicles in 2d"
        tags = ["unet", "synaptic-vesicles", "instance-segmentation", "electron-microscopy", "2d"]
    else:
        print("It's a 3D model.")
        input_path = "/mnt/ceph-hdd/cold/nim00007/new_AZ_train_data/tem/20190524_09204_F4_SC_01_SP.h5"
        input_data = _load_data(input_path).astype('float32')
        description = "Segment AZ in 3d"
        tags = ["unet", "AZ", "instance-segmentation", "electron-microscopy", "3d"]

    # eventually we should refactor the citation logic
    cite = get_default_citations(model="AnisotropicUNet")

    doc = "Lorem ipsum sit dolor amet"

    export_bioimageio_model(
        checkpoint, output_path,
        input_data=input_data,
        name=name,
        description=description,
        authors=[{"name": "Sarah Muth; @SarahMuth"}],
        tags=tags,
        license="CC-BY-4.0",
        documentation=doc,
        git_repo="https://github.com/constantinpape/torch-em.git",
        cite=cite,
        input_optional_parameters=False,
        for_deepimagej=False,
    )


def main():
    default_output_root = "/scratch-emmy/usr/nimsmuth/synapse_seg/models"

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint")
    parser.add_argument("-o", "--output_root", default=default_output_root)
    parser.add_argument("-n", "--name")
    args = parser.parse_args()

    checkpoint = args.checkpoint
    output_root = args.output_root
    tmp_folder = os.path.join(output_root, "tmp")
    os.makedirs(tmp_folder, exist_ok=True)

    name = os.path.basename(checkpoint.rstrip("/"))
    tmp_path = os.path.join(tmp_folder, f"{name}.zip")
    export_to_bioimageio(checkpoint, tmp_path, name)

    #tmp_path = os.path.join(tmp_folder, f"{name}.zip")
    out_path = os.path.join(output_root, f"{name}.zip")
    assert os.path.exists(tmp_path), tmp_path
    copyfile(tmp_path, out_path)
    print("The model is exported to", out_path)
    rmtree(tmp_folder)


if __name__ == "__main__":
    main()
