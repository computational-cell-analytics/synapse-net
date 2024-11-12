import os
from glob import glob

import h5py
import napari
from magicgui import magicgui


def run_annotation(input_path, output_path):
    with h5py.File(input_path, "r") as f:
        raw = f["raw"][:]
        seg = f["labels/compartments"][:]

    v = napari.Viewer()
    v.add_image(raw)
    v.add_labels(seg)

    @magicgui(call_button="Save Annotations")
    def save_annotations():
        seg = v.layers["seg"].data

        if os.path.exists(output_path):
            with h5py.File(output_path, "a") as f:
                f["labels/compartments"][:] = seg
        else:
            with h5py.File(output_path, "a") as f:
                f.create_dataset("raw", data=raw, compression="gzip")
                f.create_dataset("labels/compartments", data=seg, compression="gzip")

    v.window.add_dock_widget(save_annotations)

    napari.run()


def main():
    inputs = sorted(glob("./predictions/**/*.h5", recursive=True))

    output_folder = "./annotations"

    for input_path in inputs:
        ds_name, fname = os.path.split(input_path)
        ds_name = os.path.split(ds_name)[1]
        ds_folder = os.path.join(output_folder, ds_name)
        output_path = os.path.join(ds_folder, fname)

        if os.path.exists(output_path):
            print("Skipping annotations for", output_path)
            continue

        os.makedirs(ds_folder, exist_ok=True)
        run_annotation(input_path, output_path)


if __name__ == "__main__":
    main()
