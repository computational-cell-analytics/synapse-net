from common import apply_cryo_vesnet


def main():
    input_folder = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/wichmann/extracted/endbulb_of_held/Automatische_Segmentierung_Dataset_Validierung"  # noqa
    output_folder = "./predictions/endbulb"

    # Resolution in Angstrom in XYZ
    # The two tomograms have a different resolution.
    resolution = (17.48,) * 3
    apply_cryo_vesnet(
        input_folder, output_folder,
        pattern="*.h5", input_key="raw",
        mask_folder=input_folder, mask_key="labels/endbulb",
        resolution=resolution
    )


if __name__ == "__main__":
    main()
