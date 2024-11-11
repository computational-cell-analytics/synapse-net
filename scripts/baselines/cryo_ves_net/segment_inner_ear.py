from common import apply_cryo_vesnet


def main():
    input_folder = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/moser/vesicle_gt"  # noqa
    output_folder = "./predictions/inner_ear"

    # Resolution in Angstrom in XYZ
    # The two tomograms have a different resolution.
    resolution = (11.8, 11.8, 11.88)
    apply_cryo_vesnet(
        input_folder, output_folder,
        pattern="*.h5", input_key="raw",
        resolution=resolution
    )


if __name__ == "__main__":
    main()
