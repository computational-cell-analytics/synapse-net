import argparse

from synaptic_reconstruction.imod.to_imod import write_segmentation_to_imod


def export_mask_to_imod(args):
    # Test script
    # write_segmentation_to_imod(
    #     "synapse-examples/36859_J1_66K_TS_CA3_PS_26_rec_2Kb1dawbp_crop.mrc",
    #     "synapse-examples/36859_J1_66K_TS_CA3_PS_26_rec_2Kb1dawbp_crop_mitos.tif",
    #     "synapse-examples/mito.mod"
    # )
    write_segmentation_to_imod(args.input_path, args.segmentation_path, args.output_path)


def main():
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    parser.add_argument(
        "-i", "--input_path", required=True,
        help="The filepath to the mrc file containing the data."
    )
    parser.add_argument(
        "-s", "--segmentation_path", required=True,
        help="The filepath to the tif file containing the segmentation."
    )
    parser.add_argument(
        "-o", "--output_path", required=True,
        help="The filepath where the mod file with contours will be saved."
    )
    args = parser.parse_args()
    export_mask_to_imod(args)


if __name__ == "__main__":
    main()