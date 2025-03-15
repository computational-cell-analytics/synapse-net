from micro_sam.sam_annotator import image_folder_annotator

ROOT = "/home/pape/Work/data/fernandez-busnadiego/lamella/Filtered_lamellae"


def main():
    tile_shape = (384, 384)
    halo = (64, 64)
    image_folder_annotator(
        ROOT, "./segmentations",
        model_type="vit_b_lm", embedding_path="./embeddings",
        tile_shape=tile_shape, halo=halo,
        precompute_amg_state=True,
        skip_segmented=False,
    )


if __name__ == "__main__":
    main()
