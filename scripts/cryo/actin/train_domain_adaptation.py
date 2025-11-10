import os
from synapse_net.training.domain_adaptation import mean_teacher_adaptation
from pathlib import Path
from torch_em.data.sampler import MinForegroundSampler

def actin_adaptation_deepict2opto():
    PARENT_DIR = Path("/mnt/data1/sage/actin-segmentation/data/EMPIAR-12292/h5")
   
    all_paths = [str(p) for p in PARENT_DIR.glob("*.h5")]
    train_paths = all_paths[:10]
    val_paths = all_paths[10:12]

    patch_shape = (64, 384, 384)
    mean_teacher_adaptation(
        name="actin-adapted-deepict2opto-v1",
        unsupervised_train_paths=train_paths,
        unsupervised_val_paths=val_paths,
        raw_key="raw",
        patch_shape=patch_shape,
        n_iterations=int(25000),
        save_root="./output/experiment1/run2",
        source_checkpoint="./output/experiment1/run1/checkpoints/actin-deepict-v3",
        confidence_threshold=0.75,
        device=1,
    )


def actin_adaptation_opto2deepict_v1():
    PARENT_DIR = Path("/mnt/data1/sage/actin-segmentation/data/deepict/deepict_actin/")
    DATA_DIR = PARENT_DIR / "raw"
    BOUNDARY_MASK_DIR = PARENT_DIR / "boundary_masks"
    BACKGROUND_MASK_DIR = PARENT_DIR / "background_masks"
    
    data_paths = [str(p) for p in DATA_DIR.glob("*.mrc")]
    smpl_mask_paths = [str(p) for p in BOUNDARY_MASK_DIR.glob("*.mrc")] 
    bg_mask_paths = [str(p) for p in BACKGROUND_MASK_DIR.glob("*.mrc")]
    
    # train - 00004, val - 00011, test - 00012
    train_data_paths = [data_paths[2]]
    train_sample_mask_paths = [smpl_mask_paths[2]]
    train_background_mask_paths = [bg_mask_paths[2]]
    val_data_paths = [data_paths[0]]
    val_sample_mask_paths = [smpl_mask_paths[0]]

    print("train data paths:", train_data_paths)
    print("train sample paths:", train_sample_mask_paths)
    print("train background paths:", train_background_mask_paths)
    print("val data paths:", val_data_paths)
    print("val sample paths:", val_sample_mask_paths)

    patch_shape = (64, 384, 384)
    patch_sampler = MinForegroundSampler(min_fraction=0.95)

    mean_teacher_adaptation(
        name="actin-adapted-opto2deepict-v1",
        unsupervised_train_paths=train_data_paths,
        unsupervised_val_paths=val_data_paths,
        raw_key="raw",
        patch_shape=patch_shape,
        save_root="./output/experiment2/run2",
        source_checkpoint="./output/experiment2/run1/checkpoints/actin-opto-v1",
        confidence_threshold=0.75,
        batch_size=1,
        n_iterations=int(25000),
        train_sample_mask_paths=train_sample_mask_paths,
        val_sample_mask_paths=val_sample_mask_paths,
        train_background_mask_paths=train_background_mask_paths,
        patch_sampler=patch_sampler,
        device=2,
    )
 
def main():
    #actin_adaptation_deepict2opto()
    actin_adaptation_opto2deepict_v1()


if __name__ == "__main__":
    main()
