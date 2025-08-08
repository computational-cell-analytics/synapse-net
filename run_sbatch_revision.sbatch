#! /bin/bash
#SBATCH -c 4 #4 #8
#SBATCH --mem 256G #120G #32G #64G #256G
#SBATCH -p grete:shared #grete:shared #grete-h100:shared
#SBATCH -t 4:00:00 #6:00:00 #48:00:00 
#SBATCH -G A100:1 #V100:1 #2 #A100:1  #gtx1080:2 #v100:1 #H100:1
#SBATCH --output=/user/muth9/u12095/synapse-net/slurm_revision/slurm-%j.out 
#SBATCH -A nim00007 
#SBATCH --constraint 80gb 

source ~/.bashrc
conda activate synapse-net
python /user/muth9/u12095/synapse-net/scripts/cooper/vesicle_segmentation_h5.py \
 -i /mnt/ceph-hdd/cold/nim00007/LDCVs_matea/testset \
 -o /mnt/ceph-hdd/cold/nim00007/LDCVs_matea/segmentation/v2 \
 -m /mnt/ceph-hdd/cold/nim00007/models/LDCVs_matea/checkpoints/3D-vesicles-model-LDCV-v2/best.pt \
 --include_boundary