#! /bin/bash
#SBATCH -c 4 #4 #8
#SBATCH --mem 120G #120G #32G #64G #256G
#SBATCH -p grete:shared #grete:shared #grete-h100:shared
#SBATCH -t 2:00:00 #6:00:00 #48:00:00 
#SBATCH -G A100:1 #V100:1 #2 #A100:1  #gtx1080:2 #v100:1 #H100:1
#SBATCH --output=/user/muth9/u12095/synapse-net/slurm_revision/slurm-%j.out 
#SBATCH -A nim00007 #SBATCH --constraint 80gb 

source ~/.bashrc
conda activate synapse-net
python scripts/cooper/revision/surface_dice.py -i /mnt/ceph-hdd/cold/nim00007/AZ_prediction_new/endbulb_of_held/ \
 -gt /mnt/ceph-hdd/cold/nim00007/new_AZ_train_data/endbulb_of_held/ -v 7 --global_eval