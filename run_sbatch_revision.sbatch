#! /bin/bash
#SBATCH -c 4 #4 #8
#SBATCH --mem 120G #120G #32G #64G #256G
#SBATCH -p grete:shared #grete:shared #grete-h100:shared
#SBATCH -t 12:00:00 #6:00:00 #48:00:00 
#SBATCH -G A100:1 #V100:1 #2 #A100:1  #gtx1080:2 #v100:1 #H100:1
#SBATCH --output=/user/muth9/u12095/synapse-net/slurm_revision/slurm-%j.out 
#SBATCH -A nim00007 #SBATCH --constraint 80gb 

source ~/.bashrc
conda activate synapse-net
python /user/muth9/u12095/synapse-net/scripts/cooper/revision/updated_data_analysis/run_data_analysis.py \
 -i /mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/20241102_TOMO_DATA_Imig2014/exported/Munc13DKO/ \
 -o /mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/20241102_TOMO_DATA_Imig2014/afterRevision_analysis --store