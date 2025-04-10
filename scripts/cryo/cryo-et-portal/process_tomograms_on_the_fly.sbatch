#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=00-08:00:00
#SBATCH --nodes=1
#SBATCH -c 12

/scratch-grete/usr/nimcpape/software/mamba/envs/sam/bin/python process_tomograms_on_the_fly.py $@
