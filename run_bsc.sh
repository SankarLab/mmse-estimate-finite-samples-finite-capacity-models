#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -t 0-01:00:00   # time in d-hh:mm:ss
#SBATCH -G 1
#SBATCH -p htc      # partition 
#SBATCH -q public       # QOS
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)

module load mamba/latest
source activate statml

python bsc_mmse.py --sigma $1 -p $2 -n $3 --seed $4 --p_n $5
