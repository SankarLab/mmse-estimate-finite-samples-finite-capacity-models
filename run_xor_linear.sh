#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -t 0-02:00:00   # time in d-hh:mm:ss
#SBATCH -G 1
#SBATCH -p htc      # partition 
#SBATCH -q public       # QOS
#SBATCH -o logs/slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e logs/slurm.%j.err # file to save job's STDERR (%j = JobId)

module load mamba/latest
source activate statml

python gmm_xor_linear.py -n $1 --delta $2 --n_modes $3 --seed $4 --sigma $5 --val_n $7 --results_path $8 --model_path $9
