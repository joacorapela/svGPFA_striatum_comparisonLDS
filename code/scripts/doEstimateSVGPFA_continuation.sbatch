#!/bin/bash

#SBATCH --job-name=doEstimateSVGPFA_continuation
#SBATCH --output=../../logs/slurmOutputs/doEstimateSVGPFA_continuation_%j.out
#SBATCH --error=../../logs/slurmOutputs/doEstimateSVGPFA_continuation_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=a100
#SBATCH --mem=150G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.rapela@ucl.ac.uk

# Activate your environment if necessary
# conda activate myenv

if [ $# -ne 1 ]; then
    echo "Usage: sbatch $0 <est_res_number>"
    exit 1
fi

in_est_res_number=$1

# Execute your script with arguments here:
python -u doEstimateSVGPFA_continuation.py --in_est_res_number=$in_est_res_number
