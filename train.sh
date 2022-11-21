#!/usr/bin/env bash

#SBATCH --job-name=music
#SBATCH --partition=teach_gpu
#SBATCH --nodes=1
#SBATCH -o ./log.out # STDOUT out
#SBATCH -e ./log.err # STDERR out
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=64GB

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2021-3.8.8-cuda-11.1-pytorch"

python train.py -vf 1 -e 200