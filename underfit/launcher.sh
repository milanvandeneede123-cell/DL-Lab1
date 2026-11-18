#!/bin/bash
#SBATCH --qos=debug
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name="underfit_model"
#SBATCH --chdir=.
#SBATCH --output=underfit_%j.out
#SBATCH --error=underfit_%j.err

module purge
module load miniforge
module load cuda/12.6
source activate deepLearning

python train_underfit.py